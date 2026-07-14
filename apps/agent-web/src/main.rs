//! agent-web: authenticated HTTP / SSE entry for the shared agent runtime.
//!
//! `GET /health` is always public. `POST /chat` streams `AgentEvent` JSON over
//! SSE and is protected by `AGENT_WEB_TOKEN` when configured. Omitted session
//! ids are generated server-side; a per-session mutex serialises overlapping
//! turns while unrelated sessions remain concurrent.

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agent_config::AgentConfig;
use agent_core::tool::Permissions;
use agent_core::{
    Agent, AgentEvent, Message, SessionId, SessionStore, SessionStoreError, UserInput,
};
use agent_memory::SqliteSessionStore;
use agent_runtime::{build_runtime, open_session_store, BuildOptions};
use anyhow::{anyhow, Context, Result};
use axum::extract::{Request, State};
use axum::http::header::{AUTHORIZATION, WWW_AUTHENTICATE};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::stream::{BoxStream, Stream};
use futures::StreamExt;
use serde::Deserialize;
use tokio::sync::{Mutex, OwnedMutexGuard};

type SessionHistory = Arc<Mutex<Vec<Message>>>;
type Histories = Arc<Mutex<HashMap<String, SessionHistory>>>;

#[derive(Clone)]
struct AppState {
    agent: Arc<Agent>,
    histories: Histories,
    session_store: Option<Arc<SqliteSessionStore>>,
}

#[derive(Clone)]
struct AuthState {
    token: Arc<str>,
}

#[derive(Debug, Deserialize)]
struct ChatReq {
    input: String,
    #[serde(default)]
    session: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    agent_telemetry::init_default();
    let config = AgentConfig::load(None).map_err(|error| anyhow!("config: {error}"))?;
    let token = non_empty_env("AGENT_WEB_TOKEN");
    let addr = non_empty_env("AGENT_WEB_ADDR").unwrap_or_else(|| config.web.addr.clone());
    let resolved_addresses: Vec<SocketAddr> = tokio::net::lookup_host(&addr)
        .await
        .with_context(|| format!("resolve {addr}"))?
        .collect();
    validate_bind_addresses(
        &resolved_addresses,
        token.is_some(),
        config.web.allow_unauthenticated,
    )?;

    let options = BuildOptions {
        model_override: non_empty_env("AGENT_WEB_MODEL"),
        workspace: Some(std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        permissions_override: Some(web_permissions(&config)),
        tool_policy_override: config.web.tool_policy.clone(),
        env_probe_fallback: true,
        ..BuildOptions::default()
    };
    let runtime = build_runtime(&config, options).await.context("agent-web init")?;
    for warning in &runtime.warnings {
        tracing::warn!("{warning}");
    }
    let session_store = if config.web.persist_sessions {
        Some(Arc::new(open_session_store(&runtime.config_dir).await?))
    } else {
        None
    };
    let state = AppState {
        agent: Arc::new(runtime.agent),
        histories: Arc::new(Mutex::new(HashMap::new())),
        session_store,
    };
    let app = app_router(state, token);

    let listener = tokio::net::TcpListener::bind(resolved_addresses.as_slice())
        .await
        .with_context(|| format!("bind {addr}"))?;
    let bound = listener.local_addr().context("read bound address")?;
    tracing::info!("agent-web listening on http://{bound}");
    axum::serve(listener, app).await.context("serve")?;
    Ok(())
}

fn app_router(state: AppState, token: Option<String>) -> Router {
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/chat", post(chat))
        .with_state(state);
    if let Some(token) = token {
        app.layer(middleware::from_fn_with_state(
            AuthState { token: Arc::from(token) },
            require_bearer,
        ))
    } else {
        app
    }
}

async fn require_bearer(
    State(auth): State<AuthState>,
    request: Request,
    next: Next,
) -> Response {
    if request.uri().path() == "/health" {
        return next.run(request).await;
    }
    let supplied = bearer_token(request.headers());
    if supplied.is_some_and(|token| constant_time_eq(auth.token.as_bytes(), token.as_bytes())) {
        return next.run(request).await;
    }
    let mut response = StatusCode::UNAUTHORIZED.into_response();
    response
        .headers_mut()
        .insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(AUTHORIZATION)?
        .to_str()
        .ok()?
        .strip_prefix("Bearer ")
}

fn constant_time_eq(expected: &[u8], supplied: &[u8]) -> bool {
    let mut difference = expected.len() ^ supplied.len();
    let max_len = expected.len().max(supplied.len());
    for index in 0..max_len {
        let left = expected.get(index).copied().unwrap_or(0);
        let right = supplied.get(index).copied().unwrap_or(0);
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

async fn chat(
    State(state): State<AppState>,
    Json(request): Json<ChatReq>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (session, announce_session) = match request.session {
        Some(session) => (session, false),
        None => (uuid::Uuid::new_v4().to_string(), true),
    };
    let (mut history_guard, created) = lock_session(&state.histories, &session).await;
    if created {
        seed_history(
            &mut history_guard,
            state.session_store.as_deref(),
            &SessionId::from(session.as_str()),
        )
        .await;
    }
    let history = history_guard.clone();
    let sid = SessionId::from(session.as_str());
    let cancel = Arc::new(AtomicBool::new(false));
    let stream = state
        .agent
        .run_cancellable(sid, history, UserInput::new(request.input), cancel.clone());
    let events = stream_events(
        stream,
        session,
        announce_session,
        history_guard,
        state.session_store,
        cancel,
    )
    .map(|json| Ok(Event::default().data(json)));
    Sse::new(events).keep_alive(KeepAlive::default())
}

async fn lock_session(
    histories: &Histories,
    session: &str,
) -> (OwnedMutexGuard<Vec<Message>>, bool) {
    let mut all = histories.lock().await;
    if let Some(history) = all.get(session).cloned() {
        drop(all);
        return (history.lock_owned().await, false);
    }

    let history = Arc::new(Mutex::new(Vec::new()));
    let guard = history.clone().lock_owned().await;
    all.insert(session.to_string(), history);
    drop(all);
    (guard, true)
}

async fn seed_history(
    history: &mut Vec<Message>,
    store: Option<&SqliteSessionStore>,
    sid: &SessionId,
) {
    let Some(store) = store else {
        return;
    };
    match store.load_messages(sid).await {
        Ok(messages) => history.extend(messages),
        Err(SessionStoreError::NotFound(_)) => {
            if let Err(error) = store.ensure_session(sid, None).await {
                tracing::warn!(session = %sid, %error, "failed to create persistent session");
            }
        }
        Err(error) => {
            tracing::warn!(session = %sid, %error, "failed to load persistent session");
        }
    }
}

fn stream_events(
    stream: BoxStream<'static, AgentEvent>,
    session: String,
    announce_session: bool,
    mut history: OwnedMutexGuard<Vec<Message>>,
    session_store: Option<Arc<SqliteSessionStore>>,
    cancel: Arc<AtomicBool>,
) -> impl Stream<Item = String> {
    async_stream::stream! {
        let mut cancel_on_drop = CancelOnDrop::new(cancel);
        let mut stream = stream;
        if announce_session {
            yield serde_json::json!({ "kind": "session", "session": session }).to_string();
        }
        while let Some(event) = stream.next().await {
            if let AgentEvent::Done { transcript_delta, .. } = &event {
                history.extend(transcript_delta.clone());
                if let Some(store) = session_store.as_ref() {
                    let sid = SessionId::from(session.as_str());
                    if let Err(error) = store.append_messages(&sid, transcript_delta).await {
                        tracing::warn!(session = %sid, %error, "failed to persist messages");
                    }
                }
                cancel_on_drop.disarm();
            }
            yield serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string());
        }
    }
}

struct CancelOnDrop {
    cancel: Arc<AtomicBool>,
    armed: bool,
}

impl CancelOnDrop {
    fn new(cancel: Arc<AtomicBool>) -> Self {
        Self { cancel, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.cancel.store(true, Ordering::Relaxed);
        }
    }
}

fn web_permissions(config: &AgentConfig) -> Permissions {
    config.web.permissions.as_ref().map_or_else(
        || Permissions {
            allow_read: true,
            allow_write: false,
            allow_shell: false,
            allow_network: false,
            max_runtime_secs: 60,
        },
        |permissions| permissions.to_runtime(),
    )
}

fn validate_bind_addresses(
    addresses: &[SocketAddr],
    has_token: bool,
    allow_unauthenticated: bool,
) -> Result<()> {
    if addresses.is_empty() {
        return Err(anyhow!("web address resolved to no socket addresses"));
    }
    if has_token
        || allow_unauthenticated
        || addresses.iter().all(|address| address.ip().is_loopback())
    {
        return Ok(());
    }
    Err(anyhow!(
        "refusing unauthenticated non-loopback web bind; set AGENT_WEB_TOKEN or web.allow_unauthenticated=true"
    ))
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{
        ChatRequest, LlmEvent, LlmEventStream, LlmProvider, LlmResult, ProviderCapabilities,
        StopReason,
    };
    use async_trait::async_trait;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    struct DoneProvider;

    #[async_trait]
    impl LlmProvider for DoneProvider {
        fn name(&self) -> &str {
            "test"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities { streaming: true, ..Default::default() }
        }

        async fn chat_stream(&self, _request: ChatRequest) -> LlmResult<LlmEventStream> {
            Ok(futures::stream::iter(vec![
                Ok(LlmEvent::TextDelta { delta: "ok".into() }),
                Ok(LlmEvent::End(StopReason::EndTurn)),
            ])
            .boxed())
        }
    }

    fn test_state() -> AppState {
        let agent = Agent::builder()
            .with_llm(Arc::new(DoneProvider))
            .with_model("test")
            .build()
            .expect("test agent");
        AppState {
            agent: Arc::new(agent),
            histories: Arc::new(Mutex::new(HashMap::new())),
            session_store: None,
        }
    }

    fn chat_request(session: Option<&str>) -> HttpRequest<Body> {
        let mut body = serde_json::json!({ "input": "hello" });
        if let Some(session) = session {
            body["session"] = serde_json::json!(session);
        }
        HttpRequest::post("/chat")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .expect("request")
    }

    #[tokio::test]
    async fn generated_session_is_first_sse_event() {
        let response = app_router(test_state(), None)
            .oneshot(chat_request(None))
            .await
            .expect("response");
        let body = response.into_body().collect().await.expect("body").to_bytes();
        let text = String::from_utf8(body.to_vec()).expect("utf8");
        let first = text.lines().find(|line| line.starts_with("data: ")).expect("data");
        let value: serde_json::Value =
            serde_json::from_str(first.trim_start_matches("data: ")).expect("session json");
        assert_eq!(value["kind"], "session");
        assert!(value["session"].as_str().is_some_and(|id| !id.is_empty()));
    }

    #[tokio::test]
    async fn bearer_protects_chat_but_not_health() {
        let app = app_router(test_state(), Some("secret".into()));
        let health = app
            .clone()
            .oneshot(HttpRequest::get("/health").body(Body::empty()).expect("health"))
            .await
            .expect("health response");
        assert_eq!(health.status(), StatusCode::OK);

        let denied = app.clone().oneshot(chat_request(Some("s1"))).await.expect("denied");
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(denied.headers().get(WWW_AUTHENTICATE), Some(&HeaderValue::from_static("Bearer")));

        let mut allowed = chat_request(Some("s1"));
        allowed
            .headers_mut()
            .insert(AUTHORIZATION, HeaderValue::from_static("Bearer secret"));
        let allowed = app.oneshot(allowed).await.expect("allowed");
        assert_eq!(allowed.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn same_session_requests_are_serialised_without_lost_history() {
        let state = test_state();
        let histories = state.histories.clone();
        let app = app_router(state, None);
        let first = app
            .clone()
            .oneshot(chat_request(Some("shared")))
            .await
            .expect("first response");
        let second_app = app.clone();
        let mut second = tokio::spawn(async move {
            second_app.oneshot(chat_request(Some("shared"))).await
        });
        assert!(tokio::time::timeout(std::time::Duration::from_millis(20), &mut second)
            .await
            .is_err());

        first.into_body().collect().await.expect("first body");
        let second = tokio::time::timeout(std::time::Duration::from_secs(1), second)
            .await
            .expect("second unblocked")
            .expect("second task")
            .expect("second response");
        second.into_body().collect().await.expect("second body");

        let history = histories.lock().await.get("shared").cloned().expect("history");
        assert_eq!(history.lock().await.len(), 4);
    }

    #[tokio::test]
    async fn dropping_stream_signals_cancel() {
        let history = Arc::new(Mutex::new(Vec::new())).lock_owned().await;
        let cancel = Arc::new(AtomicBool::new(false));
        let events = futures::stream::pending().boxed();
        let mut stream = Box::pin(stream_events(
            events,
            "s1".into(),
            true,
            history,
            None,
            cancel.clone(),
        ));
        assert!(stream.next().await.is_some());
        drop(stream);
        assert!(cancel.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn persistent_history_survives_state_rebuild() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = Arc::new(open_session_store(dir.path()).await.expect("store"));
        let sid = SessionId::from("persisted");
        store.ensure_session(&sid, None).await.expect("ensure");
        store
            .append_messages(&sid, &[Message::user("old")])
            .await
            .expect("append old");

        let histories: Histories = Arc::new(Mutex::new(HashMap::new()));
        let (mut history, created) = lock_session(&histories, sid.as_str()).await;
        assert!(created);
        seed_history(&mut history, Some(&store), &sid).await;
        assert_eq!(history.len(), 1);
    }

    #[test]
    fn non_loopback_without_auth_is_rejected() {
        let public: SocketAddr = "0.0.0.0:8787".parse().expect("address");
        let local: SocketAddr = "127.0.0.1:8787".parse().expect("address");
        assert!(validate_bind_addresses(&[public], false, false).is_err());
        assert!(validate_bind_addresses(&[public], true, false).is_ok());
        assert!(validate_bind_addresses(&[public], false, true).is_ok());
        assert!(validate_bind_addresses(&[local], false, false).is_ok());
    }

    #[test]
    fn web_defaults_to_read_only_permissions() {
        let config = AgentConfig::default();
        let permissions = web_permissions(&config);
        assert!(permissions.allow_read);
        assert!(!permissions.allow_write);
        assert!(!permissions.allow_shell);
        assert!(!permissions.allow_network);
        assert_eq!(permissions.max_runtime_secs, 60);
    }
}
