use agent_core::{Message, Role, SessionStore, TokenUsage};
use agent_memory::SqliteSessionStore;

#[tokio::test]
async fn create_append_load_roundtrip() {
    let tmp = tempfile::TempDir::new().unwrap();
    let db = tmp.path().join("test.db");
    let store = SqliteSessionStore::open(&db).await.unwrap();

    let sid = store.create_session(Some("test session")).await.unwrap();
    store
        .append_messages(
            &sid,
            &[Message::user("hello"), Message::assistant("hi back")],
        )
        .await
        .unwrap();

    let msgs = store.load_messages(&sid).await.unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0].role, Role::User);
    assert_eq!(msgs[0].text(), "hello");
    assert_eq!(msgs[1].role, Role::Assistant);
    assert_eq!(msgs[1].text(), "hi back");
}

#[tokio::test]
async fn list_sessions_includes_all_with_counts() {
    let tmp = tempfile::TempDir::new().unwrap();
    let db = tmp.path().join("test.db");
    let store = SqliteSessionStore::open(&db).await.unwrap();

    let _a = store.create_session(Some("first")).await.unwrap();
    let b = store.create_session(Some("second")).await.unwrap();
    store
        .append_messages(&b, &[Message::user("ping")])
        .await
        .unwrap();

    let list = store.list_sessions(10).await.unwrap();
    assert_eq!(list.len(), 2);
    let b_summary = list.iter().find(|s| s.id == b).expect("b present");
    assert_eq!(b_summary.message_count, 1);
}

#[tokio::test]
async fn usage_aggregation_sums_rounds() {
    let tmp = tempfile::TempDir::new().unwrap();
    let db = tmp.path().join("test.db");
    let store = SqliteSessionStore::open(&db).await.unwrap();
    let sid = store.create_session(None).await.unwrap();

    store
        .record_usage(
            &sid,
            "gpt-4o-mini",
            TokenUsage { prompt_tokens: 100, completion_tokens: 50, cached_tokens: 0 },
            0.01,
        )
        .await
        .unwrap();
    store
        .record_usage(
            &sid,
            "gpt-4o-mini",
            TokenUsage { prompt_tokens: 200, completion_tokens: 80, cached_tokens: 10 },
            0.02,
        )
        .await
        .unwrap();

    let summary = store.session_usage(&sid).await.unwrap();
    assert_eq!(summary.prompt_tokens, 300);
    assert_eq!(summary.completion_tokens, 130);
    assert_eq!(summary.cached_tokens, 10);
    assert!((summary.cost_estimate_usd - 0.03).abs() < 1e-9);
}

#[tokio::test]
async fn load_missing_session_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    let db = tmp.path().join("test.db");
    let store = SqliteSessionStore::open(&db).await.unwrap();
    let missing = agent_core::SessionId::from("does-not-exist");
    let err = store.load_messages(&missing).await.err().unwrap();
    assert!(matches!(err, agent_core::SessionStoreError::NotFound(_)));
}
