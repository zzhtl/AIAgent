//! `propose_rule` / `propose_skill` — let the agent push candidate rules /
//! skills onto the human-review queue. Approvals happen via the CLI; the
//! tools alone never modify the user's `~/.config/agent/{rules,skills}/`.

use std::time::{SystemTime, UNIX_EPOCH};

use agent_core::evolution::{Candidate, CandidateKind};
use agent_core::tool::{Tool, ToolContext, ToolError, ToolOutcome, ToolResult};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

async fn enqueue(
    ctx: &ToolContext,
    kind: CandidateKind,
    name: String,
    rationale: String,
    body: String,
) -> ToolResult<ToolOutcome> {
    let queue = ctx
        .candidate_queue
        .as_ref()
        .ok_or_else(|| ToolError::ExecutionFailed("no candidate queue available".into()))?;
    let id = Uuid::new_v4().to_string();
    let candidate = Candidate {
        id: id.clone(),
        kind,
        name: name.clone(),
        rationale,
        body,
        created_at: now_secs(),
    };
    queue
        .enqueue(candidate)
        .await
        .map_err(|e| ToolError::ExecutionFailed(e.to_string()))?;
    Ok(ToolOutcome::ok(format!("queued {name} (id: {id})")))
}

#[derive(Default)]
pub struct ProposeRuleTool;

#[derive(Deserialize)]
struct RuleArgs {
    name: String,
    body: String,
    rationale: String,
}

#[async_trait]
impl Tool for ProposeRuleTool {
    fn name(&self) -> &str {
        "propose_rule"
    }

    fn description(&self) -> &str {
        "Propose a new global rule (markdown) to be reviewed by the user. \
         The rule is **not** applied automatically — the user runs \
         `agent evolution review` and `agent evolution apply <id>` to \
         install it. Use this when you notice a repeated correction or \
         convention worth making permanent."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name":      { "type": "string", "description": "Short title." },
                "body":      { "type": "string", "description": "Full rule body (markdown)." },
                "rationale": { "type": "string", "description": "Why this rule was proposed." }
            },
            "required": ["name", "body", "rationale"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        let args: RuleArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "propose_rule".into(),
            detail: e.to_string(),
        })?;
        enqueue(ctx, CandidateKind::Rule, args.name, args.rationale, args.body).await
    }
}

#[derive(Default)]
pub struct ProposeSkillTool;

#[derive(Deserialize)]
struct SkillArgs {
    name: String,
    body: String,
    rationale: String,
    #[serde(default)]
    triggers: Vec<String>,
}

#[async_trait]
impl Tool for ProposeSkillTool {
    fn name(&self) -> &str {
        "propose_skill"
    }

    fn description(&self) -> &str {
        "Propose a new skill (a markdown capability pack with trigger \
         keywords). Queued for user review; not applied until they accept \
         it via `agent evolution apply`."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name":      { "type": "string" },
                "body":      { "type": "string", "description": "Skill body in markdown." },
                "rationale": { "type": "string" },
                "triggers":  { "type": "array", "items": { "type": "string" } }
            },
            "required": ["name", "body", "rationale"]
        })
    }

    async fn invoke(&self, args: Value, ctx: &ToolContext) -> ToolResult<ToolOutcome> {
        let args: SkillArgs = serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments {
            tool: "propose_skill".into(),
            detail: e.to_string(),
        })?;
        // Embed triggers as frontmatter in the body so `evolution apply`
        // writes a complete skill file straight from the candidate.
        let mut body = String::new();
        body.push_str("---\n");
        body.push_str(&format!("name: {}\n", args.name));
        if !args.triggers.is_empty() {
            body.push_str("triggers:\n");
            for t in &args.triggers {
                body.push_str(&format!("  - {t}\n"));
            }
        }
        body.push_str("---\n\n");
        body.push_str(args.body.trim_end());
        body.push('\n');
        enqueue(ctx, CandidateKind::Skill, args.name, args.rationale, body).await
    }
}
