//! Behavioural tests for the file built-ins. These had no coverage before;
//! they pin the contract (workspace-relative paths, line slicing, exact-string
//! replacement, permission gating) so regressions are caught.

use agent_core::tool::{Permissions, Tool, ToolContext, ToolError};
use agent_tools::builtin::file_edit::FileEditTool;
use agent_tools::builtin::file_read::FileReadTool;
use serde_json::json;

fn ctx_at(dir: &std::path::Path) -> ToolContext {
    ToolContext::new(dir.to_path_buf())
}

#[tokio::test]
async fn file_read_resolves_relative_and_slices_lines() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join("notes.txt"), "l1\nl2\nl3\nl4\n").unwrap();
    let ctx = ctx_at(tmp.path());

    // Full read mentions the total line count and the content.
    let out = FileReadTool
        .invoke(json!({ "path": "notes.txt" }), &ctx)
        .await
        .unwrap();
    assert!(!out.is_error);
    assert!(out.text.contains("l1") && out.text.contains("l4"));

    // offset/limit slice: skip 1, take 2 → l2,l3 only.
    let out = FileReadTool
        .invoke(json!({ "path": "notes.txt", "offset": 1, "limit": 2 }), &ctx)
        .await
        .unwrap();
    assert!(out.text.contains("l2") && out.text.contains("l3"));
    assert!(!out.text.contains("l4"));
}

#[tokio::test]
async fn file_read_missing_file_is_soft_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    let ctx = ctx_at(tmp.path());
    let out = FileReadTool
        .invoke(json!({ "path": "nope.txt" }), &ctx)
        .await
        .unwrap();
    assert!(out.is_error, "missing file should be a soft error outcome");
}

#[tokio::test]
async fn file_read_denied_without_permission() {
    let tmp = tempfile::TempDir::new().unwrap();
    let perms = Permissions { allow_read: false, ..Permissions::default() };
    let ctx = ToolContext::new(tmp.path().to_path_buf()).with_permissions(perms);
    let err = FileReadTool
        .invoke(json!({ "path": "whatever.txt" }), &ctx)
        .await
        .unwrap_err();
    assert!(matches!(err, ToolError::PermissionDenied(_)));
}

#[tokio::test]
async fn file_edit_exact_replacement() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("src.txt");
    std::fs::write(&path, "hello world\n").unwrap();
    let ctx = ctx_at(tmp.path());

    let out = FileEditTool
        .invoke(
            json!({ "path": "src.txt", "old_string": "world", "new_string": "rust" }),
            &ctx,
        )
        .await
        .unwrap();
    assert!(!out.is_error);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello rust\n");
}

#[tokio::test]
async fn file_edit_missing_old_string_is_soft_error() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join("src.txt"), "abc\n").unwrap();
    let ctx = ctx_at(tmp.path());
    let out = FileEditTool
        .invoke(
            json!({ "path": "src.txt", "old_string": "xyz", "new_string": "q" }),
            &ctx,
        )
        .await
        .unwrap();
    assert!(out.is_error);
}

#[tokio::test]
async fn file_edit_multiple_matches_requires_replace_all() {
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join("src.txt");
    std::fs::write(&path, "x x x\n").unwrap();
    let ctx = ctx_at(tmp.path());

    // Ambiguous single replace → soft error, file untouched.
    let out = FileEditTool
        .invoke(
            json!({ "path": "src.txt", "old_string": "x", "new_string": "y" }),
            &ctx,
        )
        .await
        .unwrap();
    assert!(out.is_error);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "x x x\n");

    // replace_all → succeeds.
    let out = FileEditTool
        .invoke(
            json!({ "path": "src.txt", "old_string": "x", "new_string": "y", "replace_all": true }),
            &ctx,
        )
        .await
        .unwrap();
    assert!(!out.is_error);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "y y y\n");
}
