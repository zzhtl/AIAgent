//! Integration tests for `Augmenter::tool_whitelist_for` — the bridge that
//! turns a triggered skill's `tools_allowed` front-matter into the runtime's
//! per-turn tool whitelist.

use std::path::Path;

use agent_core::prompt::PromptProvider;
use agent_skills::{Augmenter, RuleSet, SkillRegistry};

fn write_skill(dir: &Path, file: &str, content: &str) {
    std::fs::write(dir.join(file), content).unwrap();
}

fn augmenter_from(dir: &Path) -> Augmenter {
    let skills = SkillRegistry::load_dir(dir).expect("load skills");
    Augmenter::new(RuleSet::default(), skills)
}

#[tokio::test]
async fn single_skill_drives_whitelist_when_triggered() {
    let tmp = tempfile::TempDir::new().unwrap();
    write_skill(
        tmp.path(),
        "review.md",
        "---\nname: review\ntriggers:\n  - review\ntools_allowed:\n  - file_read\n  - bash\n---\nbody",
    );
    let aug = augmenter_from(tmp.path());

    // Triggered → union of the skill's tools_allowed.
    let wl = aug.tool_whitelist_for("please review this code").await;
    assert_eq!(wl, Some(vec!["file_read".to_string(), "bash".to_string()]));

    // Not triggered → no restriction.
    assert_eq!(aug.tool_whitelist_for("just say hi").await, None);
}

#[tokio::test]
async fn multiple_triggered_skills_union_their_lists() {
    let tmp = tempfile::TempDir::new().unwrap();
    write_skill(
        tmp.path(),
        "a.md",
        "---\nname: a\ntriggers:\n  - deploy\ntools_allowed:\n  - bash\n---\nbody",
    );
    write_skill(
        tmp.path(),
        "b.md",
        "---\nname: b\ntriggers:\n  - deploy\ntools_allowed:\n  - fetch\n  - bash\n---\nbody",
    );
    let aug = augmenter_from(tmp.path());

    let mut wl = aug.tool_whitelist_for("deploy now").await.expect("whitelist");
    wl.sort();
    assert_eq!(wl, vec!["bash".to_string(), "fetch".to_string()]);
}

#[tokio::test]
async fn triggered_skill_without_tools_allowed_does_not_restrict() {
    let tmp = tempfile::TempDir::new().unwrap();
    write_skill(
        tmp.path(),
        "doc.md",
        "---\nname: doc\ntriggers:\n  - explain\n---\nbody",
    );
    let aug = augmenter_from(tmp.path());

    // Skill matches but declares no tools_allowed → None (unrestricted).
    assert_eq!(aug.tool_whitelist_for("explain this").await, None);
}
