//! Small text helpers shared across the workspace.
//!
//! Centralised because half the call sites were doing `&s[..N]` byte slicing
//! on UTF-8 strings, which panics on multi-byte boundaries (CJK, emoji, …).

/// Take at most `max_chars` characters. No ellipsis is added when truncation
/// happens — use [`truncate_with_ellipsis`] for that.
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        s.chars().take(max_chars).collect()
    }
}

/// Truncate to `max_chars` and append `…` if anything was dropped.
pub fn truncate_with_ellipsis(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let head: String = s.chars().take(max_chars).collect();
        format!("{head}…")
    }
}

/// First non-empty line trimmed and truncated. Returns an empty string when
/// the input is blank.
pub fn first_line_truncated(body: &str, max_chars: usize) -> String {
    let line = body.lines().next().unwrap_or("").trim();
    truncate_with_ellipsis(line, max_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_truncation() {
        assert_eq!(truncate_with_ellipsis("hello", 5), "hello");
        assert_eq!(truncate_with_ellipsis("hello world", 5), "hello…");
    }

    #[test]
    fn cjk_boundary_does_not_panic() {
        let s = "中文测试一二三四五六七八九十";
        let out = truncate_with_ellipsis(s, 4);
        assert_eq!(out, "中文测试…");
    }

    #[test]
    fn first_line_handles_empty() {
        assert_eq!(first_line_truncated("", 10), "");
        assert_eq!(first_line_truncated("\n\nfoo", 10), "");
    }
}
