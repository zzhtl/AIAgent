//! Tiny SSE line buffer shared by all providers.
//!
//! Servers send arbitrary `Bytes` chunks; SSE frames are split on `\n` and
//! each line is UTF-8. We must NOT decode chunks individually with
//! `String::from_utf8_lossy(&chunk)` because a multi-byte character can
//! straddle a chunk boundary (the upstream symptom is `�` showing up in
//! Chinese / emoji tool arguments).
//!
//! Approach: accumulate raw bytes, find `\n` byte-wise (safe because all
//! multi-byte UTF-8 continuation bytes have the high bit set, so `0x0A` only
//! appears at code-point boundaries), then decode each completed line.

pub struct SseLineBuffer {
    buf: Vec<u8>,
}

impl SseLineBuffer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn extend(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Pop one complete line (without the trailing `\n` / `\r\n`). Returns
    /// `None` if no complete line is buffered yet.
    pub fn next_line(&mut self) -> Option<String> {
        let pos = self.buf.iter().position(|&b| b == b'\n')?;
        let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
        line.pop(); // drop the `\n`
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        // SSE wire is UTF-8; if a server ever lies, lossy is the safest
        // fallback (better one `�` than dropping the whole frame).
        Some(String::from_utf8(line).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()))
    }
}

impl Default for SseLineBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_across_chunks_keeps_utf8_intact() {
        let mut buf = SseLineBuffer::new();
        // "data: 中文\n" — the 中 (E4 B8 AD) gets split mid-char.
        let bytes = b"data: \xe4\xb8\xad\xe6\x96\x87\n";
        buf.extend(&bytes[..8]); // up to E4 B8 (partial)
        assert!(buf.next_line().is_none());
        buf.extend(&bytes[8..]);
        let line = buf.next_line().expect("line ready");
        assert_eq!(line, "data: 中文");
    }

    #[test]
    fn handles_crlf() {
        let mut buf = SseLineBuffer::new();
        buf.extend(b"hello\r\n");
        assert_eq!(buf.next_line().as_deref(), Some("hello"));
    }
}
