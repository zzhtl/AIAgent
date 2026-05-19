//! Tiny YAML-front-matter splitter.
//!
//! Recognises the markdown convention: an opening line `---`, YAML key/values
//! (strings or `- item` lists), a closing line `---`, then the body. Returns
//! the parsed fields plus the body.
//!
//! Deliberately not a real YAML parser — only the field shapes Skill / Rule
//! files actually use. If a file needs anything more complex it can omit the
//! frontmatter and rely on the body alone.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum FmValue {
    String(String),
    List(Vec<String>),
}

impl FmValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            FmValue::String(s) => Some(s.as_str()),
            FmValue::List(_) => None,
        }
    }

    pub fn as_list(&self) -> Option<&[String]> {
        match self {
            FmValue::List(v) => Some(v.as_slice()),
            FmValue::String(_) => None,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Frontmatter {
    pub fields: HashMap<String, FmValue>,
}

impl Frontmatter {
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(FmValue::as_str)
    }

    pub fn get_list(&self, key: &str) -> Vec<String> {
        match self.fields.get(key) {
            Some(FmValue::List(v)) => v.clone(),
            Some(FmValue::String(s)) => vec![s.clone()],
            None => Vec::new(),
        }
    }
}

/// Split a markdown document into frontmatter + body. If no frontmatter is
/// present, returns `(None, body)` where body is the whole input.
pub fn split(content: &str) -> (Option<Frontmatter>, String) {
    let mut lines = content.lines();
    let first = match lines.next() {
        Some(l) => l,
        None => return (None, String::new()),
    };
    if first.trim() != "---" {
        return (None, content.to_string());
    }

    let mut yaml_buf = String::new();
    let mut body_buf = String::new();
    let mut in_yaml = true;
    for line in lines {
        if in_yaml {
            if line.trim() == "---" {
                in_yaml = false;
                continue;
            }
            yaml_buf.push_str(line);
            yaml_buf.push('\n');
        } else {
            body_buf.push_str(line);
            body_buf.push('\n');
        }
    }
    if in_yaml {
        // No closing `---`; treat the whole input as body.
        return (None, content.to_string());
    }
    (Some(parse_yaml_lite(&yaml_buf)), body_buf.trim_start_matches('\n').to_string())
}

fn parse_yaml_lite(yaml: &str) -> Frontmatter {
    let mut fields: HashMap<String, FmValue> = HashMap::new();
    let mut current_list_key: Option<String> = None;
    let mut current_list: Vec<String> = Vec::new();

    let flush = |fields: &mut HashMap<String, FmValue>,
                 key: &mut Option<String>,
                 list: &mut Vec<String>| {
        if let Some(k) = key.take() {
            fields.insert(k, FmValue::List(std::mem::take(list)));
        }
    };

    for raw in yaml.lines() {
        let line = raw.trim_end();
        if line.is_empty() || line.trim_start().starts_with('#') {
            continue;
        }

        // List item under the current key?
        if current_list_key.is_some() {
            if let Some(item) = line.trim_start().strip_prefix("- ") {
                current_list.push(strip_quotes(item.trim()).to_string());
                continue;
            }
        }

        // Otherwise we're starting a new key — flush any pending list.
        flush(&mut fields, &mut current_list_key, &mut current_list);

        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim().to_string();
            let val = v.trim();
            if val.is_empty() {
                current_list_key = Some(key);
            } else {
                fields.insert(key, FmValue::String(strip_quotes(val).to_string()));
            }
        }
    }

    flush(&mut fields, &mut current_list_key, &mut current_list);
    Frontmatter { fields }
}

fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_frontmatter() {
        let (fm, body) = split("plain markdown\n");
        assert!(fm.is_none());
        assert_eq!(body, "plain markdown\n");
    }

    #[test]
    fn with_frontmatter() {
        let input = "---\nname: foo\ndescription: \"bar\"\ntriggers:\n  - a\n  - b\n---\nbody here\n";
        let (fm, body) = split(input);
        let fm = fm.expect("frontmatter present");
        assert_eq!(fm.get_string("name"), Some("foo"));
        assert_eq!(fm.get_string("description"), Some("bar"));
        assert_eq!(fm.get_list("triggers"), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(body, "body here\n");
    }

    #[test]
    fn missing_close_falls_back_to_body() {
        let input = "---\nname: foo\nno closing\n";
        let (fm, body) = split(input);
        assert!(fm.is_none());
        assert_eq!(body, input);
    }
}
