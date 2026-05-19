use thiserror::Error;

#[derive(Debug, Error)]
pub enum SkillError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse {file}: {detail}")]
    Parse { file: String, detail: String },
}

pub type Result<T> = std::result::Result<T, SkillError>;
