use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "agent", version, about = "AI Agent CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start an interactive REPL chat.
    Chat,
    /// Run a single prompt and exit.
    Run {
        /// The user prompt.
        prompt: String,
    },
    /// List stored sessions.
    Sessions,
    /// Resume a previous session by id.
    Resume {
        session_id: String,
    },
    /// Skills management.
    Skills,
    /// Print the resolved configuration.
    Config,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Command::Chat) => println!("[chat] not implemented yet"),
        Some(Command::Run { prompt }) => println!("[run] not implemented yet — prompt: {prompt}"),
        Some(Command::Sessions) => println!("[sessions] not implemented yet"),
        Some(Command::Resume { session_id }) => println!("[resume] not implemented yet — id: {session_id}"),
        Some(Command::Skills) => println!("[skills] not implemented yet"),
        Some(Command::Config) => println!("[config] not implemented yet"),
        None => {
            println!("agent — AI Agent runtime");
            println!("run `agent --help` to see available commands");
        }
    }

    Ok(())
}
