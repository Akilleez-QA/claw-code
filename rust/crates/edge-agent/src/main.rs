/// ThermoLogic edge agent — minimal ARM binary for facility hardware
///
/// Designed to run on Raspberry Pi 4 (aarch64) without cloud dependency
/// for core features. Connects to Anthropic API when ANTHROPIC_API_KEY is
/// set; otherwise falls back to offline-only PT calculator and RAG.
///
/// Features:
///   pt-calculator  Pressure-Temperature calculations for common refrigerants
///   rag-sqlite     Local SQLite document store for offline knowledge lookup
///
/// Build for Pi 4:
///   cargo build --profile edge --target aarch64-unknown-linux-gnu -p edge-agent
///
/// Build static x86 (for containers):
///   cargo build --profile edge --target x86_64-unknown-linux-musl -p edge-agent

mod pt_calculator;
#[cfg(feature = "rag-sqlite")]
mod rag;

use std::io::{BufRead, Write};

use api::{
    AnthropicClient, ContentBlockDelta, InputContentBlock, InputMessage, MessageRequest,
    OutputContentBlock, StreamEvent as ApiStreamEvent, ToolChoice, ToolDefinition,
    ToolResultContentBlock,
};
use runtime::{
    ApiClient, ApiRequest, AssistantEvent, ContentBlock, ConversationMessage, ConversationRuntime,
    MessageRole, PermissionMode, PermissionPolicy, RuntimeError, Session, StaticToolExecutor,
    ToolError, ToolExecutor, TokenUsage,
};
use serde_json::Value as JsonValue;

// ── model constant ────────────────────────────────────────────────────────────

const DEFAULT_MODEL: &str = "claude-haiku-4-5-20251001";
const MAX_TOKENS: u32 = 4096;
const EDGE_SYSTEM_PROMPT: &str = "\
You are ThermoLogic Edge — a compact AI assistant deployed on facility hardware \
for HVAC/refrigeration technicians. You have access to offline tools:\n\
- pt_calculator: pressure-temperature lookups for refrigerants (R-22, R-410A, R-404A, etc.)\n\
- rag_search: search the local knowledge base for service manuals and technical notes\n\
\n\
Keep responses concise. Prioritize accuracy for safety-critical refrigerant data.\
";

// ── edge tool registration ────────────────────────────────────────────────────

#[derive(Clone)]
struct EdgeToolSpec {
    name: &'static str,
    description: &'static str,
    input_schema: JsonValue,
}

fn edge_tool_specs() -> Vec<EdgeToolSpec> {
    let mut specs = Vec::new();

    #[cfg(feature = "pt-calculator")]
    specs.push(EdgeToolSpec {
        name: "pt_calculator",
        description: "Pressure-Temperature lookup for refrigerants. \
                      Converts temperature↔pressure or generates PT charts.",
        input_schema: pt_calculator::tool_schema(),
    });

    #[cfg(feature = "rag-sqlite")]
    specs.push(EdgeToolSpec {
        name: "rag_search",
        description: "Search local knowledge base (service manuals, technical notes). \
                      Works fully offline.",
        input_schema: rag::tool_schema(),
    });

    specs
}

// ── API client bridge ─────────────────────────────────────────────────────────

struct EdgeApiClient {
    /// Tokio runtime for blocking async calls from sync ConversationRuntime
    rt: tokio::runtime::Runtime,
    client: AnthropicClient,
    model: String,
    tool_defs: Vec<ToolDefinition>,
}

impl EdgeApiClient {
    fn new(client: AnthropicClient, model: String) -> Self {
        let tool_defs = edge_tool_specs()
            .into_iter()
            .map(|spec| ToolDefinition {
                name: spec.name.to_string(),
                description: Some(spec.description.to_string()),
                input_schema: spec.input_schema,
            })
            .collect();
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        Self { rt, client, model, tool_defs }
    }
}

impl ApiClient for EdgeApiClient {
    fn stream(&mut self, request: ApiRequest) -> Result<Vec<AssistantEvent>, RuntimeError> {
        let messages = convert_messages(&request.messages);
        let system = (!request.system_prompt.is_empty())
            .then(|| request.system_prompt.join("\n\n"));
        let has_tools = !self.tool_defs.is_empty();
        let msg_req = MessageRequest {
            model: self.model.clone(),
            max_tokens: MAX_TOKENS,
            messages,
            system,
            tools: has_tools.then(|| self.tool_defs.clone()),
            tool_choice: has_tools.then_some(ToolChoice::Auto),
            stream: true,
            ..Default::default()
        };

        self.rt.block_on(async {
            let mut stream = self
                .client
                .stream_message(&msg_req)
                .await
                .map_err(|e| RuntimeError::new(e.to_string()))?;

            let mut events: Vec<AssistantEvent> = Vec::new();
            // (tool_id, tool_name, accumulated_input_json)
            let mut pending_tool: Option<(String, String, String)> = None;

            while let Some(event) = stream
                .next_event()
                .await
                .map_err(|e| RuntimeError::new(e.to_string()))?
            {
                match event {
                    ApiStreamEvent::ContentBlockStart(start) => match start.content_block {
                        OutputContentBlock::ToolUse { id, name, .. } => {
                            pending_tool = Some((id, name, String::new()));
                        }
                        OutputContentBlock::Text { text } => {
                            if !text.is_empty() {
                                events.push(AssistantEvent::TextDelta(text));
                            }
                        }
                        _ => {}
                    },
                    ApiStreamEvent::ContentBlockDelta(delta) => match delta.delta {
                        ContentBlockDelta::TextDelta { text } => {
                            if !text.is_empty() {
                                events.push(AssistantEvent::TextDelta(text));
                            }
                        }
                        ContentBlockDelta::InputJsonDelta { partial_json } => {
                            if let Some((_, _, input)) = &mut pending_tool {
                                input.push_str(&partial_json);
                            }
                        }
                    },
                    ApiStreamEvent::ContentBlockStop(_) => {
                        if let Some((id, name, input)) = pending_tool.take() {
                            events.push(AssistantEvent::ToolUse { id, name, input });
                        }
                    }
                    ApiStreamEvent::MessageStart(start) => {
                        let usage = &start.message.usage;
                        events.push(AssistantEvent::Usage(TokenUsage {
                            input_tokens: usage.input_tokens,
                            output_tokens: usage.output_tokens,
                            cache_creation_input_tokens: usage.cache_creation_input_tokens,
                            cache_read_input_tokens: usage.cache_read_input_tokens,
                        }));
                    }
                    ApiStreamEvent::MessageDelta(delta) => {
                        let usage = &delta.usage;
                        events.push(AssistantEvent::Usage(TokenUsage {
                            input_tokens: usage.input_tokens,
                            output_tokens: usage.output_tokens,
                            cache_creation_input_tokens: usage.cache_creation_input_tokens,
                            cache_read_input_tokens: usage.cache_read_input_tokens,
                        }));
                    }
                    ApiStreamEvent::MessageStop(_) => {
                        events.push(AssistantEvent::MessageStop);
                    }
                    _ => {}
                }
            }
            Ok(events)
        })
    }
}

/// Convert runtime session messages to API wire format.
fn convert_messages(messages: &[ConversationMessage]) -> Vec<InputMessage> {
    let mut api_messages: Vec<InputMessage> = Vec::new();
    // Group consecutive tool results into a single user message.
    let mut pending_results: Vec<InputContentBlock> = Vec::new();

    let flush_results =
        |pending: &mut Vec<InputContentBlock>, out: &mut Vec<InputMessage>| {
            if !pending.is_empty() {
                out.push(InputMessage {
                    role: "user".to_string(),
                    content: std::mem::take(pending),
                });
            }
        };

    for msg in messages {
        match msg.role {
            MessageRole::User => {
                flush_results(&mut pending_results, &mut api_messages);
                let content = msg
                    .blocks
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => {
                            Some(InputContentBlock::Text { text: text.clone() })
                        }
                        ContentBlock::ToolResult {
                            tool_use_id,
                            output,
                            is_error,
                            ..
                        } => Some(InputContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: vec![ToolResultContentBlock::Text {
                                text: output.clone(),
                            }],
                            is_error: *is_error,
                        }),
                        _ => None,
                    })
                    .collect();
                api_messages.push(InputMessage { role: "user".to_string(), content });
            }
            MessageRole::Assistant => {
                flush_results(&mut pending_results, &mut api_messages);
                let content = msg
                    .blocks
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => {
                            Some(InputContentBlock::Text { text: text.clone() })
                        }
                        ContentBlock::ToolUse { id, name, input } => {
                            let parsed: JsonValue =
                                serde_json::from_str(input).unwrap_or(JsonValue::Null);
                            Some(InputContentBlock::ToolUse {
                                id: id.clone(),
                                name: name.clone(),
                                input: parsed,
                            })
                        }
                        _ => None,
                    })
                    .collect();
                api_messages.push(InputMessage { role: "assistant".to_string(), content });
            }
            MessageRole::Tool => {
                for block in &msg.blocks {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        output,
                        is_error,
                        ..
                    } = block
                    {
                        pending_results.push(InputContentBlock::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: vec![ToolResultContentBlock::Text {
                                text: output.clone(),
                            }],
                            is_error: *is_error,
                        });
                    }
                }
            }
            MessageRole::System => {}
        }
    }
    flush_results(&mut pending_results, &mut api_messages);
    api_messages
}

// ── offline tool executor ─────────────────────────────────────────────────────

fn build_tool_executor(_rag_db_path: Option<String>) -> StaticToolExecutor {
    let mut executor = StaticToolExecutor::new();

    #[cfg(feature = "pt-calculator")]
    {
        executor = executor.register("pt_calculator", |input_str| {
            let input: JsonValue = serde_json::from_str(input_str).map_err(|e| {
                ToolError::new(format!("invalid JSON input for pt_calculator: {e}"))
            })?;
            let result = pt_calculator::execute_pt_tool(&input);
            Ok(result.to_string())
        });
    }

    #[cfg(feature = "rag-sqlite")]
    {
        use std::sync::{Arc, Mutex};
        let db_path = _rag_db_path.unwrap_or_else(|| "edge-rag.db".to_string());
        match rag::RagStore::open(&db_path) {
            Ok(store) => {
                let store = Arc::new(Mutex::new(store));
                executor = executor.register("rag_search", move |input_str| {
                    let input: JsonValue = serde_json::from_str(input_str).map_err(|e| {
                        ToolError::new(format!("invalid JSON input for rag_search: {e}"))
                    })?;
                    let store = store.lock().map_err(|_| ToolError::new("rag store lock poisoned"))?;
                    let result = rag::execute_rag_tool(&store, &input);
                    Ok(result.to_string())
                });
            }
            Err(e) => {
                eprintln!("[edge-agent] WARNING: could not open RAG store at {db_path}: {e}");
            }
        }
    }

    executor
}

// ── permission policy ─────────────────────────────────────────────────────────

fn edge_permission_policy() -> PermissionPolicy {
    // Edge tools are all read-only calculations — no file mutation, no bash.
    PermissionPolicy::new(PermissionMode::ReadOnly)
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let model =
        std::env::var("EDGE_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    let rag_db = std::env::var("EDGE_RAG_DB").ok();

    // Try to build the API client; fall back to offline-only mode if no key.
    let api_client_result = AnthropicClient::from_env()
        .map(|c| EdgeApiClient::new(c, model.clone()));

    let use_api = api_client_result.is_ok();
    if !use_api {
        eprintln!(
            "[edge-agent] No ANTHROPIC_API_KEY — running in offline calculator mode only."
        );
    }

    let tool_executor = build_tool_executor(rag_db);

    let session = Session::new();

    if use_api {
        let api_client = api_client_result.unwrap();
        let mut runtime = ConversationRuntime::new(
            session,
            api_client,
            tool_executor,
            edge_permission_policy(),
            vec![EDGE_SYSTEM_PROMPT.to_string()],
        );
        run_repl(&mut runtime);
    } else {
        run_offline_repl();
    }
}

/// Interactive REPL using the full Anthropic API + edge tools.
fn run_repl<C, T>(runtime: &mut ConversationRuntime<C, T>)
where
    C: ApiClient,
    T: ToolExecutor,
{
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    println!("ThermoLogic Edge Agent — type your question, Ctrl-D to exit.");

    for line in stdin.lock().lines() {
        let input = match line {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => l,
            Err(_) => break,
        };

        match runtime.run_turn(input, None) {
            Ok(summary) => {
                for msg in &summary.assistant_messages {
                    for block in &msg.blocks {
                        if let ContentBlock::Text { text } = block {
                            println!("\n{text}");
                        }
                    }
                }
                let _ = stdout.flush();
            }
            Err(e) => eprintln!("[error] {e}"),
        }
    }
}

/// Offline REPL — only PT calculator, no LLM calls.
fn run_offline_repl() {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    println!(
        "ThermoLogic Edge (offline mode) — PT Calculator\n\
         Usage: <refrigerant> <temp_f>   e.g.  R-410A 45\n\
                <refrigerant> p <psig>   e.g.  R-22 p 68.5\n\
         Ctrl-D to exit."
    );

    for line in stdin.lock().lines() {
        let input = match line {
            Ok(l) if l.trim().is_empty() => continue,
            Ok(l) => l.trim().to_string(),
            Err(_) => break,
        };

        #[cfg(feature = "pt-calculator")]
        {
            let parts: Vec<&str> = input.splitn(3, ' ').collect();
            match parts.as_slice() {
                [ref_, temp] => match temp.parse::<f64>() {
                    Ok(t) => match pt_calculator::temp_to_pressure(ref_, t) {
                        Ok(r) => println!("{}", pt_calculator::format_pt_result(&r)),
                        Err(e) => eprintln!("error: {e}"),
                    },
                    Err(_) => eprintln!("expected a number for temperature, got: {temp}"),
                },
                [ref_, "p", psig] | [ref_, "P", psig] => match psig.parse::<f64>() {
                    Ok(p) => match pt_calculator::pressure_to_temp(ref_, p) {
                        Ok(r) => println!("{}", pt_calculator::format_pt_result(&r)),
                        Err(e) => eprintln!("error: {e}"),
                    },
                    Err(_) => eprintln!("expected a number for pressure, got: {psig}"),
                },
                _ => eprintln!("usage: <refrigerant> <temp_f>  OR  <refrigerant> p <psig>"),
            }
        }
        #[cfg(not(feature = "pt-calculator"))]
        {
            eprintln!("offline mode requires pt-calculator feature");
        }

        let _ = stdout.flush();
    }
}
