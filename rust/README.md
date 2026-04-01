# Claw Code for Rust

Claw Code is a local coding-agent CLI implemented in safe Rust. The `claw` binary supports interactive sessions, one-shot prompts, workspace-aware tools, and local agent workflows from a single workspace.

## Current status

- **Version:** `0.1.0`
- **Release stage:** initial public release, source-build distribution
- **Platform focus:** macOS and Linux developer workstations
- **Rust port status:** core CLI, runtime, tools, plugins, LSP, and support services are all in the Rust workspace

## Install, build, and run

### Prerequisites

- Rust stable toolchain
- Cargo
- Provider credentials for the model you want to use

### Authentication

Anthropic-compatible models:

```bash
export ANTHROPIC_API_KEY="..."
# Optional when using a compatible endpoint
export ANTHROPIC_BASE_URL="https://api.anthropic.com"
```

Grok models:

```bash
export XAI_API_KEY="..."
# Optional when using a compatible endpoint
export XAI_BASE_URL="https://api.x.ai"
```

OAuth login is also available:

```bash
cargo run --bin claw -- login
```

### Install locally

```bash
cargo install --path crates/claw-cli --locked
```

### Build from source

```bash
cargo build --release -p claw-cli
```

### Run

From the workspace:

```bash
cargo run --bin claw -- --help
cargo run --bin claw --
cargo run --bin claw -- prompt "summarize this workspace"
cargo run --bin claw -- --model sonnet "review the latest changes"
```

From the release build:

```bash
./target/release/claw
./target/release/claw prompt "explain crates/runtime"
```

## Supported capabilities

- Interactive REPL and one-shot prompt execution
- Saved-session inspection and resume flows
- Built-in workspace tools for shell, file read/write/edit, search, web fetch/search, todos, and notebook updates
- Slash commands for status, compaction, config inspection, diff, export, session management, and version reporting
- Local agent and skill discovery with `claw agents` and `claw skills`
- Plugin discovery and management through the CLI and slash-command surfaces
- OAuth login/logout plus model/provider selection from the command line
- Workspace-aware instruction/config loading (`CLAW.md`, config files, permissions, plugin settings)

## Current limitations

- Public distribution is **source-build only** today; this workspace is not set up for crates.io publishing
- GitHub CI verifies `cargo check`, `cargo test`, and release builds, but automated release packaging is not yet present
- Current CI targets Ubuntu and macOS; Windows release readiness is still to be established
- Some live-provider integration coverage is opt-in because it requires external credentials and network access
- The command surface may continue to evolve during the `0.x` series

## Rust port status

The Rust workspace is already the primary implementation surface for this CLI. It currently includes these workspace crates:

- `claw-cli` — user-facing binary
- `api` — provider clients and streaming
- `runtime` — sessions, config, permissions, prompts, and runtime loop
- `tools` — built-in tool implementations
- `commands` — slash-command registry and handlers
- `plugins` — plugin discovery, registry, and lifecycle support
- `lsp` — language-server protocol support types and process helpers
- `server` and `compat-harness` — supporting services and compatibility tooling

## Roadmap

- Publish packaged release artifacts for public installs
- Add a repeatable release workflow and longer-lived changelog discipline
- Expand platform verification beyond the current CI matrix
- Add more task-focused examples and operator documentation
- Continue tightening feature coverage and UX polish across the Rust port

## Release notes

- Draft 0.1.0 release notes: [`docs/releases/0.1.0.md`](docs/releases/0.1.0.md)

## License

See the repository root for licensing details.
