# ============================================================
# Voicebox MCP Server — Rust binary
# 2-stage build: Builder → Runtime
# ============================================================

# === Stage 1: Build the binary ===
FROM rust:1.88-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency builds
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && echo "" > src/lib.rs
RUN cargo build --release 2>/dev/null || true
RUN rm -rf src

# Build the actual binary
COPY src/ src/
RUN cargo build --release


# === Stage 2: Runtime ===
FROM debian:bookworm-slim

# Create non-root user for security
RUN groupadd -r voicebox && \
    useradd -r -g voicebox -m -s /bin/bash voicebox

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/voicebox-mcp /usr/local/bin/voicebox-mcp

USER voicebox

# Default to HTTP transport in container
ENV VOICEBOX_URL=http://voicebox:17493
ENV VOICEBOX_MCP_TRANSPORT=http
ENV VOICEBOX_MCP_HOST=0.0.0.0
ENV VOICEBOX_MCP_PORT=3100

EXPOSE 3100

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:3100/mcp || exit 1

ENTRYPOINT ["voicebox-mcp"]
