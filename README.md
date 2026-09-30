# ESP32-S3 Edge AI Starter

The Rust firmware project is in [edge-ai-starter](edge-ai-starter/README.md). It contains a minimal board example and a MicroFlow sine-model inference example.

Start in the nested project directory:

```bash
cd edge-ai-starter
# After preparing the custom Xtensa Rust toolchain and environment:
cargo +esp build --release --bin sine-predict
```

Read the nested README, Cargo manifest, and target configuration before flashing a board. Building the example is separate from demonstrating real sensor input, networking, model quality, or hardware timing. This repository does not have a Cargo manifest at its top level.
