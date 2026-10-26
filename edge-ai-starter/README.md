# ESP32-S3 Edge AI Starter (Rust)

<!-- last-verified: 2026-10-25; local verification date (America/Chicago); README claims checked against toolchain declarations and source; build and hardware path not run -->

TinyML inference on the ESP32-S3 using Rust + MicroFlow.

## What's Here

- `src/bin/main.rs` — Minimal hello world; logs “Hello world!” every 500 ms (no GPIO LED)
- `src/bin/sine_predict.rs` — Neural network sine predictor using MicroFlow + TFLite
- `models/sine.tflite` — Quantized TFLite sine model (2.6KB)

## Hardware

ESP32-S3 (Xtensa LX7 dual-core, 240MHz):
- 512KB SRAM (384KB usable after WiFi/BLE reserve)
- 128KB ROM
- PSRAM up to 8MB (on supported boards)
- 2MB+ external flash (typical dev boards have 8-16MB)
- USB JTAG/Serial (no external programmer needed)
- Vector instructions (SIMD) for AI acceleration (not yet exposed in Rust)

## Toolchain

This project uses:
- Rust `1.88` or newer as the package's declared minimum; `rust-toolchain.toml` selects the `esp` toolchain for the Xtensa target
- `esp-hal` with the `~1.1.0` requirement and `microflow` with the `0.1` requirement, as declared in `Cargo.toml`
- The `xtensa-esp32s3-none-elf` target and an `espflash flash --monitor --chip esp32s3 --log-format defmt` Cargo runner, as configured in `.cargo/config.toml`

There is no checked-in `Cargo.lock`, so Cargo may resolve newer compatible dependency releases. The `espflash` executable version is not pinned by this repository.

### Toolchain setup

```bash
# Install ESP Rust toolchain (Xtensa fork)
espup install
source ~/export-esp.sh

# Install the flashing/monitor tool configured as the Cargo runner
cargo install espflash

# Configure serial-device access for your operating system and exact board.
```

## Build & Flash

From `edge-ai-starter/`, build the firmware with the configured Xtensa target:

```bash
cargo build --release --bin sine-predict
```

Configured flash-and-monitor path (source-derived; not run with the Xtensa toolchain or hardware):

```bash
cargo run --release --bin sine-predict
```

The Cargo runner in `.cargo/config.toml` invokes `espflash` with the ESP32-S3 chip and defmt monitor. On startup, the firmware reports predicted and `libm` reference values, error, and elapsed microseconds for five inputs, prints a completion message, then logs an alive message every two seconds. This is the source-defined output; this README update did not run the firmware on hardware.

## MicroFlow — How It Works

MicroFlow uses a compile-time procedural macro to bake the TFLite model directly
into the binary. No runtime file system or model loading needed.

```rust
use microflow::model;
use nalgebra::matrix;

#[model("models/sine.tflite")]
struct Sine;

fn main() {
    let prediction = Sine::predict(matrix![0.5]);  // Returns matrix![y]
}
```

Supported operators (all with int8 quantization):
- FullyConnected, Conv2D, DepthwiseConv2D, AveragePool2D, Reshape
- ReLU, ReLU6, Softmax

This checkout includes the sine model. Upstream MicroFlow model/operator support does not establish that other models have been built or run with this firmware.

## Model Pipeline

To deploy your own model:

1. Train in PyTorch or TensorFlow
2. Convert to TFLite with int8 quantization:
   ```python
   import tensorflow as tf
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.representative_dataset = representative_data
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   tflite_model = converter.convert()
   ```
3. Place `.tflite` file in `models/`
4. Use `#[model("models/your_model.tflite")]` in Rust

## ESP32-S3 Constraints

- Model capacity depends on firmware layout, allocation strategy, and the actual board memory; measure the linked image and runtime use.
- int8 quantization is required (float models waste 4x memory)
- Measure latency on the selected firmware/model and actual hardware. No timing guarantee is made here.
- Scalar floating-point support and SIMD kernel use are separate concerns; inspect the selected implementation rather than assuming all Rust inference paths use vector acceleration.

## Next Steps

- [ ] Add person detection model (MobileNet v1) when board arrives
- [ ] Benchmark inference times on actual hardware
- [ ] Explore ESP32-S3 vector instructions (SIMD) for matrix multiply acceleration
- [ ] WiFi telemetry: send inference results to local server
- [ ] Camera integration (OV2640) for real-time person detection
- [ ] Connect to turbo-quant/fib-quant for custom quantization schemes