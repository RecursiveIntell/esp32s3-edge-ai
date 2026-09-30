# ESP32-S3 Edge AI Starter (Rust)

TinyML inference on the ESP32-S3 using Rust + MicroFlow.

## What's Here

- `src/bin/main.rs` — Minimal hello world (blinks + logs)
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
- `esp` Rust toolchain (Xtensa fork, installed via `espup`)
- `esp-hal` v1.1.0 (no_std HAL)
- `microflow` v0.1.3 (no_std TinyML inference engine)
- `espflash` v4.4.0 (flashing tool)
- `esp-generate` v1.3.0 (project generator)

### Toolchain setup

```bash
# Install ESP Rust toolchain (Xtensa fork)
espup install
source ~/export-esp.sh

# Install flashing tools
cargo install espflash cargo-espflash esp-generate

# Add user to dialout group for USB serial access
sudo usermod -aG dialout $USER

# Configure serial-device access for your operating system and exact board.
```

## Build & Flash

```bash
# Source ESP environment (or rely on .bashrc)
source ~/export-esp.sh

# Build
cargo build --release --bin sine-predict

# Flash (plug in ESP32-S3 via USB first)
espflash flash target/xtensa-esp32s3-none-elf/release/sine-predict --monitor
```

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