#![no_std]
#![no_main]
//! Sine prediction demo — runs a quantized TFLite neural network on the ESP32-S3.
//! Uses MicroFlow (no_std TinyML inference engine) to predict sin(x) and compares
//! against the libm ground truth, measuring inference time.

use defmt::info;
use esp_backtrace as _;
use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use esp_println as _;
use libm::sinf;
use microflow::model;
use nalgebra::matrix;

// This creates a default app-descriptor required by the esp-idf bootloader.
esp_bootloader_esp_idf::esp_app_desc!();

// Compile the TFLite model into the binary at build time.
// MicroFlow's procedural macro parses the .tflite file and generates
// a predict() method with the network's weights baked in.
#[model("models/sine.tflite")]
struct Sine;

#[allow(
    clippy::large_stack_frames,
    reason = "it's not unusual to allocate larger buffers etc. in main"
)]
#[main]
fn main() -> ! {
    // Max clock for ESP32-S3: 240 MHz dual-core Xtensa LX7
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);

    // Allocate heap for potential dynamic ops (MicroFlow is mostly static,
    // but some tensor ops may need heap allocation)
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 73744);

    // Test multiple input values
    let test_values: [f32; 5] = [0.0, 0.5, 1.0, 1.5, 3.14159];

    for &x in &test_values {
        let start = Instant::now();
        let y_predicted = Sine::predict(matrix![x])[0];
        let elapsed = start.elapsed();

        let y_exact = sinf(x);

        // Log via defmt (visible in espflash monitor)
        info!(
            "sin({}) = predicted: {}, exact: {}, error: {}, time: {} us",
            x, y_predicted, y_exact, y_exact - y_predicted, elapsed.as_micros()
        );
    }

    info!("=== Sine model inference complete ===");

    // Keep the loop alive — blink the log
    loop {
        info!("ESP32-S3 edge-ai-starter alive");
        let delay_start = Instant::now();
        while delay_start.elapsed() < Duration::from_millis(2000) {}
    }
}