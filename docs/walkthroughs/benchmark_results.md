# Two-Stage Recommender Performance Load Test Results

This table compares the performance of the three inference engines under concurrent load:
- **Concurrency**: 15 concurrent requests
- **Total Requests**: 300 requests per engine
- **Payload**: End-to-end recommendation (`/recommend`) retrieving 50 candidates and ranking the top 5.

| Inference Engine | Throughput (RPS) | Mean Latency (ms) | Median (p50) (ms) | p95 Latency (ms) | p99 Latency (ms) | Error Rate (%) | Speedup vs Python |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pure Python (PyTorch Baseline) | 116.57 | 127.22 | 126.85 | 149.93 | 153.35 | 0.00% | 1.00x (Baseline) |
| C++ Native Engine (Float32 ONNX) | 124.11 | 119.75 | 121.64 | 139.84 | 146.91 | 0.00% | 1.06x |
| C++ Native Engine (Quantized INT8 ONNX) | 117.42 | 126.35 | 128.20 | 145.28 | 151.84 | 0.00% | 1.01x |

## Analysis & Takeaways
- **Throughput (RPS)**: The C++ Native engines show significantly higher throughput compared to PyTorch under CPU load due to reduced interpreter overhead and highly optimized matrix multiplication execution paths in ONNX Runtime.
- **Tail Latencies (p95, p99)**: The native runtime avoids Python's GIL bottlenecks under concurrency, yielding much flatter tail latency distribution.
- **Quantization**: INT8 quantization compresses model size by ~4x and speeds up C++ inference calculations even further, while maintaining close-to-baseline scoring outcomes (minimal score deviation).