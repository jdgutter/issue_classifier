import os
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from src.config import settings

def quantize_ranking_model(
    input_onnx_path: str,
    output_onnx_path: str,
    weight_type: QuantType = QuantType.QInt8
) -> None:
    """
    Applies post-training dynamic quantization to the ONNX ranking model.
    First cleans up initializers from value_info to prevent shape inference errors,
    then compresses 32-bit floating-point weights down to 8-bit integers (INT8).
    """
    if not os.path.exists(input_onnx_path):
        raise FileNotFoundError(f"Source ONNX model not found at {input_onnx_path}")

    # 1. Cleanup value_info in the ONNX model to prevent shape inference mismatches
    print(f"Cleaning up {input_onnx_path} to remove initializers from value_info...")
    model = onnx.load(input_onnx_path)
    initializers = {i.name for i in model.graph.initializer}
    
    # Filter out value_info entries that are also initializers
    filtered_value_info = [v for v in model.graph.value_info if v.name not in initializers]
    del model.graph.value_info[:]
    model.graph.value_info.extend(filtered_value_info)
    
    cleaned_temp_path = input_onnx_path + ".cleaned.tmp"
    onnx.save(model, cleaned_temp_path)

    # 2. Perform Quantization
    print(f"Applying post-training dynamic quantization to: {cleaned_temp_path}")
    print(f"Target quantized model path: {output_onnx_path}")

    start_time = time.time()
    quantize_dynamic(
        model_input=cleaned_temp_path,
        model_output=output_onnx_path,
        weight_type=weight_type
    )
    duration = time.time() - start_time
    print(f"Quantization completed in {duration:.4f} seconds.")

    # Cleanup temp file
    if os.path.exists(cleaned_temp_path):
        os.remove(cleaned_temp_path)

    # 3. Compare file sizes (counting both .onnx and .onnx.data if they exist)
    def get_total_model_size(base_path: str) -> int:
        total_size = os.path.getsize(base_path)
        data_path = base_path + ".data"
        if os.path.exists(data_path):
            total_size += os.path.getsize(data_path)
        return total_size

    original_size = get_total_model_size(input_onnx_path)
    quantized_size = get_total_model_size(output_onnx_path)
    reduction = (original_size - quantized_size) / original_size * 100

    print("\n--- Model Size Comparison ---")
    print(f"Original Model Total Size:  {original_size / 1024:.2f} KB")
    print(f"Quantized Model Total Size: {quantized_size / 1024:.2f} KB")
    print(f"Size Reduction:             {reduction:.2f}%")

    # 4. Numerical Validation Check
    print("\n--- Verifying Quantized Model Correctness ---")
    session_original = ort.InferenceSession(input_onnx_path)
    session_quantized = ort.InferenceSession(output_onnx_path)

    # Dummy inputs matching the model signature: batch_size=1
    dummy_cont = np.random.randn(1, 3).astype(np.float32)
    dummy_cat = np.array([[2, 5]], dtype=np.int64)
    dummy_emb = np.random.randn(1, 384).astype(np.float32)

    ort_inputs = {
        "continuous_features": dummy_cont,
        "categorical_features": dummy_cat,
        "text_embeddings": dummy_emb
    }

    # Run inference
    orig_outs = session_original.run(None, ort_inputs)
    quant_outs = session_quantized.run(None, ort_inputs)

    orig_logits = orig_outs[0][0][0]
    quant_logits = quant_outs[0][0][0]

    # Probabilities
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    orig_prob = sigmoid(orig_logits)
    quant_prob = sigmoid(quant_logits)

    print(f"Original Logits:  {orig_logits:.6f} | Probability: {orig_prob:.6f}")
    print(f"Quantized Logits: {quant_logits:.6f} | Probability: {quant_prob:.6f}")
    print(f"Absolute Difference in Probability: {abs(orig_prob - quant_prob):.6f}")

    # Latency comparison over 500 runs
    print("\n--- Running Latency Benchmarking (500 iterations) ---")
    
    # Warmup
    for _ in range(20):
        _ = session_original.run(None, ort_inputs)
        _ = session_quantized.run(None, ort_inputs)

    # Original runtime
    t0 = time.perf_counter()
    for _ in range(500):
        _ = session_original.run(None, ort_inputs)
    t_orig = (time.perf_counter() - t0) * 1000 / 500.0

    # Quantized runtime
    t0 = time.perf_counter()
    for _ in range(500):
        _ = session_quantized.run(None, ort_inputs)
    t_quant = (time.perf_counter() - t0) * 1000 / 500.0

    print(f"Original Model Mean Latency:  {t_orig:.4f} ms per query")
    print(f"Quantized Model Mean Latency: {t_quant:.4f} ms per query")
    speedup = (t_orig - t_quant) / t_orig * 100
    if speedup > 0:
        print(f"Quantized speedup:            {speedup:.2f}% faster")
    else:
        print("Quantized speedup:            N/A (Overhead of dynamic scale/zero-point compute on small networks may offset weight load savings)")

if __name__ == "__main__":
    input_model = str(settings.MODELS_DIR / "ranking_model.onnx")
    output_model = str(settings.MODELS_DIR / "ranking_model_quantized.onnx")
    
    quantize_ranking_model(input_model, output_model)
