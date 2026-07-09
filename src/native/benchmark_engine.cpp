#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model> [num_iterations]\n";
        return 1;
    }

    std::string model_path = argv[1];
    int num_iterations = 10000;
    if (argc >= 3) {
        num_iterations = std::stoi(argv[2]);
    }
    const int warmup_iterations = 100;

    std::cout << "Loading model from: " << model_path << "\n";
    std::cout << "Warmup iterations: " << warmup_iterations << "\n";
    std::cout << "Benchmark iterations: " << num_iterations << "\n";

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "High_Performance_Benchmark");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1); // Single-thread CPU determinism
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        Ort::Session session(env, model_path.c_str(), session_options);

        // Define shapes and inputs
        const int64_t batch_size = 1;
        
        std::vector<float> continuous_input = { 1.5f, 0.8f, -0.4f };
        std::vector<int64_t> continuous_shape = { batch_size, 3 };

        std::vector<int64_t> categorical_input = { 2, 5 };
        std::vector<int64_t> categorical_shape = { batch_size, 2 };

        std::vector<float> embedding_input(384, 0.0f);
        for (int i = 0; i < 384; ++i) {
            embedding_input[i] = static_cast<float>(i % 10) / 10.0f - 0.5f;
        }
        std::vector<int64_t> embedding_shape = { batch_size, 384 };

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        const char* input_names[] = { "continuous_features", "categorical_features", "text_embeddings" };
        const char* output_names[] = { "logits" };

        // 1. Warmup loop
        for (int i = 0; i < warmup_iterations; ++i) {
            Ort::Value continuous_tensor = Ort::Value::CreateTensor<float>(
                memory_info, continuous_input.data(), continuous_input.size(), continuous_shape.data(), continuous_shape.size());
            Ort::Value categorical_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, categorical_input.data(), categorical_input.size(), categorical_shape.data(), categorical_shape.size());
            Ort::Value embedding_tensor = Ort::Value::CreateTensor<float>(
                memory_info, embedding_input.data(), embedding_input.size(), embedding_shape.data(), embedding_shape.size());

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(continuous_tensor));
            input_tensors.push_back(std::move(categorical_tensor));
            input_tensors.push_back(std::move(embedding_tensor));

            auto output_tensors = session.Run(
                Ort::RunOptions{ nullptr },
                input_names,
                input_tensors.data(),
                input_tensors.size(),
                output_names,
                1
            );
        }

        // 2. Benchmark loop
        std::vector<double> latencies;
        latencies.reserve(num_iterations);

        for (int i = 0; i < num_iterations; ++i) {
            // Re-wrap input tensors to simulate realistic workflow
            Ort::Value continuous_tensor = Ort::Value::CreateTensor<float>(
                memory_info, continuous_input.data(), continuous_input.size(), continuous_shape.data(), continuous_shape.size());
            Ort::Value categorical_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, categorical_input.data(), categorical_input.size(), categorical_shape.data(), categorical_shape.size());
            Ort::Value embedding_tensor = Ort::Value::CreateTensor<float>(
                memory_info, embedding_input.data(), embedding_input.size(), embedding_shape.data(), embedding_shape.size());

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(std::move(continuous_tensor));
            input_tensors.push_back(std::move(categorical_tensor));
            input_tensors.push_back(std::move(embedding_tensor));

            auto start = std::chrono::high_resolution_clock::now();
            
            auto output_tensors = session.Run(
                Ort::RunOptions{ nullptr },
                input_names,
                input_tensors.data(),
                input_tensors.size(),
                output_names,
                1
            );

            auto end = std::chrono::high_resolution_clock::now();
            
            double duration_us = std::chrono::duration<double, std::micro>(end - start).count();
            latencies.push_back(duration_us);
        }

        // 3. Compute stats
        double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double mean = sum / latencies.size();

        std::sort(latencies.begin(), latencies.end());

        double min_val = latencies.front();
        double max_val = latencies.back();
        double p50 = latencies[static_cast<size_t>(num_iterations * 0.50)];
        double p95 = latencies[static_cast<size_t>(num_iterations * 0.95)];
        double p99 = latencies[static_cast<size_t>(num_iterations * 0.99)];

        std::cout << "\n================ Benchmark Results ================\n";
        std::cout << "Mean Latency: " << mean << " us (" << mean / 1000.0 << " ms)\n";
        std::cout << "Min Latency:  " << min_val << " us (" << min_val / 1000.0 << " ms)\n";
        std::cout << "Max Latency:  " << max_val << " us (" << max_val / 1000.0 << " ms)\n";
        std::cout << "p50 Median:   " << p50 << " us (" << p50 / 1000.0 << " ms)\n";
        std::cout << "p95 Latency:  " << p95 << " us (" << p95 / 1000.0 << " ms)\n";
        std::cout << "p99 Latency:  " << p99 << " us (" << p99 / 1000.0 << " ms)\n";
        std::cout << "===================================================\n";

    } catch (const Ort::Exception& oe) {
        std::cerr << "ONNX Runtime C++ Error: " << oe.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << "\n";
        return 3;
    }

    return 0;
}
