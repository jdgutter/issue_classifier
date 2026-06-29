#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <string>

// Helper to print shapes/vectors
void print_vector(const std::string& name, const std::vector<float>& vec) {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << "Initializing ONNX Runtime C++ Environment...\n";

    try {
        // 1. Initialize environment and session options
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "High_Performance_Inference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1); // Set single-thread for deterministic CPU execution
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // 2. Instantiate session
        std::cout << "Loading ONNX model from: " << model_path << "...\n";
        Ort::Session session(env, model_path.c_str(), session_options);

        // 3. Define batch size and features dimensions
        const int64_t batch_size = 1;
        
        // Mock Input 1: Continuous Features (clicks, popularity, age) -> Shape (batch_size, 3)
        std::vector<float> continuous_input = { 1.5f, 0.8f, -0.4f }; // scaled values
        std::vector<int64_t> continuous_shape = { batch_size, 3 };

        // Mock Input 2: Categorical Features (2 tags) -> Shape (batch_size, 2)
        std::vector<int64_t> categorical_input = { 2, 5 }; // tag indices
        std::vector<int64_t> categorical_shape = { batch_size, 2 };

        // Mock Input 3: Text Embeddings -> Shape (batch_size, 384)
        std::vector<float> embedding_input(384, 0.0f);
        // Fill with some mock values
        for (int i = 0; i < 384; ++i) {
            embedding_input[i] = static_cast<float>(i % 10) / 10.0f - 0.5f;
        }
        std::vector<int64_t> embedding_shape = { batch_size, 384 };

        // 4. Set memory allocation info (CPU allocator)
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // 5. Wrap raw continuous and categorical C++ buffers into Ort::Value tensors (zero-copy)
        Ort::Value continuous_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            continuous_input.data(),
            continuous_input.size(),
            continuous_shape.data(),
            continuous_shape.size()
        );

        Ort::Value categorical_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            categorical_input.data(),
            categorical_input.size(),
            categorical_shape.data(),
            categorical_shape.size()
        );

        Ort::Value embedding_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            embedding_input.data(),
            embedding_input.size(),
            embedding_shape.data(),
            embedding_shape.size()
        );

        // 6. Assemble inputs
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(continuous_tensor));
        input_tensors.push_back(std::move(categorical_tensor));
        input_tensors.push_back(std::move(embedding_tensor));

        // Define exact ONNX model input/output names
        const char* input_names[] = { "continuous_features", "categorical_features", "text_embeddings" };
        const char* output_names[] = { "logits" };

        std::cout << "Executing forward native tensor pass (Inference)...\n";
        
        // 7. Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_names,
            input_tensors.data(),
            input_tensors.size(),
            output_names,
            1
        );

        // 8. Extract logits output
        float* raw_logits = output_tensors[0].GetTensorMutableData<float>();
        size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::vector<float> logits(raw_logits, raw_logits + output_size);
        
        std::cout << "Inference completed successfully!\n";
        print_vector("Output Raw Logits", logits);

        // Calculate and display sigmoid engagement probability
        float prob = 1.0f / (1.0f + std::exp(-logits[0]));
        std::cout << "Sigmoid Probability Score: " << prob << "\n";

    } catch (const Ort::Exception& oe) {
        std::cerr << "ONNX Runtime C++ Error: " << oe.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << "\n";
        return 3;
    }

    return 0;
}
