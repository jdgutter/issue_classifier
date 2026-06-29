#include <cmath>
#include <cstdint>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

class NativeInfoInferenceEngine {
private:
  Ort::Env env;
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;

  // Extract input/output names defined during ONNX export
  std::vector<const char *> input_names = {
      "continuous_features", "categorical_features", "text_embeddings"};
  std::vector<const char *> output_names = {"logits"};

public:
  NativeInfoInferenceEngine(const std::string &model_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "High_Performance_Inference") {

    // Single-threaded CPU execution for determinism
    session_options.SetIntraOpNumThreads(1);

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Load session
    session = std::make_unique<Ort::Session>(env, model_path.c_str(),
                                             session_options);
  }

  std::vector<float>
  predict_probabilities(const std::vector<float> &continuous_flat,
                        const std::vector<int64_t> &categorical_flat,
                        const std::vector<float> &embedding_flat,
                        int64_t batch_size) {
    if (batch_size <= 0)
      return {};

    // Bounds check input vector size to avoid memory faults
    if (continuous_flat.size() != static_cast<size_t>(batch_size * 3)) {
      throw std::invalid_argument(
          "continuous_flat size must be batch_size * 3");
    }

    if (categorical_flat.size() != static_cast<size_t>(batch_size * 2)) {
      throw std::invalid_argument(
          "categorical_flat size must be batch_size * 2");
    }

    if (embedding_flat.size() != static_cast<size_t>(batch_size * 384)) {
      throw std::invalid_argument(
          "embedding_flat size must be batch_size * 384");
    }

    std::vector<int64_t> continuous_shape = {batch_size, 3};
    std::vector<int64_t> categorical_shape = {batch_size, 2};
    std::vector<int64_t> embedding_shape = {batch_size, 384};

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value continuous_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(continuous_flat.data()),
        continuous_flat.size(), continuous_shape.data(),
        continuous_shape.size());

    Ort::Value categorical_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t *>(categorical_flat.data()),
        categorical_flat.size(), categorical_shape.data(),
        categorical_shape.size());

    Ort::Value embedding_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(embedding_flat.data()),
        embedding_flat.size(), embedding_shape.data(),
        embedding_shape.size());

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(continuous_tensor));
    input_tensors.push_back(std::move(categorical_tensor));
    input_tensors.push_back(std::move(embedding_tensor));

    // Run inference
    auto output_tensors = session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        output_names.data(),
        output_names.size()
    );

    // Extract raw logits and calculate Sigmoid probability score element-wise
    float* raw_logits = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    std::vector<float> probabilities(output_size);
    for (size_t i = 0; i < output_size; ++i) {
      probabilities[i] = 1.0f / (1.0f + std::exp(-raw_logits[i]));
    }

    return probabilities;
  }
};

PYBIND11_MODULE(native_inference_py, m) {
  m.doc() = "ONNX Runtime C++ Inference Engine Python bindings";

  py::class_<NativeInfoInferenceEngine>(m, "InferenceEngine")
      .def(py::init<const std::string &>(), py::arg("model_path"))
      .def("predict_probabilities", &NativeInfoInferenceEngine::predict_probabilities,
           py::arg("continuous_flat"), py::arg("categorical_flat"),
           py::arg("embeddings_flat"), py::arg("batch_size"));
}