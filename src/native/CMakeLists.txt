cmake_minimum_required(VERSION 3.12)
project(native_inference LANGUAGES CXX)

# Enforce strict C++17 compilation
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable custom path configuration via: -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "ONNX Runtime root installation path")

# Locate header files directory
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT_DIR}/include
        ${ONNXRUNTIME_ROOT_DIR}/include/onnxruntime
        /opt/homebrew/include
        /opt/homebrew/include/onnxruntime
        /usr/local/include
        /usr/local/include/onnxruntime
)

# Locate library binary file (.dylib or .so)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT_DIR}/lib
        /opt/homebrew/lib
        /usr/local/lib
)

# Enforce library resolution check
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR 
        "ONNX Runtime library or headers not found.\n"
        "Please install via Homebrew ('brew install onnxruntime') or specify "
        "the installation directory root using -DONNXRUNTIME_ROOT_DIR=<path_to_onnxruntime_root>"
    )
endif()

message(STATUS "Found ONNX Runtime Headers: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "Found ONNX Runtime Library: ${ONNXRUNTIME_LIBRARY}")

# Locate pybind11 by querying python in the current virtual environment
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_FIND_RESULT
)

if (PYBIND11_FIND_RESULT EQUAL 0)
    message(STATUS "Found pybind11 directory via python: ${pybind11_DIR}")
    find_package(pybind11 REQUIRED)
else()
    find_package(pybind11 REQUIRED)
endif()

# Add the compiled executable target
add_executable(inference_engine inference_engine.cpp)
target_include_directories(inference_engine PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(inference_engine PRIVATE ${ONNXRUNTIME_LIBRARY})

# Add the compiled benchmark target
add_executable(benchmark_engine benchmark_engine.cpp)
target_include_directories(benchmark_engine PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(benchmark_engine PRIVATE ${ONNXRUNTIME_LIBRARY})

# Add the pybind11 python module
pybind11_add_module(native_inference_py binding.cpp)
target_include_directories(native_inference_py PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(native_inference_py PRIVATE ${ONNXRUNTIME_LIBRARY})

# Compile target into the src/native directory directly for easy python imports
set_target_properties(native_inference_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
