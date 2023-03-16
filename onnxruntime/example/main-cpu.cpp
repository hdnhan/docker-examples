#include <chrono>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "onnxruntime_cxx_api.h"

using namespace cv;

void transforms(const std::vector<cv::Mat>& images, float* inputData,
                const std::vector<int64_t>& inputDims, const cv::Scalar& mean,
                const cv::Scalar& std, const float& scale) {
    if (images.size() != inputDims[0])
        throw std::runtime_error(
            "Number of input images is " + std::to_string(images.size()) +
            ", but batch size is " + std::to_string(inputDims[0]));
    if (inputDims.size() != 4)
        throw std::runtime_error("Dims mismatch, expcted 4, got " +
                                 std::to_string(inputDims.size()));

    size_t hw = inputDims[2] * inputDims[3];
    size_t chw = inputDims[1] * inputDims[2] * inputDims[3];
    int i = 0;
    for (int i = 0; i < inputDims[0]; ++i) {
        int height = images[i].rows;
        int width = images[i].cols;
        int channels = images[i].channels();

        // Validate
        if (channels != 3 || images[i].type() != CV_8UC3)
            throw std::runtime_error("Output must be 8UC3!");

        // Resize
        cv::Mat resized;
        if (height != inputDims[2] || width != inputDims[3])
            cv::resize(images[i], resized,
                       cv::Size(inputDims[3], inputDims[2]));
        else
            resized = images[i];

        // Swap R <-> B
        cv::Mat swapped;
        cv::cvtColor(resized, swapped, cv::COLOR_BGR2RGB);

        // Check type
        cv::Mat f32_image;
        if (swapped.type() != CV_32FC3)
            swapped.convertTo(f32_image, CV_32FC3);
        else
            f32_image = swapped;

        // Rescale and  Normalize
        f32_image *= scale;
        f32_image -= mean;
        f32_image /= std;

        std::vector<cv::Mat> split;
        cv::split(f32_image, split);

        for (int j = 0; j < channels; ++j) {
            std::memcpy(inputData + i * chw + j * hw, split[i].data,
                        hw * sizeof(float));
        }
    }
}

int main() {
    // Generate an image using OpenCV
    int h = 512, w = 512;
    cv::Mat image(h, w, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    float scale = 1.0 / 255;

    // Create an ONNX Runtime session and load the model
    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, "/model.onnx", session_options);

    // Assume that we only have an input and an output
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
        throw std::runtime_error(
            "Currently, only support one input and one output, got " +
            std::to_string(session.GetInputCount()) + "/" +
            std::to_string(session.GetOutputCount()) + "inputs/outputs");

    // Get the input and output info
    Ort::AllocatorWithDefaultOptions allocator;

    auto inputPtr = session.GetInputNameAllocated(0, allocator);
    const char* inputName = inputPtr.get();
    std::vector<int64_t> inputDims(
        session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape());
    Ort::Value inputTensor(Ort::Value::CreateTensor<float>(
        allocator, inputDims.data(), inputDims.size()));

    auto outputPtr = session.GetOutputNameAllocated(0, allocator);
    const char* outputName = outputPtr.get();
    std::vector<int64_t> outputDims(
        session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape());
    Ort::Value outputTensor(Ort::Value::CreateTensor<float>(
        allocator, outputDims.data(), outputDims.size()));

    std::cout << inputName << std::endl;
    for (auto e : inputDims) std::cout << e << " ";
    std::cout << std::endl;

    std::cout << outputName << std::endl;
    for (auto e : outputDims) std::cout << e << " ";
    std::cout << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    // Copy image data to input tensor
    transforms({image}, inputTensor.GetTensorMutableData<float>(), inputDims,
               mean, std, scale);

    // Run the model and get the output
    session.Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1,
                &outputName, &outputTensor, 1);
    float* outputData = outputTensor.GetTensorMutableData<float>();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "ONNXRuntime (CPU): " << diff.count() << std::endl;

    // Print the top 5 predictions
    std::vector<float> scores(outputData, outputData + 1000);
    std::vector<size_t> idx(1000);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(
        idx.begin(), idx.begin() + 5, idx.end(),
        [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });
    for (int i = 0; i < 5; ++i) {
        std::cout << "Prediction #" << i << ": " << idx[i] << " ("
                  << scores[idx[i]] << ")" << std::endl;
    }

    return 0;
}
