#include <chrono>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <optional>
#include <thread>

#include "onnxruntime_cxx_api.h"

class OrtInference {
   public:
    OrtInference(const std::string& model_path, const cv::Scalar& mean, const cv::Scalar& sd)
        : mean(mean), sd(sd) {
        // Create an ONNX Runtime session and load the model
        env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "LOG");
        Ort::SessionOptions options;
        /*
        Tried 0 -> std::thread::hardware_concurrency(), it seems 4 works the
        best?
        */
        options.SetIntraOpNumThreads(4);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.DisableProfiling();
        session = Ort::Session(env, model_path.c_str(), options);

        // Assume that we only have an input and an output
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
            throw std::runtime_error("Currently, only support one input and one output, got " +
                                     std::to_string(session.GetInputCount()) + " input(s), and " +
                                     std::to_string(session.GetOutputCount()) + " output(s)");

        // Get the input and output info
        Ort::AllocatorWithDefaultOptions allocator;
        inputNamePtr.emplace(session.GetInputNameAllocated(0, allocator));
        outputNamePtr.emplace(session.GetOutputNameAllocated(0, allocator));

        inputName = inputNamePtr->get();
        outputName = outputNamePtr->get();

        inputDim = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        outputDim = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

        inputTensor = Ort::Value::CreateTensor<float>(allocator, inputDim.data(), inputDim.size());
        outputTensor =
            Ort::Value::CreateTensor<float>(allocator, outputDim.data(), outputDim.size());

        // Assumption
        if (size(inputDim) != 4)
            throw std::runtime_error("Support an 4 dimensional input, got " +
                                     std::to_string(size(inputDim)));
        if (inputDim[0] != 1)
            throw std::runtime_error("Support a batch size of 1, got " +
                                     std::to_string(inputDim[0]));
    }

   public:
    void preprocess(const cv::Mat& image) {
        int height = image.rows;
        int width = image.cols;
        int channels = image.channels();

        // Assumption
        if (channels != 3 || image.type() != CV_8UC3)
            throw std::runtime_error("Output must be 8UC3!");

        // Resize
        cv::Mat resized;
        if (height != inputDim[2] || width != inputDim[3])
            cv::resize(image, resized, cv::Size(inputDim[3], inputDim[2]));
        else
            resized = image;

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
        f32_image /= sd;

        std::vector<cv::Mat> split;
        cv::split(f32_image, split);

        float* inputData = inputTensor.GetTensorMutableData<float>();
        for (int c = 0; c < channels; ++c) {
            std::memcpy(inputData + c * inputDim[2] * inputDim[3], split[c].data,
                        inputDim[2] * inputDim[3] * sizeof(float));
        }
    }

    void inference() {
        try {
            session.Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName,
                        &outputTensor, 1);
        } catch (const Ort::Exception& ex) {
            throw std::runtime_error(ex.what());
        }
    }
    std::tuple<int, float> postprocess() {
        float* outputData = outputTensor.GetTensorMutableData<float>();

        float sum =
            std::accumulate(outputData, outputData + outputDim[1], 0.0,
                            [](const float& acc, const float& e) { return acc + std::exp(e); });
        auto it = std::max_element(outputData, outputData + outputDim[1]);
        int idx = std::distance(outputData, it);
        return {idx, std::exp(*it) / sum};
    }

   private:
    Ort::Env env{nullptr};
    Ort::Session session{nullptr};
    std::optional<Ort::AllocatedStringPtr> inputNamePtr;
    std::optional<Ort::AllocatedStringPtr> outputNamePtr;

    const char* inputName;
    const char* outputName;

    std::vector<int64_t> inputDim;
    std::vector<int64_t> outputDim;

    Ort::Value inputTensor{nullptr};
    Ort::Value outputTensor{nullptr};

    cv::Scalar mean;
    cv::Scalar sd;
    float scale = 1.0 / 255;
};

int main() {
    int h = 512, w = 512;
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar sd(0.229, 0.224, 0.225);
    OrtInference sess("/model.onnx", mean, sd);

    int ntimes = 100;
    double t_pre = 0, t_model = 0, t_post = 0;
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> diff;
    for (int i = 0; i < ntimes; ++i) {
        // Generate an image using OpenCV
        cv::Mat image(h, w, CV_8UC3);
        cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        start = std::chrono::high_resolution_clock::now();
        sess.preprocess(image);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        t_pre += diff.count();

        start = std::chrono::high_resolution_clock::now();
        sess.inference();
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        t_model += diff.count();

        start = std::chrono::high_resolution_clock::now();
        auto [pred, prob] = sess.postprocess();
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        t_post += diff.count();

        // std::cout << "Prediction: " << pred << " - Probability: " << prob << std::endl;
    }
    std::cout << "ONNXRuntime (CPU): t_pre " << t_pre / ntimes << ", t_model " << t_model / ntimes
              << ", t_post " << t_post / ntimes << std::endl;

    return 0;
}
