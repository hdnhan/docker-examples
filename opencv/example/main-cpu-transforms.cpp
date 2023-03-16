#include <chrono>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

std::vector<cv::Mat> transforms(const std::vector<cv::Mat>& images,
                                const std::vector<int> &shape,
                                const cv::Scalar& mean,
                                const cv::Scalar& std,
                                const float& scale) {
    std::vector<cv::Mat> transformed;
    for (cv::Mat image : images) {
        int height = image.rows;
        int width = image.cols;
        int channels = image.channels();

        // Validate
        if (channels != 3 || image.type() != CV_8UC(channels))
            throw std::runtime_error("Output must be 8UC3!");

        // Resize
        cv::Mat resized;
        if (height != shape[0] || width != shape[1])
            cv::resize(image, resized, cv::Size(shape[1], shape[0]));
        else
            resized = image;

        // Swap R <-> B
        cv::Mat swapped;
        cv::cvtColor(resized, swapped, cv::COLOR_BGR2RGB);

        // Check type
        cv::Mat f32_image;
        if (swapped.type() != CV_32FC(channels))
            swapped.convertTo(f32_image, CV_32FC(channels));
        else
            f32_image = swapped;

        // Rescale and  Normalize
        f32_image *= scale;
        f32_image -= mean;
        f32_image /= std;

        transformed.emplace_back(f32_image);
    }
    return transformed;
}

int main() {
    int nims = 1000;
    int h = 512, w = 512;
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar std(0.229, 0.224, 0.225);
    float scale = 1 / 255.0;

    float total = 0;
    for (int i = 0; i < nims; ++i) {
        cv::Mat im(h, w, CV_8UC3);
        cv::randu(im, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        auto start = std::chrono::high_resolution_clock::now();
        auto transformed = transforms({im}, {640, 640}, mean, std, scale);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        total += diff.count();
    }
    std::cout << "Transforms (CPU): " << total / nims << std::endl;
    return 0;
}
