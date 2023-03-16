#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    // Generate and write 1000 random RGB uint8 images to disk
    int nims = 1000;
    int h = 512, w = 512;
    for (int i = 0; i < nims; ++i) {
        cv::Mat im(h, w, CV_8UC3);

        cv::randu(im, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        cv::imwrite("im1_" + std::to_string(i) + ".jpg", im);

        cv::randu(im, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        cv::imwrite("im2_" + std::to_string(i) + ".jpg", im);
    }

    auto start = std::chrono::high_resolution_clock::now();
    float tot0 = 0, tot1 = 0, tot2 = 0;
    for (int i = 0; i < nims; ++i) {
        auto s0 = std::chrono::high_resolution_clock::now();
        cv::Mat im1 = cv::imread("im1_" + std::to_string(i) + ".jpg");
        cv::Mat im2 = cv::imread("im1_" + std::to_string(i) + ".jpg");

        if (im1.empty() || im2.empty()) {
            std::cerr << "Failed to read image" << std::endl;
            return 1;
        }
        auto e0 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d0 = e0 - s0;
        tot0 += d0.count();

        auto s1 = std::chrono::high_resolution_clock::now();
        cv::Mat res1(im1.rows, im1.cols, CV_8UC3);
        for (int r = 0; r < im1.rows; ++r) {
            for (int c = 0; c < im1.cols; ++c) {
                cv::Vec3b p1 = im1.at<cv::Vec3b>(r, c);
                cv::Vec3b p2 = im2.at<cv::Vec3b>(r, c);
                res1.at<cv::Vec3b>(r, c) = p1 - p2;
            }
        }
        auto e1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d1 = e1 - s1;
        tot1 += d1.count();

        auto s2 = std::chrono::high_resolution_clock::now();
        cv::Mat res2;
        cv::subtract(im1, im2, res2);
        auto e2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> d2 = e2 - s2;
        tot2 += d2.count();

        /*
        cv::Mat d;
        cv::subtract(res1, res2, d);

        cv::Scalar sum_channels = cv::sum(d);
        float total_sum = sum_channels[0] + sum_channels[1] + sum_channels[2];
        std::cout << "Total sum of all channels: " << total_sum << std::endl;
        */
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Load: " << tot0 << std::endl;
    std::cout << "Loop: " << tot1 << std::endl;
    std::cout << "Subtract: " << tot2 << std::endl;
    std::cout << "Total (CPU): " << diff.count() << std::endl;
    return 0;
}
