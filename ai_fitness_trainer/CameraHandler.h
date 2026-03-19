#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class CameraHandler {
public:
    CameraHandler();
    ~CameraHandler();

    bool initCamera(int cameraIndex = 1);
    bool initCamera(const std::string& source);  // URL/RTSP/HTTP stream
    bool readFrame(cv::Mat& frame);
    void releaseCamera();
    bool isCameraOpen() const;

private:
    cv::VideoCapture cap;
};
