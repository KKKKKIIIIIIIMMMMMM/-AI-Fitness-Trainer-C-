#include "CameraHandler.h"

CameraHandler::CameraHandler() {}

CameraHandler::~CameraHandler() {
    releaseCamera();
}

bool CameraHandler::initCamera(int cameraIndex) {
    if (cap.isOpened()) cap.release();

    // DirectShow is typically faster than MSMF on Windows
    if (!cap.open(cameraIndex, cv::CAP_DSHOW)) {
        // Fallback to default backend if DirectShow fails
        if (!cap.open(cameraIndex)) return false;
    }

    // Set optimal camera parameters
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 854);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS,          30);
    cap.set(cv::CAP_PROP_BUFFERSIZE,   1);   // only keep latest frame → less latency

    return true;
}

bool CameraHandler::initCamera(const std::string& source) {
    if (cap.isOpened()) cap.release();
    if (!cap.open(source)) return false;
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    return true;
}

bool CameraHandler::readFrame(cv::Mat& frame) {
    if (!cap.isOpened()) return false;
    return cap.read(frame);
}

void CameraHandler::releaseCamera() {
    if (cap.isOpened()) cap.release();
}

bool CameraHandler::isCameraOpen() const {
    return cap.isOpened();
}
