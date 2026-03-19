#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <array>

struct Keypoint {
    cv::Point2f pt;
    float confidence;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Simple Adaptive EMA (1Euro-inspired Filter)
// ─────────────────────────────────────────────────────────────────────────────
struct AdaptiveFilter {
    cv::Point2f curr;
    cv::Point2f prev;
    cv::Point2f dx;
    bool        first = true;
    float       min_cutoff = 0.5f;   // Lower = more smoothing at low speeds
    float       beta       = 0.05f;  // Higher = less lag at high speeds
    float       d_cutoff   = 1.0f;

    static float applyEMA(float val, float prev_val, float alpha) {
        return alpha * val + (1.0f - alpha) * prev_val;
    }

    cv::Point2f filter(cv::Point2f x, float dt = 1.0f/30.0f) {
        if (first) {
            prev = x;
            curr = x;
            dx   = cv::Point2f(0, 0);
            first = false;
            return curr;
        }

        // Compute velocity
        cv::Point2f v = (x - prev) / dt;
        
        // Filter velocity
        float alpha_d = smoothingFactor(d_cutoff, dt);
        dx.x = applyEMA(v.x, dx.x, alpha_d);
        dx.y = applyEMA(v.y, dx.y, alpha_d);

        // Magnitude of velocity (speed)
        float speed = std::sqrt(dx.x * dx.x + dx.y * dx.y);
        
        // Adaptive cutoff based on speed
        float cutoff = min_cutoff + beta * speed;
        float alpha  = smoothingFactor(cutoff, dt);

        // Filter position
        curr.x = applyEMA(x.x, prev.x, alpha);
        curr.y = applyEMA(x.y, prev.y, alpha);
        
        prev = curr;
        return curr;
    }

private:
    float smoothingFactor(float cutoff, float dt) {
        float r = 2.0f * 3.1415926535f * cutoff * dt;
        return r / (r + 1.0f);
    }
};

struct Detection {
    cv::Rect box;
    float score;
    std::vector<Keypoint> keypoints;
};

// ─────────────────────────────────────────────────────────────────────────────
//  PoseEstimator
//  Inference priority:  TensorRT FP16  →  CUDA FP32  →  CPU
//  TensorRT engine is compiled on first run and cached to disk (trt_cache/).
//  Every subsequent launch loads the cached engine and starts instantly.
// ─────────────────────────────────────────────────────────────────────────────
class PoseEstimator {
public:
    PoseEstimator();
    ~PoseEstimator();

    bool loadModel(const std::string& modelPath);
    std::vector<Detection> runInference(const cv::Mat& frame);
    void drawPose(cv::Mat& frame, const std::vector<Detection>& detections,
                  cv::Scalar boneColor = cv::Scalar(255, 0, 0));

    bool               isGPUEnabled()     const { return gpuEnabled; }
    const std::string& getCudaError()     const { return cudaError; }
    const std::string& getProviderName()  const { return providerName; }

private:
    // ONNX Runtime
    Ort::Env            env;
    Ort::Session*       session    = nullptr;
    Ort::MemoryInfo     memInfo;
    bool                gpuEnabled = false;
    std::string         cudaError;
    std::string         providerName = "None";  // "TensorRT" | "CUDA" | "CPU"

    // Model constants
    const int   inputWidth      = 640;
    const int   inputHeight     = 640;
    const float scoreThreshold  = 0.5f;
    const float nmsThreshold    = 0.45f;

    // I/O names
    const char* inputName  = "images";
    const char* outputName = "output0";

    // Pre-allocated blob (avoids per-frame heap churn)
    std::vector<float> blob;

    // Letterbox parameters (set in preprocess, used in parseOutput)
    float letterboxScale = 1.0f;
    int   letterboxPadX  = 0;
    int   letterboxPadY  = 0;

    // ── ROI Tracking Pipeline ──────────────────────────────────────────────────
    cv::Rect lastRoi;              // Holds the previous frame's bounding box
    int      roiLostFrames = 0;    // Counter for lost tracking
    static constexpr int MARGIN_PERCENT = 20; // 20% margin around ROI
    
    // ── Anti-Jitter Keypoint Smoothing ─────────────────────────────────────────
    std::vector<AdaptiveFilter> filters; // 17 filters for 17 keypoints

    // Helpers
    void preprocess(const cv::Mat& frame);
    std::vector<Detection> parseOutput(const float* data,
                                       int64_t numFeatures,
                                       int64_t numDetections,
                                       int frameW, int frameH);
};
