#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <deque>
#include <string>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
//  Classification result
// ─────────────────────────────────────────────────────────────────────────────
struct ClassificationResult {
    int         classIndex = -1;       // -1 = no valid prediction yet
    std::string className  = "Unknown";
    float       confidence = 0.0f;
};

// ─────────────────────────────────────────────────────────────────────────────
//  ExerciseClassifier
//
//  Supports TWO model types — auto-detected from the loaded ONNX input shape:
//
//  IMAGE model  (exercise_classifier.onnx):
//    Input  : [1, 3, 224, 224]             — single frame
//    Output : [1, 22]                       — probabilities (softmax built-in)
//    Rate   : runs ONNX every 30 calls to classify()
//
//  VIDEO model  (video_classifier.onnx):
//    Input  : [1, 3, 16, 112, 112]  BCTHW  — 16-frame sliding window
//    Output : [1, 22]                       — raw logits (softmax applied here)
//    Rate   : pushes one frame per classify() call;
//             runs ONNX every VIDEO_INFER_INTERVAL calls once buffer is full
//
//  Inference priority for BOTH models:  TensorRT FP16 → CUDA FP32 → CPU
// ─────────────────────────────────────────────────────────────────────────────
class ExerciseClassifier {
public:
    ExerciseClassifier();
    ~ExerciseClassifier();

    // Load model — auto-detects image vs video from input tensor rank
    bool loadModel(const std::string& modelPath);

    // Call every frame from the capture loop.
    // Handles rate-limiting internally.
    // Returns a valid result (classIndex >= 0) when a new prediction is ready;
    // otherwise returns last cached result (or empty on first N frames).
    ClassificationResult classify(const cv::Mat& frame);

    bool               isLoaded()    const { return session != nullptr; }
    bool               isVideoMode() const { return videoModel; }
    const std::string& getProviderName() const { return providerName; }

    // #8 fix: Set actual camera FPS so FRAME_STRIDE adapts dynamically.
    // Training used stride=3 at 30fps (≈10fps sampling).
    // Call this when FPS is known (e.g. after first second of capture).
    void setActualFps(double fps);

    // 22 exercise classes — alphabetical order matching training dataset
    static const char* CLASS_NAMES[];
    static const int   NUM_CLASSES;

private:
    // ── ONNX Runtime ──────────────────────────────────────────────────────────
    Ort::Env         env;
    Ort::Session*    session = nullptr;
    Ort::MemoryInfo  memInfo;
    std::string      providerName = "None";

    // I/O names (read dynamically from model)
    std::string inputNameStr, outputNameStr;
    const char* inputName  = nullptr;
    const char* outputName = nullptr;

    // ── Model config (set at load time) ───────────────────────────────────────
    bool videoModel = false;
    int  inputH     = 224;   // height
    int  inputW     = 224;   // width
    int  inputT     = 1;     // temporal frames (16 for video, 1 for image)

    // ── Inference blob & rate control ─────────────────────────────────────────
    std::vector<float> blob;          // inference input buffer
    int                callCount = 0; // frames seen since last inference

    static constexpr int IMAGE_INFER_INTERVAL = 30;  // ~1.5s at 20fps
    static constexpr int VIDEO_INFER_INTERVAL = 8;   // ~0.4s at 20fps

    // ── Video-mode temporal stride ─────────────────────────────────────────────
    // Training used FRAME_STRIDE=3 at 30fps (≈10fps sampling).
    // We dynamically compute stride = round(actualFps / targetSamplingFps),
    // where targetSamplingFps = trainingFps / trainingStride = 30/3 = 10.
    static constexpr int    TRAINING_STRIDE = 3;
    static constexpr double TRAINING_FPS    = 30.0;
    static constexpr double TARGET_SAMPLE_FPS = TRAINING_FPS / (double)TRAINING_STRIDE; // 10.0
    int                  frameStride        = TRAINING_STRIDE; // adaptive, starts at training default
    int                  frameStrideCounter = 0;

    // ── Video-mode frame buffer ────────────────────────────────────────────────
    // Each element = one preprocessed frame in CHW float32 (Kinetics-normalised)
    std::deque<std::vector<float>> frameBuffer;
    ClassificationResult           lastResult;  // cached until next inference

    // ── Temporal voting buffer ─────────────────────────────────────────────────
    // Stores the last VOTE_WINDOW raw inference results.
    // Majority vote is applied before returning to caller.
    static constexpr int VOTE_WINDOW = 7;
    std::deque<ClassificationResult> voteBuffer;

    // Minimum confidence to return a valid result.
    // Lowered to 0.35 so top-1 prediction is visible even when uncertain.
    // MainForm uses colour-coding to indicate confidence level.
    static constexpr float CONFIDENCE_THRESHOLD = 0.35f;

    // ── Kinetics-400 normalisation constants (for video model) ────────────────
    static constexpr float KINETICS_MEAN[3] = {0.43216f, 0.394666f, 0.37645f};  // RGB
    static constexpr float KINETICS_STD[3]  = {0.22803f,  0.22145f, 0.216989f};

    // ── Helpers ───────────────────────────────────────────────────────────────
    // Image mode: preprocess frame directly into blob[]
    void preprocessImage(const cv::Mat& frame);

    // Video mode: preprocess one frame into out[] (CHW, Kinetics-normalised)
    void preprocessVideoFrame(const cv::Mat& frame, std::vector<float>& out);

    // Video mode: assemble frameBuffer → blob[] in BCTHW layout
    void assembleVideoBlob();

    // Run ONNX session and decode output
    ClassificationResult runInference();

    // Majority vote over voteBuffer → stable top-1 result
    ClassificationResult applyTemporalVoting();
};
