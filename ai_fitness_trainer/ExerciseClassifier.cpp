#include "ExerciseClassifier.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <direct.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Class names — alphabetical order matching training dataset.
//  The training scripts sort class folders alphabetically.
// ─────────────────────────────────────────────────────────────────────────────
const char* ExerciseClassifier::CLASS_NAMES[] = {
    "barbell_biceps_curl",   // 0
    "bench_press",           // 1
    "chest_fly_machine",     // 2
    "deadlift",              // 3
    "decline_bench_press",   // 4
    "hammer_curl",           // 5
    "hip_thrust",            // 6
    "incline_bench_press",   // 7
    "lat_pulldown",          // 8
    "lateral_raise",         // 9
    "leg_extension",         // 10
    "leg_raises",            // 11
    "plank",                 // 12
    "pull_up",               // 13
    "push_up",               // 14
    "romanian_deadlift",     // 15
    "russian_twist",         // 16
    "shoulder_press",        // 17
    "squat",                 // 18
    "t_bar_row",             // 19
    "tricep_dips",           // 20
    "tricep_pushdown"        // 21
};
const int ExerciseClassifier::NUM_CLASSES = 22;

// constexpr definitions (required in .cpp for non-inline static constexpr)
constexpr float ExerciseClassifier::KINETICS_MEAN[3];
constexpr float ExerciseClassifier::KINETICS_STD[3];

// ─────────────────────────────────────────────────────────────────────────────
ExerciseClassifier::ExerciseClassifier()
    : env(ORT_LOGGING_LEVEL_WARNING, "ExerciseClassifier"),
      session(nullptr),
      memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{}

ExerciseClassifier::~ExerciseClassifier() {
    delete session;
    session = nullptr;
}

// #8 fix: dynamically compute stride to maintain ~10fps temporal sampling
void ExerciseClassifier::setActualFps(double fps) {
    if (fps < 5.0) return;  // ignore nonsense values during startup
    int computed = (int)std::round(fps / TARGET_SAMPLE_FPS);
    if (computed < 1) computed = 1;
    if (computed > 10) computed = 10;  // cap for safety
    frameStride = computed;
}

// ─────────────────────────────────────────────────────────────────────────────
//  loadModel
//  Priority: TensorRT FP16 → CUDA FP32 → CPU
//  Auto-detects image (4D input) vs video (5D input) from ONNX model shape.
// ─────────────────────────────────────────────────────────────────────────────
bool ExerciseClassifier::loadModel(const std::string& modelPath) {
    std::wstring wpath(modelPath.begin(), modelPath.end());
    _mkdir("trt_cache");

    // ── 1. TensorRT FP16 ──────────────────────────────────────────────────
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        OrtTensorRTProviderOptions trtOpts{};
        trtOpts.device_id                = 0;
        trtOpts.trt_max_workspace_size   = (size_t)1 * 1024 * 1024 * 1024;
        trtOpts.trt_fp16_enable          = 1;
        trtOpts.trt_engine_cache_enable  = 1;
        trtOpts.trt_engine_cache_path    = "trt_cache";
        opts.AppendExecutionProvider_TensorRT(trtOpts);

        OrtCUDAProviderOptions cudaFallback{};
        cudaFallback.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cudaFallback);

        session      = new Ort::Session(env, wpath.c_str(), opts);
        providerName = "TensorRT";
        std::cout << "[Classifier] TensorRT FP16 enabled.\n";
    }
    catch (...) { delete session; session = nullptr; }

    // ── 2. CUDA FP32 ──────────────────────────────────────────────────────
    if (!session) {
        try {
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            OrtCUDAProviderOptions cudaOpts{};
            cudaOpts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cudaOpts);

            session      = new Ort::Session(env, wpath.c_str(), opts);
            providerName = "CUDA";
            std::cout << "[Classifier] CUDA FP32 enabled.\n";
        }
        catch (...) { delete session; session = nullptr; }
    }

    // ── 3. CPU fallback ───────────────────────────────────────────────────
    if (!session) {
        try {
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(4);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            session      = new Ort::Session(env, wpath.c_str(), opts);
            providerName = "CPU";
            std::cout << "[Classifier] Running on CPU.\n";
        }
        catch (const Ort::Exception& e) {
            std::cerr << "[Classifier] Load failed: " << e.what() << "\n";
            return false;
        }
    }

    // ── Read I/O names ────────────────────────────────────────────────────
    Ort::AllocatorWithDefaultOptions alloc;
    auto inNamePtr  = session->GetInputNameAllocated(0, alloc);
    auto outNamePtr = session->GetOutputNameAllocated(0, alloc);
    inputNameStr  = inNamePtr.get();
    outputNameStr = outNamePtr.get();
    inputName     = inputNameStr.c_str();
    outputName    = outputNameStr.c_str();

    // ── Detect model type from input tensor rank ──────────────────────────
    auto inputInfo  = session->GetInputTypeInfo(0);
    auto inputShape = inputInfo.GetTensorTypeAndShapeInfo().GetShape();

    if (inputShape.size() == 5) {
        // Video model: [batch, C, T, H, W]
        videoModel = true;
        inputT = (inputShape[2] > 0) ? (int)inputShape[2] : 16;
        inputH = (inputShape[3] > 0) ? (int)inputShape[3] : 112;
        inputW = (inputShape[4] > 0) ? (int)inputShape[4] : 112;
        std::cout << "[Classifier] VIDEO model — T=" << inputT
                  << " H=" << inputH << " W=" << inputW << "\n";
    } else {
        // Image model: [batch, C, H, W]
        videoModel = false;
        inputT = 1;
        inputH = (inputShape.size() >= 3 && inputShape[2] > 0) ? (int)inputShape[2] : 224;
        inputW = (inputShape.size() >= 4 && inputShape[3] > 0) ? (int)inputShape[3] : 224;
        std::cout << "[Classifier] IMAGE model — H=" << inputH
                  << " W=" << inputW << "\n";
    }

    blob.assign(3 * inputT * inputH * inputW, 0.0f);
    frameBuffer.clear();
    voteBuffer.clear();
    callCount         = 0;
    frameStrideCounter = 0;
    frameStride       = TRAINING_STRIDE;  // reset to default, updated by setActualFps()
    lastResult        = ClassificationResult{};

    std::cout << "[Classifier] Ready (" << providerName << "). "
              << inputNameStr << " -> " << outputNameStr << "\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  preprocessImage
//  Resize → BGR→RGB → /255 → CHW layout into blob[]
//  No per-channel normalisation — YOLOv8-cls expects values in [0, 1].
// ─────────────────────────────────────────────────────────────────────────────
void ExerciseClassifier::preprocessImage(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(inputW, inputH));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; c++) {
        std::memcpy(blob.data() + c * inputH * inputW,
                    channels[c].data,
                    inputH * inputW * sizeof(float));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  preprocessVideoFrame
//  Resize → BGR→RGB → /255 → Kinetics-400 normalise → CHW into out[]
// ─────────────────────────────────────────────────────────────────────────────
void ExerciseClassifier::preprocessVideoFrame(const cv::Mat& frame,
                                              std::vector<float>& out) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(inputW, inputH));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    out.resize(3 * inputH * inputW);
    for (int c = 0; c < 3; c++) {
        const float  mean = KINETICS_MEAN[c];
        const float  std  = KINETICS_STD[c];
        const float* src  = channels[c].ptr<float>(0);
        float*       dst  = out.data() + c * inputH * inputW;
        for (int i = 0; i < inputH * inputW; i++) {
            dst[i] = (src[i] - mean) / std;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  assembleVideoBlob
//  Packs frameBuffer (each frame: CHW) into blob[] in BCTHW order:
//    blob[c * T*HW + t * HW + i]  =  frameBuffer[t][c * HW + i]
// ─────────────────────────────────────────────────────────────────────────────
void ExerciseClassifier::assembleVideoBlob() {
    const int HW = inputH * inputW;
    const int T  = inputT;
    for (int c = 0; c < 3; c++) {
        for (int t = 0; t < T; t++) {
            std::memcpy(blob.data() + c * T * HW + t * HW,
                        frameBuffer[t].data() + c * HW,
                        HW * sizeof(float));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  runInference
//  Creates input tensor, runs session, decodes output.
//  Video model:  raw logits → softmax applied here
//  Image model:  YOLOv8-cls has softmax built-in, use raw values
// ─────────────────────────────────────────────────────────────────────────────
ClassificationResult ExerciseClassifier::runInference() {
    ClassificationResult result;

    std::vector<int64_t> inputShape;
    if (videoModel)
        inputShape = { 1, 3, (int64_t)inputT, (int64_t)inputH, (int64_t)inputW };
    else
        inputShape = { 1, 3, (int64_t)inputH, (int64_t)inputW };

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        blob.data(), blob.size(),
        inputShape.data(), inputShape.size());

    std::vector<Ort::Value> outputs;
    try {
        outputs = session->Run(
            Ort::RunOptions{ nullptr },
            &inputName,  &inputTensor, 1,
            &outputName, 1);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[Classifier] Inference error: " << e.what() << "\n";
        return result;
    }

    auto   outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    float* rawData  = outputs[0].GetTensorMutableData<float>();
    int    numCls   = (outShape.size() >= 2) ? (int)outShape[1] : (int)outShape[0];
    numCls = std::min(numCls, NUM_CLASSES);

    // Copy to local buffer so we can optionally apply softmax
    std::vector<float> probs(rawData, rawData + numCls);

    if (videoModel) {
        // R2Plus1D outputs raw logits — apply numerically stable softmax
        float maxLogit = *std::max_element(probs.begin(), probs.end());
        float sum = 0.0f;
        for (float& v : probs) { v = std::exp(v - maxLogit); sum += v; }
        for (float& v : probs) { v /= sum; }
    }
    // else: YOLOv8-cls ONNX already applies Softmax — use as-is

    int   bestIdx  = (int)std::distance(probs.begin(),
                                         std::max_element(probs.begin(), probs.end()));
    float bestConf = probs[bestIdx];

    if (bestConf >= CONFIDENCE_THRESHOLD && bestIdx < NUM_CLASSES) {
        result.classIndex = bestIdx;
        result.className  = CLASS_NAMES[bestIdx];
        result.confidence = bestConf;
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  applyTemporalVoting
//  Majority vote over the last VOTE_WINDOW inference results.
//  Counts votes per class index, picks the winner.
//  Confidence = average confidence of all votes for the winning class.
// ─────────────────────────────────────────────────────────────────────────────
ClassificationResult ExerciseClassifier::applyTemporalVoting() {
    if (voteBuffer.empty()) return lastResult;

    // Count votes and accumulate confidence per class
    std::vector<int>   votes(NUM_CLASSES, 0);
    std::vector<float> confSum(NUM_CLASSES, 0.0f);

    for (const auto& r : voteBuffer) {
        if (r.classIndex >= 0 && r.classIndex < NUM_CLASSES) {
            votes[r.classIndex]++;
            confSum[r.classIndex] += r.confidence;
        }
    }

    // Find class with most votes
    int bestClass = (int)std::distance(votes.begin(),
                                        std::max_element(votes.begin(), votes.end()));
    int bestVotes = votes[bestClass];

    ClassificationResult result;
    if (bestVotes > 0 && bestClass < NUM_CLASSES) {
        result.classIndex = bestClass;
        result.className  = CLASS_NAMES[bestClass];
        result.confidence = confSum[bestClass] / (float)bestVotes; // avg confidence
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  classify — call every frame from capture loop
//  Rate-limiting is handled internally; returns cached lastResult when skipping.
//  Every new inference result is pushed into voteBuffer for temporal smoothing.
// ─────────────────────────────────────────────────────────────────────────────
ClassificationResult ExerciseClassifier::classify(const cv::Mat& frame) {
    if (!session) return lastResult;

    callCount++;

    if (!videoModel) {
        // ── Image mode: run every IMAGE_INFER_INTERVAL frames ────────────
        if (callCount < IMAGE_INFER_INTERVAL) return lastResult;
        callCount = 0;
        preprocessImage(frame);

        ClassificationResult raw = runInference();
        voteBuffer.push_back(raw);
        if ((int)voteBuffer.size() > VOTE_WINDOW)
            voteBuffer.pop_front();

        lastResult = applyTemporalVoting();
        return lastResult;
    }
    else {
        // ── Video mode: sliding-window buffer ─────────────────────────────────
        // FRAME_STRIDE adaptive: only add every Nth camera frame
        // so temporal speed matches training (16 frames × stride ≈ 1.6s).
        frameStrideCounter++;
        if (frameStrideCounter >= frameStride) {
            frameStrideCounter = 0;
            std::vector<float> chw;
            preprocessVideoFrame(frame, chw);
            frameBuffer.push_back(std::move(chw));
            if ((int)frameBuffer.size() > inputT)
                frameBuffer.pop_front();
        }

        // Only run inference when buffer is full AND interval elapsed
        if (callCount < VIDEO_INFER_INTERVAL || (int)frameBuffer.size() < inputT)
            return lastResult;

        callCount = 0;
        assembleVideoBlob();

        ClassificationResult raw = runInference();
        voteBuffer.push_back(raw);
        if ((int)voteBuffer.size() > VOTE_WINDOW)
            voteBuffer.pop_front();

        lastResult = applyTemporalVoting();
        return lastResult;
    }
}
