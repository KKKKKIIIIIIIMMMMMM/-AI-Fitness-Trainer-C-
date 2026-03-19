#include "PoseEstimator.h"
#include <iostream>
#include <direct.h>   // _mkdir (Windows)

// ─────────────────────────────────────────────────────────────────────────────
PoseEstimator::PoseEstimator()
    : env(ORT_LOGGING_LEVEL_WARNING, "PoseEstimator"),
      session(nullptr),
      memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    // Pre-allocate inference blob once (3 × 640 × 640 × 4 bytes ≈ 4.7 MB)
    blob.resize(3 * inputHeight * inputWidth);
    filters.resize(17);
}

PoseEstimator::~PoseEstimator() {
    delete session;
    session = nullptr;
}

// ─────────────────────────────────────────────────────────────────────────────
//  loadModel
//  Priority:  TensorRT FP16  →  CUDA FP32  →  CPU
//
//  TensorRT first launch:  compiles & caches the engine (~30–60 sec).
//  Every later launch:     loads the cached engine instantly.
// ─────────────────────────────────────────────────────────────────────────────
bool PoseEstimator::loadModel(const std::string& modelPath) {
    std::wstring wpath(modelPath.begin(), modelPath.end());
    cudaError.clear();

    // Ensure the TRT engine cache directory exists
    _mkdir("trt_cache");

    // ── 1. Try TensorRT FP16 ────────────────────────────────────────────────
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        OrtTensorRTProviderOptions trtOpts{};
        trtOpts.device_id                 = 0;
        trtOpts.trt_max_partition_iterations = 1000;
        trtOpts.trt_min_subgraph_size     = 1;
        trtOpts.trt_max_workspace_size    = (size_t)2 * 1024 * 1024 * 1024; // 2 GB
        trtOpts.trt_fp16_enable           = 1;   // FP16 → ~2× faster than FP32
        trtOpts.trt_engine_cache_enable   = 1;   // cache compiled engine to disk
        trtOpts.trt_engine_cache_path     = "trt_cache";
        trtOpts.trt_force_sequential_engine_build = 0;

        // TensorRT handles supported ops; CUDA handles the rest
        opts.AppendExecutionProvider_TensorRT(trtOpts);

        // CUDA fallback for any ops TRT cannot handle
        OrtCUDAProviderOptions cudaFallback{};
        cudaFallback.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cudaFallback);

        session      = new Ort::Session(env, wpath.c_str(), opts);
        gpuEnabled   = true;
        providerName = "TensorRT";
        std::cout << "[PoseEstimator] TensorRT FP16 enabled. Engine cached in trt_cache/\n";
        return true;
    }
    catch (const Ort::Exception& e) {
        cudaError = std::string("TRT: ") + e.what();
        std::cerr << "[PoseEstimator] TensorRT unavailable: " << cudaError << "\n";
        delete session; session = nullptr;
    }
    catch (const std::exception& e) {
        cudaError = std::string("TRT std::exception: ") + e.what();
        std::cerr << "[PoseEstimator] TensorRT unavailable: " << cudaError << "\n";
        delete session; session = nullptr;
    }
    catch (...) {
        cudaError = "TRT: Unknown exception";
        std::cerr << "[PoseEstimator] TensorRT unavailable: unknown\n";
        delete session; session = nullptr;
    }

    // ── 2. Fallback: CUDA FP32 ──────────────────────────────────────────────
    cudaError.clear();
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        OrtCUDAProviderOptions cudaOpts{};
        cudaOpts.device_id              = 0;
        cudaOpts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        cudaOpts.gpu_mem_limit          = (size_t)4 * 1024 * 1024 * 1024;
        cudaOpts.arena_extend_strategy  = 1;
        opts.AppendExecutionProvider_CUDA(cudaOpts);

        session      = new Ort::Session(env, wpath.c_str(), opts);
        gpuEnabled   = true;
        providerName = "CUDA";
        std::cout << "[PoseEstimator] CUDA FP32 enabled.\n";
        return true;
    }
    catch (const Ort::Exception& e) {
        cudaError = std::string("CUDA: ") + e.what();
        std::cerr << "[PoseEstimator] CUDA failed: " << cudaError << "\n";
        delete session; session = nullptr;
    }
    catch (const std::exception& e) {
        cudaError = std::string("CUDA std::exception: ") + e.what();
        std::cerr << "[PoseEstimator] CUDA failed: " << cudaError << "\n";
        delete session; session = nullptr;
    }
    catch (...) {
        cudaError = "CUDA: Unknown exception";
        std::cerr << "[PoseEstimator] CUDA failed: unknown\n";
        delete session; session = nullptr;
    }

    // ── 3. Final fallback: CPU ───────────────────────────────────────────────
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session      = new Ort::Session(env, wpath.c_str(), opts);
        gpuEnabled   = false;
        providerName = "CPU";
        std::cout << "[PoseEstimator] Running on CPU (no GPU available).\n";
        return true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[PoseEstimator] CPU load failed: " << e.what() << "\n";
        return false;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  preprocess  — letterbox → float32 NCHW [1,3,640,640]
// ─────────────────────────────────────────────────────────────────────────────
void PoseEstimator::preprocess(const cv::Mat& frame) {
    float scaleW = (float)inputWidth  / frame.cols;
    float scaleH = (float)inputHeight / frame.rows;
    float scale  = std::min(scaleW, scaleH);

    int newW = (int)(frame.cols * scale);
    int newH = (int)(frame.rows * scale);
    int padX = (inputWidth  - newW) / 2;
    int padY = (inputHeight - newH) / 2;

    letterboxScale = scale;
    letterboxPadX  = padX;
    letterboxPadY  = padY;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));

    cv::Mat padded(inputHeight, inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(padX, padY, newW, newH)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    for (int c = 0; c < 3; c++) {
        std::memcpy(blob.data() + c * inputHeight * inputWidth,
                    channels[c].data,
                    inputHeight * inputWidth * sizeof(float));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  runInference
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> PoseEstimator::runInference(const cv::Mat& frame) {
    if (!session || frame.empty()) return {};

    cv::Rect roi(0, 0, frame.cols, frame.rows);

    // If we have a tracked target, crop around it with a margin
    if (lastRoi.area() > 0 && roiLostFrames < 5) {
        int marginX = lastRoi.width * MARGIN_PERCENT / 100;
        int marginY = lastRoi.height * MARGIN_PERCENT / 100;
        
        roi.x = std::max(0, lastRoi.x - marginX);
        roi.y = std::max(0, lastRoi.y - marginY);
        roi.width = std::min(frame.cols - roi.x, lastRoi.width + 2 * marginX);
        roi.height = std::min(frame.rows - roi.y, lastRoi.height + 2 * marginY);
    }

    cv::Mat inputFrame = frame(roi);
    preprocess(inputFrame);

    std::array<int64_t, 4> inputShape = { 1, 3, inputHeight, inputWidth };
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo,
        blob.data(), blob.size(),
        inputShape.data(), inputShape.size());

    auto outputs = session->Run(
        Ort::RunOptions{ nullptr },
        &inputName,  &inputTensor, 1,
        &outputName, 1);

    auto shape      = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t numFeat = shape[1];
    int64_t numDets = shape[2];

    const float* data = outputs[0].GetTensorMutableData<float>();
    auto detections = parseOutput(data, numFeat, numDets, inputFrame.cols, inputFrame.rows);

    // Map coordinates back to full frame and apply anti-jitter smoothing
    if (!detections.empty()) {
        auto& primary = detections[0]; // Assuming largest/most confident is [0] after NMS
        
        // Map bounding box back
        primary.box.x += roi.x;
        primary.box.y += roi.y;
        
        // Update ROI state
        lastRoi = primary.box;
        roiLostFrames = 0;

        // Map and smooth keypoints
        for (int j = 0; j < 17; j++) {
            primary.keypoints[j].pt.x += roi.x;
            primary.keypoints[j].pt.y += roi.y;
            primary.keypoints[j].pt = filters[j].filter(primary.keypoints[j].pt);
        }
    } else {
        roiLostFrames++;
        if (roiLostFrames >= 5) {
            lastRoi = cv::Rect(); // Reset tracking
            for (auto& f : filters) f.first = true; // Reset filters
        }
    }

    return detections;
}

// ─────────────────────────────────────────────────────────────────────────────
//  parseOutput  — letterbox-corrected coordinates
// ─────────────────────────────────────────────────────────────────────────────
std::vector<Detection> PoseEstimator::parseOutput(const float* data,
                                                   int64_t numFeatures,
                                                   int64_t numDetections,
                                                   int frameW, int frameH)
{
    std::vector<cv::Rect>                boxes;
    std::vector<float>                   scores;
    std::vector<std::vector<Keypoint>>   kptsAll;

    for (int64_t i = 0; i < numDetections; i++) {
        float conf = data[4 * numDetections + i];
        if (conf < scoreThreshold) continue;

        float cx = data[0 * numDetections + i];
        float cy = data[1 * numDetections + i];
        float bw = data[2 * numDetections + i];
        float bh = data[3 * numDetections + i];

        float x1 = (cx - bw * 0.5f - letterboxPadX) / letterboxScale;
        float y1 = (cy - bh * 0.5f - letterboxPadY) / letterboxScale;
        float rw  = bw / letterboxScale;
        float rh  = bh / letterboxScale;

        boxes.push_back(cv::Rect((int)x1, (int)y1, (int)rw, (int)rh));
        scores.push_back(conf);

        std::vector<Keypoint> kpts;
        for (int j = 0; j < 17; j++) {
            float kx = (data[(5 + j * 3)     * numDetections + i] - letterboxPadX) / letterboxScale;
            float ky = (data[(5 + j * 3 + 1) * numDetections + i] - letterboxPadY) / letterboxScale;
            float kc =  data[(5 + j * 3 + 2) * numDetections + i];
            kpts.push_back({ cv::Point2f(kx, ky), kc });
        }
        kptsAll.push_back(kpts);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, nmsThreshold, indices);

    std::vector<Detection> detections;
    detections.reserve(indices.size());
    for (int idx : indices)
        detections.push_back({ boxes[idx], scores[idx], kptsAll[idx] });

    return detections;
}

// ─────────────────────────────────────────────────────────────────────────────
//  drawPose
// ─────────────────────────────────────────────────────────────────────────────
static const int SKELETON[][2] = {
    {0,1},{0,2},{1,3},{2,4},{5,6},{5,7},{7,9},{6,8},{8,10},
    {5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}
};
static const int SKELETON_LEN = 16;

void PoseEstimator::drawPose(cv::Mat& frame,
                              const std::vector<Detection>& detections,
                              cv::Scalar boneColor)
{
    for (const auto& det : detections) {
        cv::rectangle(frame, det.box, cv::Scalar(0, 255, 0), 2);

        for (const auto& kp : det.keypoints)
            if (kp.confidence > 0.5f)
                cv::circle(frame, kp.pt, 5, cv::Scalar(0, 0, 255), -1);

        for (int s = 0; s < SKELETON_LEN; s++) {
            const auto& k1 = det.keypoints[SKELETON[s][0]];
            const auto& k2 = det.keypoints[SKELETON[s][1]];
            if (k1.confidence > 0.5f && k2.confidence > 0.5f)
                cv::line(frame, k1.pt, k2.pt, boneColor, 2);
        }
    }
}
