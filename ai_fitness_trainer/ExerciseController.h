#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include "PoseEstimator.h"
#include "BiomechanicsMath.h"

enum class ExerciseType {
    BicepCurl,        // 0
    Squat,            // 1
    PushUp,           // 2
    Lunge,            // 3
    ShoulderPress,    // 4
    Deadlift,         // 5
    LateralRaise,     // 6
    Plank,            // 7
    TricepPushdown,   // 8
    PullUp,           // 9
    HipThrust,        // 10
    LegExtension,     // 11
    BenchPress,       // 12
    ChestFly,         // 13
    DeclineBenchPress,// 14
    HammerCurl,       // 15
    InclineBenchPress,// 16
    LatPulldown,      // 17
    LegRaises,        // 18
    RomanianDeadlift, // 19
    RussianTwist,     // 20
    TBarRow,          // 21
    TricepDips        // 22
};

enum class ExerciseState {
    Up,
    Down
};

// ─────────────────────────────────────────────────────────────────────────────
//  ExerciseConfig — all thresholds for one exercise type in one place.
//  upThreshold:   angle above which state becomes Up (or Down, depending on direction)
//  downThreshold: angle below which a rep is counted
//  upIsStart:     true = start at high angle → go low → count rep (e.g. BicepCurl, Squat)
//                 false = start at low angle → go high → count rep (e.g. ShoulderPress)
//  fullROM / partialROM: range thresholds for rep quality
//  left/right A,B,C: COCO keypoint indices for the 3-point angle
//  feedbackUp/Down: feedback message shown during each phase
// ─────────────────────────────────────────────────────────────────────────────
struct ExerciseConfig {
    double upThreshold;
    double downThreshold;
    bool   upIsStart;     // true = high angle = ready position; false = low angle = ready
    double fullROM;
    double partialROM;
    int    leftA, leftB, leftC;
    int    rightA, rightB, rightC;
    const char* feedbackUp;
    const char* feedbackDown;
};

class ExerciseController {
public:
    ExerciseController();
    ~ExerciseController();

    void setExerciseType(ExerciseType type);
    void update(const std::vector<Keypoint>& keypoints);
    int getRepCount() const;
    std::string getFeedback() const;
    double getCurrentAngle() const;
    cv::Point2f getLastJointPosition() const;
    void getActiveKeypoints(cv::Point2f& p1, cv::Point2f& p2, cv::Point2f& p3) const;

    // Rep quality
    std::string getLastRepQuality() const;
    int getFullReps() const;
    int getPartialReps() const;
    int getPoorReps() const;

    // Plank
    bool isHoldExercise() const;
    double getHoldTimeSec() const;

    void reset();

    // Map classifier class index (0-21) to ExerciseType enum
    static ExerciseType mapClassifierToExercise(int classIndex);
    static const char*  getExerciseDisplayName(ExerciseType type);

    // Get current exercise type
    ExerciseType getCurrentExercise() const { return currentExercise; }

    // Get config for a given exercise type
    static const ExerciseConfig& getConfig(ExerciseType type);

private:
    ExerciseType currentExercise;
    ExerciseState currentState;
    int repCount;
    double lastAngle;
    double smoothedAngle;
    cv::Point2f lastP1, lastJointPos, lastP3;
    std::string feedback;

    // ── Side-locking for temporal consistency ────────────────────────────
    int  lastChosenSide       = 0;   // 0=none, 1=left, 2=right
    int  sideConsistencyCount = 0;   // consecutive frames on same side
    static constexpr int   SIDE_LOCK_FRAMES    = 3;    // lock after N frames (was 5 — faster lock)
    static constexpr float SIDE_LOCK_DROP_CONF = 0.5f; // unlock if conf drops below (was 0.3 — releases faster)

    // Rep quality tracking
    double maxAngleInRep;
    double minAngleInRep;
    std::string lastRepQuality;
    int fullReps;
    int partialReps;
    int poorReps;

    // Plank hold tracking (uses real elapsed time)
    double holdAccumulator;
    double holdTimeSec;
    std::chrono::steady_clock::time_point lastPlankTime;
    bool   plankTimeInit;

    // Rep debounce — prevents false reps from oscillation at threshold
    std::chrono::steady_clock::time_point lastRepTime;
    static constexpr double kMinRepIntervalSec = 0.5;

    static constexpr double kSmoothingAlpha = 0.3;

    // Smoothing + angle update helper
    void applySmoothingAndTrackRange(double rawAngle);

    // Data-driven exercise processor (replaces 11 individual methods)
    void processGenericExercise(const std::vector<Keypoint>& keypoints, const ExerciseConfig& cfg);

    // Plank still has dedicated method (hold exercise, not rep-based)
    void processPlank(const std::vector<Keypoint>& keypoints);

    bool pickBestSideAndComputeAngle(
        const std::vector<Keypoint>& keypoints,
        int leftA, int leftB, int leftC,
        int rightA, int rightB, int rightC,
        double& outAngle);

    void classifyAndCountRep(const ExerciseConfig& cfg);

    // Static config table
    static const ExerciseConfig CONFIGS[23];
};
