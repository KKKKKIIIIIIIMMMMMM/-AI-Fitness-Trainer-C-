#pragma once

#include <string>
#include <vector>
#include "PoseEstimator.h"
#include "BiomechanicsMath.h"

// Forward declaration
enum class ExerciseType;

// ─────────────────────────────────────────────────────────────────────────────
//  Risk levels
// ─────────────────────────────────────────────────────────────────────────────
enum class RiskLevel {
    Safe,       // Green  — no issues
    Warning,    // Yellow — minor form issue
    Danger      // Red    — injury risk
};

struct InjuryRisk {
    RiskLevel   level   = RiskLevel::Safe;
    std::string message = "Good form";
};

// ─────────────────────────────────────────────────────────────────────────────
//  InjuryRiskAssessor
//  Rule-based biomechanical safety checks per exercise.
//  Uses keypoint positions + angles to detect dangerous form patterns.
// ─────────────────────────────────────────────────────────────────────────────
class InjuryRiskAssessor {
public:
    InjuryRiskAssessor();
    ~InjuryRiskAssessor();

    // Assess form for the given exercise using current keypoints
    // frameHeight: camera frame height in pixels (for resolution-independent checks)
    InjuryRisk assess(ExerciseType exercise,
                      const std::vector<Keypoint>& keypoints,
                      float frameHeight = 480.0f);

private:
    // Exercise-specific assessors (frameHeight for resolution-independent thresholds)
    InjuryRisk assessSquat(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessDeadlift(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessBicepCurl(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessShoulderPress(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessPushUp(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessLunge(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessLateralRaise(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessPlank(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessTricepPushdown(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessPullUp(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessHipThrust(const std::vector<Keypoint>& kp, float fH);
    InjuryRisk assessLegExtension(const std::vector<Keypoint>& kp, float fH);

    // Confidence threshold for rule checks
    static constexpr float MIN_CONF = 0.4f;

    // Helper: pick best-confidence keypoint between left/right
    bool getBestSide(const std::vector<Keypoint>& kp,
                     int leftIdx, int rightIdx,
                     cv::Point2f& out, float& outConf) const;
};
