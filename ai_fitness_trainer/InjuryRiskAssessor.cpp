#include "InjuryRiskAssessor.h"
#include "ExerciseController.h"  // for ExerciseType enum
#include <cmath>

// COCO keypoint indices
//  0=Nose, 1=LEye, 2=REye, 3=LEar, 4=REar
//  5=LShoulder, 6=RShoulder, 7=LElbow, 8=RElbow, 9=LWrist, 10=RWrist
// 11=LHip, 12=RHip, 13=LKnee, 14=RKnee, 15=LAnkle, 16=RAnkle

InjuryRiskAssessor::InjuryRiskAssessor() {}
InjuryRiskAssessor::~InjuryRiskAssessor() {}

// ─────────────────────────────────────────────────────────────────────────────
//  getBestSide — picks the left or right keypoint with higher confidence
// ─────────────────────────────────────────────────────────────────────────────
bool InjuryRiskAssessor::getBestSide(const std::vector<Keypoint>& kp,
                                      int leftIdx, int rightIdx,
                                      cv::Point2f& out, float& outConf) const
{
    float lc = kp[leftIdx].confidence;
    float rc = kp[rightIdx].confidence;

    if (lc >= rc && lc > MIN_CONF) {
        out     = kp[leftIdx].pt;
        outConf = lc;
        return true;
    }
    if (rc > MIN_CONF) {
        out     = kp[rightIdx].pt;
        outConf = rc;
        return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main dispatcher — now passes frameHeight to all assessors (#10)
// ─────────────────────────────────────────────────────────────────────────────
InjuryRisk InjuryRiskAssessor::assess(ExerciseType exercise,
                                       const std::vector<Keypoint>& keypoints,
                                       float frameHeight)
{
    if (keypoints.size() < 17) return { RiskLevel::Safe, "Good form" };
    if (frameHeight < 1.0f) frameHeight = 480.0f;  // safety fallback

    switch (exercise) {
        case ExerciseType::Squat:          return assessSquat(keypoints, frameHeight);
        case ExerciseType::Deadlift:       return assessDeadlift(keypoints, frameHeight);
        case ExerciseType::BicepCurl:      return assessBicepCurl(keypoints, frameHeight);
        case ExerciseType::ShoulderPress:  return assessShoulderPress(keypoints, frameHeight);
        case ExerciseType::PushUp:         return assessPushUp(keypoints, frameHeight);
        case ExerciseType::Lunge:          return assessLunge(keypoints, frameHeight);
        case ExerciseType::LateralRaise:   return assessLateralRaise(keypoints, frameHeight);
        case ExerciseType::Plank:          return assessPlank(keypoints, frameHeight);
        case ExerciseType::TricepPushdown: return assessTricepPushdown(keypoints, frameHeight);
        case ExerciseType::PullUp:         return assessPullUp(keypoints, frameHeight);
        case ExerciseType::HipThrust:      return assessHipThrust(keypoints, frameHeight);
        case ExerciseType::LegExtension:   return assessLegExtension(keypoints, frameHeight);
        default:                           return { RiskLevel::Safe, "Good form" };
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  SQUAT — check knee valgus + forward lean
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessSquat(const std::vector<Keypoint>& kp, float fH)
{
    if (kp[13].confidence > MIN_CONF && kp[14].confidence > MIN_CONF &&
        kp[11].confidence > MIN_CONF && kp[12].confidence > MIN_CONF)
    {
        float kneeWidth = BiomechanicsMath::horizontalDistance(kp[13].pt, kp[14].pt);
        float hipWidth  = BiomechanicsMath::horizontalDistance(kp[11].pt, kp[12].pt);

        if (hipWidth > fH * 0.01f && kneeWidth < hipWidth * 0.65f) {
            return { RiskLevel::Danger, "Knees caving in!" };
        }
    }

    cv::Point2f shoulder, hip;
    float sc, hc;
    if (getBestSide(kp, 5, 6, shoulder, sc) && getBestSide(kp, 11, 12, hip, hc)) {
        double lean = BiomechanicsMath::angleFromVertical(shoulder, hip);
        if (lean > 50.0) {
            return { RiskLevel::Danger, "Too much forward lean!" };
        }
        if (lean > 35.0) {
            return { RiskLevel::Warning, "Watch your forward lean" };
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  DEADLIFT — rounded back + bar drift
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessDeadlift(const std::vector<Keypoint>& kp, float fH)
{
    cv::Point2f shoulder, hip, knee;
    float sc, hc, kc;
    if (getBestSide(kp, 5, 6, shoulder, sc) &&
        getBestSide(kp, 11, 12, hip, hc) &&
        getBestSide(kp, 13, 14, knee, kc))
    {
        double backAngle = BiomechanicsMath::calculateAngle(shoulder, hip, knee);
        if (backAngle < 120.0) {
            return { RiskLevel::Danger, "Keep back straight!" };
        }
        if (backAngle < 145.0) {
            return { RiskLevel::Warning, "Watch your back angle" };
        }
    }

    // Bar drift: normalized by torso height
    cv::Point2f wrist, ankle;
    float wc, ac;
    if (getBestSide(kp, 9, 10, wrist, wc) && getBestSide(kp, 15, 16, ankle, ac)) {
        float drift = std::abs(wrist.x - ankle.x);
        if (getBestSide(kp, 5, 6, shoulder, sc) && getBestSide(kp, 11, 12, hip, hc)) {
            float torsoH = BiomechanicsMath::verticalDistance(shoulder, hip);
            if (torsoH > fH * 0.04f && drift > torsoH * 0.5f) {
                return { RiskLevel::Warning, "Keep bar close to body" };
            }
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  BICEP CURL — body swing + elbow drift
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessBicepCurl(const std::vector<Keypoint>& kp, float fH)
{
    cv::Point2f shoulder, hip;
    float sc, hc;
    if (getBestSide(kp, 5, 6, shoulder, sc) && getBestSide(kp, 11, 12, hip, hc)) {
        double lean = BiomechanicsMath::angleFromVertical(shoulder, hip);
        if (lean > 20.0) {
            return { RiskLevel::Danger, "Stop swinging!" };
        }
        if (lean > 12.0) {
            return { RiskLevel::Warning, "Reduce body swing" };
        }
    }

    // Elbow drift — all thresholds relative to body dimensions
    cv::Point2f elbow;
    float ec;
    if (getBestSide(kp, 7, 8, elbow, ec) && getBestSide(kp, 11, 12, hip, hc) &&
        getBestSide(kp, 5, 6, shoulder, sc))
    {
        float torsoW = BiomechanicsMath::horizontalDistance(kp[5].pt, kp[6].pt);
        if (torsoW < fH * 0.02f) {
            // Side view: use vertical torso height as reference
            float torsoH = BiomechanicsMath::verticalDistance(shoulder, hip);
            if (torsoH > fH * 0.04f) {
                float drift = std::abs(elbow.x - hip.x);
                if (drift > torsoH * 0.3f) {
                    return { RiskLevel::Warning, "Keep elbows pinned" };
                }
            }
        } else {
            float drift = std::abs(elbow.x - hip.x);
            if (drift > torsoW * 0.5f) {
                return { RiskLevel::Warning, "Keep elbows pinned" };
            }
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  SHOULDER PRESS — excessive back arch
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessShoulderPress(const std::vector<Keypoint>& kp, float fH)
{
    (void)fH;  // angle-based only
    cv::Point2f shoulder, hip;
    float sc, hc;
    if (getBestSide(kp, 5, 6, shoulder, sc) && getBestSide(kp, 11, 12, hip, hc)) {
        double lean = BiomechanicsMath::angleFromVertical(shoulder, hip);
        if (lean > 25.0) {
            return { RiskLevel::Danger, "Don't arch back!" };
        }
        if (lean > 15.0) {
            return { RiskLevel::Warning, "Watch your back arch" };
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  PUSH-UP — hip sag / pike
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessPushUp(const std::vector<Keypoint>& kp, float fH)
{
    (void)fH;  // angle-based only
    cv::Point2f shoulder, hip, ankle;
    float sc, hc, ac;
    if (getBestSide(kp, 5, 6, shoulder, sc) &&
        getBestSide(kp, 11, 12, hip, hc) &&
        getBestSide(kp, 15, 16, ankle, ac))
    {
        double bodyAngle = BiomechanicsMath::calculateAngle(shoulder, hip, ankle);

        if (bodyAngle < 140.0) {
            return { RiskLevel::Danger, "Keep hips up!" };
        }
        if (bodyAngle < 155.0) {
            return { RiskLevel::Warning, "Straighten your body" };
        }

        bool hipAbove = BiomechanicsMath::pointAboveLine(hip, shoulder, ankle);
        if (hipAbove && bodyAngle < 155.0) {
            return { RiskLevel::Warning, "Lower your hips" };
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  LUNGE — knee past toes
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessLunge(const std::vector<Keypoint>& kp, float fH)
{
    // #10 fix: shinLen threshold now relative to frame height
    float minShinLen = fH * 0.02f;  // was hardcoded 10.0f

    // Left side
    if (kp[13].confidence > MIN_CONF && kp[15].confidence > MIN_CONF) {
        float kneeX  = kp[13].pt.x;
        float ankleX = kp[15].pt.x;
        float kneeY  = kp[13].pt.y;
        float ankleY = kp[15].pt.y;

        if (kp[11].confidence > MIN_CONF && kneeY > kp[11].pt.y) {
            float overshoot = std::abs(kneeX - ankleX);
            float shinLen = std::sqrt((kneeX - ankleX) * (kneeX - ankleX) +
                                       (kneeY - ankleY) * (kneeY - ankleY));
            if (shinLen > minShinLen && overshoot / shinLen > 0.5f) {
                return { RiskLevel::Warning, "Knee behind toes!" };
            }
        }
    }

    // Right side
    if (kp[14].confidence > MIN_CONF && kp[16].confidence > MIN_CONF) {
        float kneeX  = kp[14].pt.x;
        float ankleX = kp[16].pt.x;
        float kneeY  = kp[14].pt.y;
        float ankleY = kp[16].pt.y;

        if (kp[12].confidence > MIN_CONF && kneeY > kp[12].pt.y) {
            float overshoot = std::abs(kneeX - ankleX);
            float shinLen = std::sqrt((kneeX - ankleX) * (kneeX - ankleX) +
                                       (kneeY - ankleY) * (kneeY - ankleY));
            if (shinLen > minShinLen && overshoot / shinLen > 0.5f) {
                return { RiskLevel::Warning, "Knee behind toes!" };
            }
        }
    }

    // Forward lean (angle-based, no pixel dependency)
    cv::Point2f shoulder, hip;
    float sc, hc;
    if (getBestSide(kp, 5, 6, shoulder, sc) && getBestSide(kp, 11, 12, hip, hc)) {
        double lean = BiomechanicsMath::angleFromVertical(shoulder, hip);
        if (lean > 40.0) {
            return { RiskLevel::Warning, "Keep torso upright" };
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  LATERAL RAISE — shrugging (pixel distance → relative to frame height)
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessLateralRaise(const std::vector<Keypoint>& kp, float fH)
{
    // #10 fix: was hardcoded 15.0f, now ~3.1% of frame height
    float shrugThreshold = fH * 0.031f;

    // Left side
    if (kp[5].confidence > MIN_CONF && kp[3].confidence > MIN_CONF) {
        float dist = kp[5].pt.y - kp[3].pt.y;
        if (dist < shrugThreshold) {
            return { RiskLevel::Warning, "Don't shrug!" };
        }
    }
    // Right side
    if (kp[6].confidence > MIN_CONF && kp[4].confidence > MIN_CONF) {
        float dist = kp[6].pt.y - kp[4].pt.y;
        if (dist < shrugThreshold) {
            return { RiskLevel::Warning, "Don't shrug!" };
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  PLANK — hip sag or pike (angle-based)
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessPlank(const std::vector<Keypoint>& kp, float fH)
{
    (void)fH;  // angle-based only
    cv::Point2f shoulder, hip, ankle;
    float sc, hc, ac;
    if (getBestSide(kp, 5, 6, shoulder, sc) &&
        getBestSide(kp, 11, 12, hip, hc) &&
        getBestSide(kp, 15, 16, ankle, ac))
    {
        double bodyAngle = BiomechanicsMath::calculateAngle(shoulder, hip, ankle);

        if (bodyAngle < 145.0) {
            return { RiskLevel::Danger, "Hips sagging! Lift up!" };
        }
        if (bodyAngle < 160.0) {
            return { RiskLevel::Warning, "Keep hips level" };
        }

        bool hipAbove = BiomechanicsMath::pointAboveLine(hip, shoulder, ankle);
        if (hipAbove && bodyAngle < 160.0) {
            return { RiskLevel::Warning, "Lower your hips" };
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  TRICEP PUSHDOWN — elbow flare + body lean
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessTricepPushdown(const std::vector<Keypoint>& kp, float fH)
{
    cv::Point2f shoulder, hip;
    float sc, hc;
    if (getBestSide(kp, 5, 6, shoulder, sc) && getBestSide(kp, 11, 12, hip, hc)) {
        double lean = BiomechanicsMath::angleFromVertical(shoulder, hip);
        if (lean > 25.0) {
            return { RiskLevel::Danger, "Stand upright!" };
        }
        if (lean > 15.0) {
            return { RiskLevel::Warning, "Reduce forward lean" };
        }
    }

    // Elbow flare — torsoH threshold relative
    cv::Point2f elbow;
    float ec;
    if (getBestSide(kp, 7, 8, elbow, ec) && getBestSide(kp, 11, 12, hip, hc) &&
        getBestSide(kp, 5, 6, shoulder, sc))
    {
        float torsoH = BiomechanicsMath::verticalDistance(shoulder, hip);
        if (torsoH > fH * 0.04f) {
            float drift = std::abs(elbow.x - hip.x);
            if (drift > torsoH * 0.35f) {
                return { RiskLevel::Warning, "Keep elbows tucked" };
            }
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  PULL-UP — shoulder shrug + kipping swing
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessPullUp(const std::vector<Keypoint>& kp, float fH)
{
    // Kipping swing (angle-based)
    cv::Point2f shoulder, hip;
    float sc, hc;
    if (getBestSide(kp, 5, 6, shoulder, sc) && getBestSide(kp, 11, 12, hip, hc)) {
        double lean = BiomechanicsMath::angleFromVertical(shoulder, hip);
        if (lean > 35.0) {
            return { RiskLevel::Warning, "Reduce kipping swing" };
        }
    }

    // Shoulder shrug — #10 fix: relative threshold
    float shrugThreshold = fH * 0.021f;  // was hardcoded 10.0f

    if (kp[5].confidence > MIN_CONF && kp[3].confidence > MIN_CONF) {
        float dist = kp[5].pt.y - kp[3].pt.y;
        if (dist < shrugThreshold) {
            return { RiskLevel::Warning, "Depress your shoulders" };
        }
    }
    if (kp[6].confidence > MIN_CONF && kp[4].confidence > MIN_CONF) {
        float dist = kp[6].pt.y - kp[4].pt.y;
        if (dist < shrugThreshold) {
            return { RiskLevel::Warning, "Depress your shoulders" };
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  HIP THRUST — lower back overextension + knee cave
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessHipThrust(const std::vector<Keypoint>& kp, float fH)
{
    cv::Point2f shoulder, hip, knee;
    float sc, hc, kc;
    if (getBestSide(kp, 5, 6, shoulder, sc) &&
        getBestSide(kp, 11, 12, hip, hc) &&
        getBestSide(kp, 13, 14, knee, kc))
    {
        double hipAngle = BiomechanicsMath::calculateAngle(shoulder, hip, knee);
        if (hipAngle > 195.0) {
            return { RiskLevel::Danger, "Don't overextend lower back!" };
        }
        if (hipAngle > 180.0) {
            return { RiskLevel::Warning, "Slight overextension" };
        }
    }

    // Knee cave — ratio-based
    if (kp[13].confidence > MIN_CONF && kp[14].confidence > MIN_CONF &&
        kp[11].confidence > MIN_CONF && kp[12].confidence > MIN_CONF)
    {
        float kneeWidth = BiomechanicsMath::horizontalDistance(kp[13].pt, kp[14].pt);
        float hipWidth  = BiomechanicsMath::horizontalDistance(kp[11].pt, kp[12].pt);
        if (hipWidth > fH * 0.01f && kneeWidth < hipWidth * 0.6f) {
            return { RiskLevel::Warning, "Push knees out!" };
        }
    }

    return { RiskLevel::Safe, "Good form" };
}

// ═════════════════════════════════════════════════════════════════════════════
//  LEG EXTENSION — knee hyperextension + torso lean
// ═════════════════════════════════════════════════════════════════════════════
InjuryRisk InjuryRiskAssessor::assessLegExtension(const std::vector<Keypoint>& kp, float fH)
{
    (void)fH;  // angle-based only
    cv::Point2f hip, knee, ankle;
    float hc, kc, ac;
    if (getBestSide(kp, 11, 12, hip, hc) &&
        getBestSide(kp, 13, 14, knee, kc) &&
        getBestSide(kp, 15, 16, ankle, ac))
    {
        double kneeAngle = BiomechanicsMath::calculateAngle(hip, knee, ankle);
        if (kneeAngle > 185.0) {
            return { RiskLevel::Danger, "Avoid hyperextending knee!" };
        }
        if (kneeAngle > 175.0) {
            return { RiskLevel::Warning, "Control the lockout" };
        }
    }

    // Torso lean (angle-based)
    cv::Point2f shoulder, hipPt;
    float sc, hPc;
    if (getBestSide(kp, 5, 6, shoulder, sc) && getBestSide(kp, 11, 12, hipPt, hPc)) {
        double lean = BiomechanicsMath::angleFromVertical(shoulder, hipPt);
        if (lean > 40.0) {
            return { RiskLevel::Warning, "Sit upright" };
        }
    }

    return { RiskLevel::Safe, "Good form" };
}
