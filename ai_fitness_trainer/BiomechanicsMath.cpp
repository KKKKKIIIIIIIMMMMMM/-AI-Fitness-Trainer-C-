#include "BiomechanicsMath.h"
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
//  3-point angle (p1-vertex-p3) via dot product. Returns 0-180°.
// ─────────────────────────────────────────────────────────────────────────────
double BiomechanicsMath::calculateAngle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {
    cv::Point2f v1 = p1 - p2;
    cv::Point2f v2 = p3 - p2;

    double dotProduct = v1.x * v2.x + v1.y * v2.y;
    double mag1 = std::sqrt(v1.x * v1.x + v1.y * v1.y);
    double mag2 = std::sqrt(v2.x * v2.x + v2.y * v2.y);

    if (mag1 == 0 || mag2 == 0) return 0;

    double cosTheta = dotProduct / (mag1 * mag2);
    if (cosTheta > 1.0) cosTheta = 1.0;
    if (cosTheta < -1.0) cosTheta = -1.0;

    double angleRad = std::acos(cosTheta);
    return angleRad * 180.0 / CV_PI;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Angle of line from vertical. 0° = perfectly upright (top directly above bottom).
//  Uses screen coordinates (Y increases downward).
// ─────────────────────────────────────────────────────────────────────────────
double BiomechanicsMath::angleFromVertical(cv::Point2f top, cv::Point2f bottom) {
    float dx = top.x - bottom.x;
    float dy = bottom.y - top.y;  // flip Y for screen coords (bottom.y > top.y normally)

    if (dy <= 0) return 90.0;  // top is below bottom → completely horizontal

    double angleRad = std::atan2(std::abs(dx), dy);
    return angleRad * 180.0 / CV_PI;
}

// ─────────────────────────────────────────────────────────────────────────────
float BiomechanicsMath::horizontalDistance(cv::Point2f a, cv::Point2f b) {
    return std::abs(a.x - b.x);
}

// ─────────────────────────────────────────────────────────────────────────────
float BiomechanicsMath::verticalDistance(cv::Point2f a, cv::Point2f b) {
    return std::abs(a.y - b.y);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Check if point p is above the line connecting lineA → lineB.
//  In screen coords (Y down), "above" means p.y is less than the line's Y at p.x.
// ─────────────────────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────────────────────
//  detectViewAngle
//  Uses shoulder keypoints to determine camera-facing direction.
//
//  Logic:
//   1. If both shoulders confident + wide span  → Front
//   2. If left shoulder much more confident     → LeftSide  (left side to camera)
//   3. If right shoulder much more confident    → RightSide (right side to camera)
// ─────────────────────────────────────────────────────────────────────────────
ViewAngle BiomechanicsMath::detectViewAngle(const std::vector<Keypoint>& kps,
                                            float frameWidth) {
    if ((int)kps.size() < 13 || frameWidth < 1.0f)
        return ViewAngle::Unknown;

    const Keypoint& lShoulder = kps[5];
    const Keypoint& rShoulder = kps[6];
    const Keypoint& lHip      = kps[11];
    const Keypoint& rHip      = kps[12];

    // Need at least one shoulder with usable confidence
    if (lShoulder.confidence < 0.2f && rShoulder.confidence < 0.2f)
        return ViewAngle::Unknown;

    // Shoulder horizontal span normalised by frame width
    float span = 0.0f;
    if (lShoulder.confidence > 0.2f && rShoulder.confidence > 0.2f)
        span = std::abs(lShoulder.pt.x - rShoulder.pt.x) / frameWidth;

    // Also check hip span for confirmation
    float hipSpan = 0.0f;
    if (lHip.confidence > 0.2f && rHip.confidence > 0.2f)
        hipSpan = std::abs(lHip.pt.x - rHip.pt.x) / frameWidth;

    // Front view: both shoulders clearly visible AND significant width
    bool bothShouldersSeen = lShoulder.confidence > 0.35f &&
                             rShoulder.confidence > 0.35f;
    bool wideSpan          = span > 0.12f || hipSpan > 0.08f;

    if (bothShouldersSeen && wideSpan)
        return ViewAngle::Front;

    // Side view: determine which side is showing by confidence asymmetry
    float confDiff = lShoulder.confidence - rShoulder.confidence;

    if (confDiff > 0.15f)
        return ViewAngle::LeftSide;   // left shoulder much clearer → left side to camera
    if (confDiff < -0.15f)
        return ViewAngle::RightSide;  // right shoulder much clearer → right side to camera

    // Both roughly equal but narrow span → front (person may be far from camera)
    if (bothShouldersSeen)
        return ViewAngle::Front;

    return ViewAngle::Unknown;
}

// ─────────────────────────────────────────────────────────────────────────────
//  recommendedViewLabel
//  Returns a short hint string for each exercise, or "" if any view is fine.
//  exerciseType matches ExerciseType enum values (0-11).
// ─────────────────────────────────────────────────────────────────────────────
const char* BiomechanicsMath::recommendedViewLabel(int exerciseType) {
    switch (exerciseType) {
        case 2:  // PushUp
        case 5:  // Deadlift
        case 7:  // Plank
        case 10: // HipThrust
        case 12: // BenchPress
        case 14: // DeclineBenchPress
        case 16: // InclineBenchPress
        case 18: // LegRaises
        case 19: // RomanianDeadlift
        case 20: // RussianTwist
            return "Side";
        case 4:  // ShoulderPress
        case 6:  // LateralRaise
        case 9:  // PullUp
        case 13: // ChestFly
        case 15: // HammerCurl
        case 17: // LatPulldown
        case 21: // TBarRow
        case 22: // TricepDips
            return "Front";
        default:
            return "";  // any view OK
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  poseExerciseHint
//  Scores all 12 ExerciseType values using pure keypoint geometry (angles +
//  relative positions).  Returns the winner if it's confident enough and
//  clearly ahead of the second-best candidate.
//
//  Image-coordinate convention (Y increases downward):
//    "above" in image  →  smaller Y  →  (refY − kpY) > 0
//    "below" in image  →  larger Y   →  (kpY − refY) > 0
// ─────────────────────────────────────────────────────────────────────────────
PoseHint BiomechanicsMath::poseExerciseHint(const std::vector<Keypoint>& kps,
                                            float frameH, float frameW)
{
    PoseHint result;
    if ((int)kps.size() < 17 || frameH < 1.0f || frameW < 1.0f)
        return result;

    // ── helpers ──────────────────────────────────────────────────────────────
    auto vis = [&](int i) { return kps[i].confidence > 0.25f; };
    auto pt  = [&](int i) -> cv::Point2f { return kps[i].pt; };
    auto cl  = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };

    // 3-point angle — returns a neutral 150° if any keypoint is occluded
    auto ang3 = [&](int a, int b, int c) -> float {
        if (!vis(a) || !vis(b) || !vis(c)) return 150.0f;
        return (float)calculateAngle(pt(a), pt(b), pt(c));
    };

    // ── key angles ───────────────────────────────────────────────────────────
    // COCO: 5=lShoulder 6=rShoulder 7=lElbow 8=rElbow 9=lWrist 10=rWrist
    //       11=lHip 12=rHip 13=lKnee 14=rKnee 15=lAnkle 16=rAnkle
    float lElbow = ang3(5,  7,  9);   // shoulder-elbow-wrist (left)
    float rElbow = ang3(6,  8, 10);   // shoulder-elbow-wrist (right)
    float lKnee  = ang3(11, 13, 15);  // hip-knee-ankle       (left)
    float rKnee  = ang3(12, 14, 16);  // hip-knee-ankle       (right)
    float lHip   = ang3(5,  11, 13);  // shoulder-hip-knee    (left)
    float rHip   = ang3(6,  12, 14);  // shoulder-hip-knee    (right)

    float avgElbow = (lElbow + rElbow) * 0.5f;
    float minElbow = std::min(lElbow, rElbow);
    float avgKnee  = (lKnee  + rKnee)  * 0.5f;
    float avgHip   = (lHip   + rHip)   * 0.5f;

    // ── torso orientation ────────────────────────────────────────────────────
    // torsoUp: 1.0 = fully upright, 0.0 = fully horizontal
    float torsoUp = 0.5f;
    bool  hasTorso = false;
    float dx = 0.0f, dy = 0.0f;

    // Try averaging if both sides visible
    if (vis(5) && vis(6) && vis(11) && vis(12)) {
        float sx = (kps[5].pt.x + kps[6].pt.x) * 0.5f;
        float sy = (kps[5].pt.y + kps[6].pt.y) * 0.5f;
        float hx = (kps[11].pt.x + kps[12].pt.x) * 0.5f;
        float hy = (kps[11].pt.y + kps[12].pt.y) * 0.5f;
        dx = hx - sx; dy = hy - sy;
        hasTorso = true;
    }
    // Fallback to right side
    else if (vis(6) && vis(12)) {
        dx = kps[12].pt.x - kps[6].pt.x;
        dy = kps[12].pt.y - kps[6].pt.y;
        hasTorso = true;
    }
    // Fallback to left side
    else if (vis(5) && vis(11)) {
        dx = kps[11].pt.x - kps[5].pt.x;
        dy = kps[11].pt.y - kps[5].pt.y;
        hasTorso = true;
    }

    if (hasTorso) {
        float len = std::sqrt(dx*dx + dy*dy);
        if (len > 1.0f) torsoUp = cl(dy / len);
    }
    float torsoHoriz = 1.0f - torsoUp;

    // ── average Y positions (raw image pixels) ───────────────────────────────
    float shoulderY = 0.0f, elbowY = 0.0f, wristY = 0.0f;
    int   sc = 0, ec = 0, wc = 0;
    if (vis(5))  { shoulderY += kps[5].pt.y;  sc++; }
    if (vis(6))  { shoulderY += kps[6].pt.y;  sc++; }
    if (vis(7))  { elbowY   += kps[7].pt.y;  ec++; }
    if (vis(8))  { elbowY   += kps[8].pt.y;  ec++; }
    if (vis(9))  { wristY   += kps[9].pt.y;  wc++; }
    if (vis(10)) { wristY   += kps[10].pt.y; wc++; }
    if (sc > 0) shoulderY /= sc;
    if (ec > 0) elbowY    /= ec;
    if (wc > 0) wristY    /= wc;

    // Positive = keypoint is ABOVE the reference (smaller Y = higher in frame)
    float wristAboveShoulder = (shoulderY - wristY)  / frameH;
    float wristAboveElbow    = (elbowY    - wristY)  / frameH;
    float elbowBelowShoulder = (elbowY    - shoulderY) / frameH; // >0 = arm at side

    // Horizontal wrist spread (bilateral)
    float wristSpread = 0.0f;
    if (vis(9) && vis(10))
        wristSpread = std::abs(kps[10].pt.x - kps[9].pt.x) / frameW;

    // ── score each ExerciseType (indices 0-22) ───────────────────────────────
    float s[23] = {};

    // 0 BicepCurl — upright + at least one elbow bent + wrist rising
    s[0] = torsoUp * 0.35f
         + cl((155.f - minElbow) / 100.f) * 0.40f
         + cl(wristAboveElbow * 6.f)      * 0.25f;

    // 1 Squat — upright + knees clearly bent
    s[1] = torsoUp * 0.40f
         + cl((155.f - avgKnee) / 90.f)  * 0.60f;

    // 2 PushUp — body horizontal + arms varying
    s[2] = torsoHoriz * 0.70f
         + cl((165.f - avgElbow) / 80.f) * 0.30f;

    // 3 Lunge — upright + left/right knee angles differ significantly
    {
        float diff = std::abs(lKnee - rKnee);
        s[3] = torsoUp * 0.25f
             + cl(diff / 55.f)             * 0.50f
             + cl((150.f - avgKnee) / 80.f)* 0.25f;
    }

    // 4 ShoulderPress — upright + wrists clearly above shoulders
    s[4] = torsoUp * 0.25f
         + cl(wristAboveShoulder * 7.f)   * 0.75f;

    // 5 Deadlift — hip hinge (shoulder-hip-knee angle well below 165°)
    s[5] = cl((165.f - avgHip) / 80.f)   * 0.65f
         + torsoUp * 0.35f;

    // 6 LateralRaise — upright + wrists near shoulder height + arms spread
    {
        float nearShoulder = cl(1.f - std::abs(wristAboveShoulder) * 9.f);
        s[6] = torsoUp * 0.30f
             + nearShoulder * 0.40f
             + cl(wristSpread * 2.5f) * 0.30f;
    }

    // 7 Plank — body horizontal + legs relatively straight
    s[7] = torsoHoriz * 0.75f
         + cl((avgKnee - 150.f) / 30.f)  * 0.25f;

    // 8 TricepPushdown — upright + elbow at side + wrist below/at elbow
    s[8] = torsoUp * 0.30f
         + cl(elbowBelowShoulder * 6.f)        * 0.40f
         + cl(-wristAboveElbow * 6.f + 0.3f)   * 0.30f;

    // 9 PullUp — upright + wrists well above shoulders (at or above head)
    s[9] = torsoUp * 0.20f
         + cl(wristAboveShoulder * 5.f + 0.15f)* 0.80f;

    // 10 HipThrust — leaning back + knees bent
    s[10] = cl((0.75f - torsoUp) * 3.f)  * 0.45f
          + cl((155.f - avgKnee) / 70.f) * 0.55f;

    // 11 LegExtension — upright + knee in mid-range (extending from seat)
    s[11] = torsoUp * 0.40f
          + cl((avgKnee - 70.f) / 110.f) * 0.35f
          + 0.10f;

    // 12 BenchPress — horizontal + elbows bend/extend + wrists above shoulders
    s[12] = torsoHoriz * 0.60f
          + cl((165.f - avgElbow) / 80.f) * 0.20f
          + cl(wristAboveShoulder * 4.f) * 0.20f;

    // 13 ChestFly — body horizontal/semi + arm spread + wrists above/level with shoulders
    s[13] = torsoHoriz * 0.50f
          + cl(wristSpread * 2.5f) * 0.50f;

    // 14 DeclineBenchPress — same logic footprint
    s[14] = s[12] * 0.95f; 

    // 15 HammerCurl — very similar to BicepCurl
    s[15] = s[0] * 0.95f;

    // 16 InclineBenchPress — semi-upright + elbows bend
    s[16] = torsoUp * 0.40f + torsoHoriz * 0.40f
          + cl((165.f - avgElbow) / 80.f) * 0.20f;

    // 17 LatPulldown — upright + wrists well above shoulders + elbows bend
    s[17] = torsoUp * 0.40f
          + cl((165.f - avgElbow) / 80.f) * 0.30f
          + cl(wristAboveShoulder * 5.f) * 0.30f;

    // 18 LegRaises — horizontal + hips bend (knee angle stays high)
    s[18] = torsoHoriz * 0.60f
          + cl((avgKnee - 150.f) / 30.f) * 0.20f
          + cl((165.f - avgHip) / 80.f) * 0.20f;

    // 19 RomanianDeadlift — upright + hips hinge + knee relatively straight
    s[19] = torsoUp * 0.30f
          + cl((165.f - avgHip) / 80.f) * 0.50f
          + cl((avgKnee - 130.f) / 50.f) * 0.20f;

    // 20 RussianTwist — semi-upright/horizontal + hips bent
    s[20] = torsoUp * 0.20f + torsoHoriz * 0.20f
          + cl((165.f - avgHip) / 70.f) * 0.60f;

    // 21 TBarRow — bent over (torso horiz or semi) + pulling arms back
    s[21] = torsoHoriz * 0.40f + torsoUp * 0.20f
          + cl(elbowBelowShoulder * 4.f) * 0.20f
          + cl((165.f - avgElbow) / 80.f) * 0.20f;

    // 22 TricepDips — upright + arms at sides pushing down
    s[22] = torsoUp * 0.50f
          + cl(elbowBelowShoulder * 6.f) * 0.30f
          + cl((165.f - avgElbow) / 80.f) * 0.20f;

    // ── find winner ───────────────────────────────────────────────────────────
    int   best = 0;
    float bval = s[0];
    for (int i = 1; i < 23; i++)
        if (s[i] > bval) { bval = s[i]; best = i; }

    float second = 0.0f;
    for (int i = 0; i < 23; i++)
        if (i != best && s[i] > second) second = s[i];

    // Accept only if confident enough AND clearly better than runner-up
    if (bval >= 0.42f && (bval - second) >= 0.08f) {
        result.exerciseTypeInt = best;
        result.conf            = bval;
    }
    return result;
}

bool BiomechanicsMath::pointAboveLine(cv::Point2f p, cv::Point2f lineA, cv::Point2f lineB) {
    if (std::abs(lineB.x - lineA.x) < 1e-6f) {
        // Near-vertical line: compare y to midpoint
        float midY = (lineA.y + lineB.y) / 2.0f;
        return p.y < midY;
    }

    // Interpolate line Y at p.x
    float t = (p.x - lineA.x) / (lineB.x - lineA.x);
    float lineY = lineA.y + t * (lineB.y - lineA.y);
    return p.y < lineY;
}
