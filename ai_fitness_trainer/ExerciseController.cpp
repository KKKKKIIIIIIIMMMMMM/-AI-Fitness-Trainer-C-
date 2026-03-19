#include "ExerciseController.h"

// ═════════════════════════════════════════════════════════════════════════════
//  Static config table — ALL thresholds in one place (#4 magic numbers fix)
//
//  { upThreshold, downThreshold, upIsStart,
//    fullROM, partialROM,
//    leftA, leftB, leftC, rightA, rightB, rightC,
//    feedbackUp, feedbackDown }
// ═════════════════════════════════════════════════════════════════════════════
const ExerciseConfig ExerciseController::CONFIGS[23] = {
    // [0] BicepCurl — shoulder-elbow-wrist, start high → curl low
    { 160.0, 30.0, true,  100.0, 70.0,  5,7,9,  6,8,10,  "Curl Up!",      "Good Job!" },
    // [1] Squat — hip-knee-ankle, start high → squat low
    { 160.0, 90.0, true,  60.0,  40.0,  11,13,15, 12,14,16, "Go Lower!",     "Good Job!" },
    // [2] PushUp — shoulder-elbow-wrist, start high → push low
    { 160.0, 90.0, true,  60.0,  40.0,  5,7,9,  6,8,10,  "Go Lower!",     "Good Job!" },
    // [3] Lunge — hip-knee-ankle, start high → lunge low
    { 150.0, 100.0, true, 45.0,  25.0,  11,13,15, 12,14,16, "Lunge Down!",   "Good Job!" },
    // [4] ShoulderPress — shoulder-elbow-wrist, start LOW → press HIGH
    { 160.0, 100.0, false, 55.0, 35.0,  5,7,9,  6,8,10,  "Good Job!",     "Press Up!" },
    // [5] Deadlift — shoulder-hip-knee, start LOW → lift HIGH
    { 160.0, 100.0, false, 55.0, 35.0,  5,11,13, 6,12,14, "Good Job!",     "Lift Up!" },
    // [6] LateralRaise — hip-shoulder-wrist, start LOW → raise HIGH
    { 70.0,  30.0,  false, 35.0, 20.0,  11,5,9,  12,6,10, "Good Job!",     "Raise Arms!" },
    // [7] Plank — placeholder (not used by processGenericExercise)
    { 170.0, 150.0, true,  0.0,  0.0,   5,11,15, 6,12,16, "Perfect Form!", "Hold Steady!" },
    // [8] TricepPushdown — shoulder-elbow-wrist, start LOW → extend HIGH
    { 155.0, 80.0,  false, 80.0, 50.0,  5,7,9,  6,8,10,  "Good Job!",     "Push Down!" },
    // [9] PullUp — shoulder-elbow-wrist, start HIGH (155) → pull LOW (60)
    { 155.0, 60.0,  true,  100.0, 70.0, 5,7,9,  6,8,10,  "Pull Up!",      "Good Job!" },
    // [10] HipThrust — shoulder-hip-knee, start LOW
    { 160.0, 110.0, false, 40.0, 20.0,  5,11,13, 6,12,14, "Good Job!",     "Drop Hips!" },
    // [11] LegExtension — hip-knee-ankle, start LOW angle -> extend HIGH
    { 150.0, 100.0, false, 40.0, 20.0,  11,13,15, 12,14,16, "Good Job!",    "Lower Legs!" },
    
    // --- 11 NEW EXERCISES ---
    // [12] BenchPress — shoulder-elbow-wrist, start HIGH (straight = 160) → lower (90)
    { 160.0, 90.0, true,  60.0, 40.0,  5,7,9,  6,8,10,  "Lower Weight!", "Good Job!" },
    // [13] ChestFly — hip-shoulder-wrist, start HIGH (~90) → open wide (~160+)
    { 150.0, 100.0, false, 40.0, 20.0, 11,5,9, 12,6,10, "Good Job!",     "Open Arms!" },
    // [14] DeclineBenchPress
    { 160.0, 90.0, true,  60.0, 40.0,  5,7,9,  6,8,10,  "Lower Weight!", "Good Job!" },
    // [15] HammerCurl
    { 160.0, 45.0, true,  90.0, 60.0,  5,7,9,  6,8,10,  "Curl Up!",      "Good Job!" },
    // [16] InclineBenchPress
    { 160.0, 90.0, true,  60.0, 40.0,  5,7,9,  6,8,10,  "Lower Weight!", "Good Job!" },
    // [17] LatPulldown — arms straight up (~160) -> pull down (~60)
    { 155.0, 70.0, true,  80.0, 50.0,  5,7,9,  6,8,10,  "Pull Down!",    "Good Job!" },
    // [18] LegRaises — shoulder-hip-ankle. Flat = 180. Raise = 90.
    { 160.0, 100.0, true, 50.0, 30.0,  5,11,15, 6,12,16, "Raise Legs!",   "Good Job!" },
    // [19] RomanianDeadlift — shoulder-hip-knee. Stand = 180. Hinge = < 120.
    { 160.0, 110.0, true, 40.0, 20.0,  5,11,13, 6,12,14, "Hinge Hips!",   "Good Job!" },
    // [20] RussianTwist — (proxy track elbow or shoulder-hip)
    { 140.0, 90.0, true,  40.0, 20.0,  5,11,13, 6,12,14, "Twist!",        "Good Job!" },
    // [21] TBarRow — shoulder-elbow-wrist. Arms straight (160) -> pull to chest (80).
    { 160.0, 90.0, true,  60.0, 40.0,  5,7,9,  6,8,10,  "Row Up!",       "Good Job!" },
    // [22] TricepDips — shoulder-elbow-wrist. Arms straight (160) -> dip (90)
    { 160.0, 90.0, true,  60.0, 40.0,  5,7,9,  6,8,10,  "Dip Down!",     "Good Job!" }
};

const ExerciseConfig& ExerciseController::getConfig(ExerciseType type) {
    int idx = static_cast<int>(type);
    if (idx < 0 || idx >= 23) idx = 0;
    return CONFIGS[idx];
}

ExerciseController::ExerciseController()
    : currentExercise(ExerciseType::BicepCurl), currentState(ExerciseState::Up),
      repCount(0), lastAngle(0.0), smoothedAngle(0.0),
      lastP1(0, 0), lastJointPos(0, 0), lastP3(0, 0),
      feedback("Ready"),
      maxAngleInRep(0.0), minAngleInRep(999.0),
      lastRepQuality(""), fullReps(0), partialReps(0), poorReps(0),
      holdAccumulator(0.0), holdTimeSec(0.0),
      plankTimeInit(false),
      lastRepTime(std::chrono::steady_clock::now()) {}

ExerciseController::~ExerciseController() {}

void ExerciseController::setExerciseType(ExerciseType type) {
    currentExercise = type;
    reset();
}

void ExerciseController::update(const std::vector<Keypoint>& keypoints) {
    if (keypoints.size() < 17) return;

    if (currentExercise == ExerciseType::Plank) {
        processPlank(keypoints);
    } else {
        processGenericExercise(keypoints, getConfig(currentExercise));
    }
}

int ExerciseController::getRepCount() const { return repCount; }
std::string ExerciseController::getFeedback() const { return feedback; }
double ExerciseController::getCurrentAngle() const { return lastAngle; }
cv::Point2f ExerciseController::getLastJointPosition() const { return lastJointPos; }

void ExerciseController::getActiveKeypoints(cv::Point2f& p1, cv::Point2f& p2, cv::Point2f& p3) const {
    p1 = lastP1;
    p2 = lastJointPos;
    p3 = lastP3;
}

std::string ExerciseController::getLastRepQuality() const { return lastRepQuality; }
int ExerciseController::getFullReps() const { return fullReps; }
int ExerciseController::getPartialReps() const { return partialReps; }
int ExerciseController::getPoorReps() const { return poorReps; }

bool ExerciseController::isHoldExercise() const {
    return currentExercise == ExerciseType::Plank;
}

double ExerciseController::getHoldTimeSec() const {
    return holdTimeSec;
}

void ExerciseController::reset() {
    repCount = 0;
    lastAngle = 0.0;
    smoothedAngle = 0.0;
    lastP1 = cv::Point2f(0, 0);
    lastJointPos = cv::Point2f(0, 0);
    lastP3 = cv::Point2f(0, 0);
    currentState = ExerciseState::Up;
    feedback = "Ready";
    maxAngleInRep = 0.0;
    minAngleInRep = 999.0;
    lastRepQuality = "";
    fullReps = 0;
    partialReps = 0;
    poorReps = 0;
    holdAccumulator = 0.0;
    holdTimeSec = 0.0;
    plankTimeInit = false;
    lastRepTime = std::chrono::steady_clock::now();
    lastChosenSide = 0;
    sideConsistencyCount = 0;
}

// =============================================
// Classifier → ExerciseType mapping
// =============================================
ExerciseType ExerciseController::mapClassifierToExercise(int classIndex) {
    switch (classIndex) {
        case 0:  return ExerciseType::BicepCurl;         // barbell_biceps_curl
        case 1:  return ExerciseType::BenchPress;        // bench_press
        case 2:  return ExerciseType::ChestFly;          // chest_fly_machine
        case 3:  return ExerciseType::Deadlift;          // deadlift
        case 4:  return ExerciseType::DeclineBenchPress; // decline_bench_press
        case 5:  return ExerciseType::HammerCurl;        // hammer_curl
        case 6:  return ExerciseType::HipThrust;         // hip_thrust
        case 7:  return ExerciseType::InclineBenchPress; // incline_bench_press
        case 8:  return ExerciseType::LatPulldown;       // lat_pulldown
        case 9:  return ExerciseType::LateralRaise;      // lateral_raise
        case 10: return ExerciseType::LegExtension;      // leg_extension
        case 11: return ExerciseType::LegRaises;         // leg_raises
        case 12: return ExerciseType::Plank;             // plank
        case 13: return ExerciseType::PullUp;            // pull_up
        case 14: return ExerciseType::PushUp;            // push_up
        case 15: return ExerciseType::RomanianDeadlift;  // romanian_deadlift
        case 16: return ExerciseType::RussianTwist;      // russian_twist
        case 17: return ExerciseType::ShoulderPress;     // shoulder_press
        case 18: return ExerciseType::Squat;             // squat
        case 19: return ExerciseType::TBarRow;           // t_bar_row
        case 20: return ExerciseType::TricepDips;        // tricep_dips
        case 21: return ExerciseType::TricepPushdown;    // tricep_pushdown
        default: return ExerciseType::BicepCurl;
    }
}

const char* ExerciseController::getExerciseDisplayName(ExerciseType type) {
    switch (type) {
        case ExerciseType::BicepCurl:         return "Bicep Curl";
        case ExerciseType::Squat:             return "Squat";
        case ExerciseType::PushUp:            return "Push-Up";
        case ExerciseType::Lunge:             return "Lunge";
        case ExerciseType::ShoulderPress:     return "Shoulder Press";
        case ExerciseType::Deadlift:          return "Deadlift";
        case ExerciseType::LateralRaise:      return "Lateral Raise";
        case ExerciseType::Plank:             return "Plank";
        case ExerciseType::TricepPushdown:    return "Tricep Pushdown";
        case ExerciseType::PullUp:            return "Pull-Up";
        case ExerciseType::HipThrust:         return "Hip Thrust";
        case ExerciseType::LegExtension:      return "Leg Extension";
        case ExerciseType::BenchPress:        return "Bench Press";
        case ExerciseType::ChestFly:          return "Chest Fly";
        case ExerciseType::DeclineBenchPress: return "Decline Bench Press";
        case ExerciseType::HammerCurl:        return "Hammer Curl";
        case ExerciseType::InclineBenchPress: return "Incline Bench Press";
        case ExerciseType::LatPulldown:       return "Lat Pulldown";
        case ExerciseType::LegRaises:         return "Leg Raises";
        case ExerciseType::RomanianDeadlift:  return "Romanian Deadlift";
        case ExerciseType::RussianTwist:      return "Russian Twist";
        case ExerciseType::TBarRow:           return "T-Bar Row";
        case ExerciseType::TricepDips:        return "Tricep Dips";
        default:                              return "Unknown";
    }
}

// =============================================
// Shared helpers
// =============================================

bool ExerciseController::pickBestSideAndComputeAngle(
    const std::vector<Keypoint>& keypoints,
    int leftA, int leftB, int leftC,
    int rightA, int rightB, int rightC,
    double& outAngle)
{
    float leftConf  = (keypoints[leftA].confidence  + keypoints[leftB].confidence  + keypoints[leftC].confidence)  / 3.0f;
    float rightConf = (keypoints[rightA].confidence + keypoints[rightB].confidence + keypoints[rightC].confidence) / 3.0f;

    // ── Determine which side to use this frame ──────────────────────────
    int chosenSide = 0;

    // If side is locked, keep it unless confidence drops below threshold
    // #6 fix: SIDE_LOCK_FRAMES=3 (faster lock), SIDE_LOCK_DROP_CONF=0.5 (faster release)
    if (sideConsistencyCount >= SIDE_LOCK_FRAMES && lastChosenSide != 0) {
        float lockedConf = (lastChosenSide == 1) ? leftConf : rightConf;
        if (lockedConf > SIDE_LOCK_DROP_CONF) {
            chosenSide = lastChosenSide;
        }
    }

    if (chosenSide == 0) {
        if (leftConf >= rightConf && leftConf > 0.5f) {
            chosenSide = 1;
        } else if (rightConf > 0.5f) {
            chosenSide = 2;
        } else {
            lastChosenSide = 0;
            sideConsistencyCount = 0;
            return false;
        }
    }

    if (chosenSide == lastChosenSide) {
        sideConsistencyCount++;
    } else {
        lastChosenSide = chosenSide;
        sideConsistencyCount = 1;
    }

    int a, b, c;
    if (chosenSide == 1) {
        a = leftA;  b = leftB;  c = leftC;
    } else {
        a = rightA; b = rightB; c = rightC;
    }

    outAngle = BiomechanicsMath::calculateAngle(keypoints[a].pt, keypoints[b].pt, keypoints[c].pt);
    lastP1       = keypoints[a].pt;
    lastJointPos = keypoints[b].pt;
    lastP3       = keypoints[c].pt;
    return true;
}

void ExerciseController::applySmoothingAndTrackRange(double rawAngle) {
    if (smoothedAngle == 0.0) {
        smoothedAngle = rawAngle;
    } else {
        smoothedAngle = kSmoothingAlpha * rawAngle + (1.0 - kSmoothingAlpha) * smoothedAngle;
    }
    lastAngle = smoothedAngle;

    maxAngleInRep = std::max(maxAngleInRep, smoothedAngle);
    minAngleInRep = std::min(minAngleInRep, smoothedAngle);
}

// ── Rep quality classification — now config-driven (#4, #9) ─────────────
void ExerciseController::classifyAndCountRep(const ExerciseConfig& cfg) {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - lastRepTime).count();
    if (elapsed < kMinRepIntervalSec) return;
    lastRepTime = now;

    double range = maxAngleInRep - minAngleInRep;

    if (range >= cfg.fullROM) {
        lastRepQuality = "Full";
        fullReps++;
    } else if (range >= cfg.partialROM) {
        lastRepQuality = "Partial";
        partialReps++;
    } else {
        lastRepQuality = "Poor";
        poorReps++;
    }

    repCount++;
    feedback = "Good Job!";

    maxAngleInRep = smoothedAngle;
    minAngleInRep = smoothedAngle;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Generic exercise processor — replaces 11 near-identical methods (#9)
//
//  upIsStart=true:  state=Down when angle > upThreshold, count when angle < downThreshold
//  upIsStart=false: state=Down when angle < downThreshold, count when angle > upThreshold
// ═════════════════════════════════════════════════════════════════════════════
void ExerciseController::processGenericExercise(
    const std::vector<Keypoint>& keypoints, const ExerciseConfig& cfg)
{
    double angle;
    if (!pickBestSideAndComputeAngle(keypoints,
            cfg.leftA, cfg.leftB, cfg.leftC,
            cfg.rightA, cfg.rightB, cfg.rightC, angle)) {
        feedback = "Adjust Position";
        return;
    }
    applySmoothingAndTrackRange(angle);

    if (cfg.upIsStart) {
        // High angle = ready, go low to count (BicepCurl, Squat, PushUp, Lunge, PullUp)
        if (smoothedAngle > cfg.upThreshold) {
            currentState = ExerciseState::Down;
            feedback = cfg.feedbackUp;
        } else if (smoothedAngle < cfg.downThreshold && currentState == ExerciseState::Down) {
            currentState = ExerciseState::Up;
            classifyAndCountRep(cfg);
        }
    } else {
        // Low angle = ready, go high to count (ShoulderPress, Deadlift, LateralRaise, etc.)
        if (smoothedAngle < cfg.downThreshold) {
            currentState = ExerciseState::Down;
            feedback = cfg.feedbackDown;
        } else if (smoothedAngle > cfg.upThreshold && currentState == ExerciseState::Down) {
            currentState = ExerciseState::Up;
            classifyAndCountRep(cfg);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Plank — hold exercise, NOT rep-based (#5 fix)
//  repCount = whole seconds held (displayed as SECONDS in UI)
//  holdTimeSec = precise elapsed time in good form
// ═════════════════════════════════════════════════════════════════════════════
void ExerciseController::processPlank(const std::vector<Keypoint>& keypoints) {
    double angle;
    if (!pickBestSideAndComputeAngle(keypoints, 5, 11, 15, 6, 12, 16, angle)) {
        feedback = "Adjust Position"; return;
    }
    applySmoothingAndTrackRange(angle);

    // ── Real elapsed time ──────────────────────────────────────────────
    auto now = std::chrono::steady_clock::now();
    double dt = 0.033;
    if (plankTimeInit) {
        dt = std::chrono::duration<double>(now - lastPlankTime).count();
        if (dt > 0.5) dt = 0.033;
    }
    lastPlankTime = now;
    plankTimeInit = true;

    // Good plank form: body nearly straight (150-180 degrees)
    if (smoothedAngle > 150) {
        holdTimeSec += dt;
        // repCount tracks whole seconds (UI shows "SECONDS" label)
        repCount = (int)holdTimeSec;

        if (smoothedAngle > 170) {
            feedback = "Perfect Form!";
            lastRepQuality = "Full";
        } else {
            feedback = "Hold Steady!";
            lastRepQuality = "Partial";
        }
    } else {
        feedback = "Straighten Body!";
        lastRepQuality = "Poor";
        // Don't reset holdTimeSec — total good-form time is preserved
    }
}
