#pragma once
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include "PoseEstimator.h"

// Which way the person is facing the camera
enum class ViewAngle {
    Unknown,    // can't determine (low keypoint confidence)
    Front,      // facing camera directly (both shoulders visible)
    LeftSide,   // person's left side toward camera
    RightSide   // person's right side toward camera
};

// Pose-based exercise hint — computed purely from keypoint geometry.
// exerciseTypeInt matches ExerciseType enum (0=BicepCurl … 11=LegExtension), -1 = unknown.
struct PoseHint {
    int   exerciseTypeInt = -1;
    float conf            = 0.0f;
};

class BiomechanicsMath {
public:
    // Original 3-point angle (p1-p2-p3), returns 0-180 degrees
    static double calculateAngle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);

    // Angle of line (top→bottom) from vertical (0° = straight up)
    static double angleFromVertical(cv::Point2f top, cv::Point2f bottom);

    // Absolute horizontal distance |a.x - b.x|
    static float horizontalDistance(cv::Point2f a, cv::Point2f b);

    // Absolute vertical distance |a.y - b.y|
    static float verticalDistance(cv::Point2f a, cv::Point2f b);

    // Check if point p is above the line connecting lineA→lineB
    // (in screen coords where Y increases downward, "above" means smaller Y)
    static bool pointAboveLine(cv::Point2f p, cv::Point2f lineA, cv::Point2f lineB);

    // Detect which way the person faces the camera using shoulder/hip keypoints.
    // frameWidth: pixel width of the camera frame (used to normalise shoulder span).
    static ViewAngle detectViewAngle(const std::vector<Keypoint>& kps, float frameWidth);

    // Return the recommended view for a given exercise (for on-screen tip)
    // Returns "" if any view is acceptable
    static const char* recommendedViewLabel(int exerciseType);

    // Classify likely exercise from pose keypoints alone (geometry / angle rules).
    // frameH/frameW: camera frame dimensions for normalisation.
    // Returns PoseHint with exerciseTypeInt=-1 when too uncertain.
    static PoseHint poseExerciseHint(const std::vector<Keypoint>& kps,
                                     float frameH, float frameW);
};
