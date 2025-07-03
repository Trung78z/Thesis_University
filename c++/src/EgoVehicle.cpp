#include <EgoVehicle.h>

// --- Function to get driving state ---
std::pair<std::string, int> getDrivingState(float distance, float frontSpeed, float egoSpeed) {
    if (distance < Config::adaptiveSpeed.cruise) {
        return {"EMERGENCY_BRAKE", 3};
    } else if (distance < Config::adaptiveSpeed.minFollowDistance) {
        return {"CLOSE_FOLLOW", 2};
    } else if (distance < Config::adaptiveSpeed.followDistance) {
        if (frontSpeed < egoSpeed - 15) {
            return {"SLOW_TRAFFIC", 2};
        } else {
            return {"NORMAL_FOLLOW", 1};
        }
    } else {
        return {"FREE_DRIVE", 0};
    }
}

// --- Function to calculate target speed ---
float calculateTargetSpeed(float distance, float frontSpeed, float egoSpeed,
                           const std::string& drivingState, int urgency) {
    if (drivingState == "EMERGENCY_BRAKE") {
        // Emergency: reduce to 70% of current speed immediately
        return std::max(Config::adjustment.min, egoSpeed * 0.7f);
    } else if (drivingState == "CLOSE_FOLLOW") {
        // Too close: match front vehicle speed with safety margin
        float safetyMargin = 0.9f;  // 90% of front vehicle speed
        return std::max(Config::adjustment.min, frontSpeed * safetyMargin);
    } else if (drivingState == "SLOW_TRAFFIC") {
        // Slow traffic ahead: gradually reduce speed
        float slowTrafficSpeedMargin = 1.1f;
        return std::max(Config::adjustment.min,
                        std::min(frontSpeed * slowTrafficSpeedMargin, egoSpeed * 0.95f));
    } else if (drivingState == "NORMAL_FOLLOW") {
        // Normal following: maintain similar speed to front vehicle
        if (std::abs(frontSpeed - egoSpeed) < 5) {
            return egoSpeed;  // Already at good speed
        } else {
            // Gradually converge to front vehicle speed
            return frontSpeed * 0.98f;
        }
    } else {  // FREE_DRIVE
        // No obstacles: can return to cruise speed
        return Config::adaptiveSpeed.cruise;
    }
}

void getActionAndColor(const std::string& drivingState, float speedChange, std::string& action,
                       cv::Scalar& color) {
    if (drivingState == "EMERGENCY_BRAKE") {
        action = "EMERGENCY BRAKE";
        color = cv::Scalar(0, 0, 255);  // Red
    } else if (drivingState == "CLOSE_FOLLOW") {
        action = "BRAKE/SLOW";
        color = cv::Scalar(0, 100, 255);  // Orange-Red
    } else if (drivingState == "SLOW_TRAFFIC") {
        action = "DECELERATE";
        color = cv::Scalar(0, 255, 255);  // Yellow
    } else if (drivingState == "NORMAL_FOLLOW") {
        if (std::abs(speedChange) < 1.0f) {
            action = "MAINTAIN";
            color = cv::Scalar(0, 255, 0);  // Green
        } else if (speedChange > 0) {
            action = "ACCELERATE";
            color = cv::Scalar(255, 255, 0);  // Cyan
        } else {
            action = "DECELERATE";
            color = cv::Scalar(0, 255, 255);  // Yellow
        }
    } else {  // FREE_DRIVE
        if (speedChange > 2.0f) {
            action = "ACCELERATE";
            color = cv::Scalar(255, 255, 0);  // Cyan
        } else {
            action = "CRUISE";
            color = cv::Scalar(0, 255, 0);  // Green
        }
    }
}

float updateEgoSpeedSmooth(float currentSpeed, float targetSpeed, int urgencyLevel, float dt) {
    float speedDiff = targetSpeed - currentSpeed;

    // Determine adjustment rate
    float maxChange;
    if (urgencyLevel >= 3) {
        maxChange = Config::adjustment.aggressive;
    } else if (urgencyLevel >= 2) {
        maxChange = Config::adjustment.moderate;
    } else if (urgencyLevel >= 1) {
        maxChange = Config::adjustment.gentle;
    } else {
        maxChange = Config::adjustment.gentle * 0.5f;
    }

    // Time-based scaling
    float timeFactor = std::min(dt / Config::adjustment.interval, 2.0f);  // Cap at 2x
    maxChange *= timeFactor;

    float newSpeed;
    if (std::abs(speedDiff) <= maxChange) {
        newSpeed = targetSpeed;
    } else if (speedDiff > 0) {
        newSpeed = currentSpeed + maxChange;
    } else {
        newSpeed = currentSpeed - maxChange;
    }

    // Clamp to min/max speed
    newSpeed = std::clamp(newSpeed, Config::adjustment.min, Config::adjustment.max);

    return newSpeed;
}