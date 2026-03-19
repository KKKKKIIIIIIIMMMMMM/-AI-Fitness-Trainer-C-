# AI Fitness Trainer and Rep Counter: Setup Instructions

This document provides step-by-step instructions to set up the development environment for the AI Fitness Trainer and Rep Counter application using Visual Studio, C++/CLI, and OpenCV with YOLOv8-Pose.

## 1. Prerequisites

Before you begin, ensure you have the following installed:

*   **Visual Studio 2019 or later:** Download from the official Microsoft website [1]. During installation, make sure to select the "Desktop development with C++" workload and the ".NET desktop development" workload.
*   **Git (Optional but Recommended):** For cloning the repository and managing versions [2].

## 2. Project Setup

1.  **Clone or Download the Project:**
    If using Git, clone the repository:
    ```bash
    git clone <repository_url>
    cd ai_fitness_trainer
    ```
    Otherwise, download the project archive and extract it to your desired location.

2.  **Open the Solution in Visual Studio:**
    Navigate to the `ai_fitness_trainer` directory and open the `ai_fitness_trainer.sln` file with Visual Studio.

## 3. OpenCV Installation via NuGet

NuGet is the recommended way to integrate OpenCV into your Visual Studio C++ project due to its simplicity and ease of management.

1.  **Open NuGet Package Manager:**
    In Visual Studio, go to `Tools > NuGet Package Manager > Manage NuGet Packages for Solution...`.

2.  **Search and Install OpenCV:**
    *   Go to the `Browse` tab.
    *   Search for `OpenCV.Win.Native` (or `OpenCV.WebDriver` for older versions, ensure it's compatible with C++).
    *   Select the latest stable version (e.g., 4.x) and install it for your `ai_fitness_trainer` project. This will install the necessary OpenCV libraries and headers.
    *   You might also need `OpenCV.Contrib.Win.Native` if you plan to use modules like `cv::dnn` from `opencv_contrib`.

3.  **Verify Installation:**
    After installation, check your project's `References` or `vcpkg` integration (if you used vcpkg instead of NuGet) to ensure OpenCV libraries are linked correctly.

## 4. YOLOv8-Pose Model Download

The application uses the YOLOv8-Pose model in ONNX format for pose estimation.

1.  **Download `yolov8n-pose.onnx`:**
    Download the `yolov8n-pose.onnx` model file from the official Ultralytics GitHub repository or website [3]. Look for the `yolov8n-pose.onnx` file under the releases or assets section.

2.  **Place the Model File:**
    Create a folder named `models` in the root directory of your project (e.g., `ai_fitness_trainer/models/`). Place the downloaded `yolov8n-pose.onnx` file into this `models` folder.

    Your project structure should look like this:
    ```
    ai_fitness_trainer/
    ├── ai_fitness_trainer.sln
    ├── ai_fitness_trainer/
    │   ├── ... (source files)
    │   └── ai_fitness_trainer.vcxproj
    ├── models/
    │   └── yolov8n-pose.onnx  <-- Place the model here
    └── docs/
        └── SetupInstructions.md
    ```

## 5. Build and Run the Application

1.  **Set Solution Platform:**
    In Visual Studio, ensure the Solution Platform is set to `x64` (Build > Configuration Manager).

2.  **Build the Solution:**
    Go to `Build > Build Solution` or press `F7`. This will compile the C++ and C++/CLI code.

3.  **Run the Application:**
    Press `F5` or go to `Debug > Start Debugging` to run the application. The application window should appear, and you can start the camera and select an exercise.

## References

[1] Microsoft Visual Studio. *Visual Studio Downloads*. Available at: [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/)
[2] Git. *Downloads*. Available at: [https://git-scm.com/downloads](https://git-scm.com/downloads)
[3] Ultralytics. *YOLOv8*. Available at: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) (Look for `yolov8n-pose.onnx` in releases or assets) 
    model hub.)
