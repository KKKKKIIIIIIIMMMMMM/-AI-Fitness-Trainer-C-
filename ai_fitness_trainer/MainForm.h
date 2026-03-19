#pragma once

#include <mutex>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <msclr/marshal_cppstd.h>
#include "CameraHandler.h"
#include "PoseEstimator.h"
#include "ExerciseClassifier.h"
#include "InjuryRiskAssessor.h"
#include "ExerciseController.h"
#include "UserProfile.h"
#include "ProfileForm.h"

namespace ai_fitness_trainer {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::Drawing::Imaging;
	using namespace System::Drawing::Drawing2D;
	using namespace System::Runtime::InteropServices;

	// Workout Program Data
	struct WorkoutExerciseStep { int exerciseType; int sets; int reps; };
	struct WorkoutProgram {
		const wchar_t* nameEn;
		const wchar_t* nameTh;
		const wchar_t* desc;
		int stepCount;
		WorkoutExerciseStep steps[6];
	};
	static const WorkoutProgram PROGRAMS[5] = {
		{ L"Beginner Full Body", L"\u0E1C\u0E39\u0E49\u0E40\u0E23\u0E34\u0E48\u0E21\u0E15\u0E49\u0E19 (\u0E17\u0E31\u0E49\u0E07\u0E23\u0E48\u0E32\u0E07\u0E01\u0E32\u0E22)",
		  L"Squat - Push-up - Bicep Curl", 3, {{1,3,12},{2,3,10},{0,3,12}} },
		{ L"Upper Body Strength", L"\u0E01\u0E25\u0E49\u0E32\u0E21\u0E40\u0E19\u0E37\u0E49\u0E2D\u0E2A\u0E48\u0E27\u0E19\u0E1A\u0E19",
		  L"Bicep Curl - Shoulder Press - Lateral Raise - Pull-Up", 4, {{0,3,12},{4,3,10},{6,3,15},{9,3,8}} },
		{ L"Lower Body Power", L"\u0E01\u0E25\u0E49\u0E32\u0E21\u0E40\u0E19\u0E37\u0E49\u0E2D\u0E2A\u0E48\u0E27\u0E19\u0E25\u0E48\u0E32\u0E07",
		  L"Squat - Lunge - Hip Thrust - Leg Extension", 4, {{1,3,15},{3,3,12},{10,3,12},{11,3,15}} },
		{ L"Core & Endurance", L"\u0E41\u0E01\u0E19\u0E01\u0E25\u0E32\u0E07\u0E41\u0E25\u0E30\u0E04\u0E27\u0E32\u0E21\u0E2D\u0E14\u0E17\u0E19",
		  L"Plank - Push-up - Deadlift", 3, {{7,3,30},{2,3,15},{5,3,10}} },
		{ L"Home Workout", L"\u0E2D\u0E2D\u0E01\u0E01\u0E33\u0E25\u0E31\u0E07\u0E01\u0E32\u0E22\u0E17\u0E35\u0E48\u0E1A\u0E49\u0E32\u0E19",
		  L"Push-up - Squat - Lunge", 3, {{2,3,15},{1,3,15},{3,3,10}} }
	};

	public ref class MainForm : public System::Windows::Forms::Form
	{
	public:
		MainForm(void)
		{
			InitializeComponent();
			cameraHandler      = new CameraHandler();
			poseEstimator      = new PoseEstimator();
			exerciseClassifier = new ExerciseClassifier();
			injuryRiskAssessor = new InjuryRiskAssessor();
			exerciseController = new ExerciseController();

			// Threading state
			frameMutex     = new std::mutex();
			queueMutex     = new std::mutex();
			frameCondVar   = new std::condition_variable();
			frameQueue     = new std::queue<cv::Mat>();
			sharedFrame    = new cv::Mat();
			sharedFeedback         = new std::string("Ready");
			sharedQuality          = new std::string();
			sharedDetectedExercise = new std::string();
			sharedInjuryMsg        = new std::string("Good form");
			sharedCaptureError     = new std::string();
			customSteps            = new std::vector<WorkoutExerciseStep>();
			threadRunning = false;
			resetRequested = false;
			exerciseChangeRequested = false;
			hasNewFrame    = false;
			mirrorEnabled  = true;
			isResting = false;
			modelReady = false;
			consecutiveReadFails = 0;

			// Model loads asynchronously in Form_Shown — show loading state now
			lblGpuStatus->ForeColor = colWarning;
			lblGpuStatus->Text      = L"\u25CF  Loading model...";
			btnStart->Enabled       = false;

			// ── Profile ─────────────────────────────────────────────────────
			profile = UserProfile::Load();
			if (!profile.isValid) {
				ProfileForm^ pf = gcnew ProfileForm(profile, false);
				pf->ShowDialog(this);
				profile = pf->profile;
			}
			UpdateProfileDisplay();
			if (lblDashUser != nullptr)
				lblDashUser->Text = L"USER: " + profile.name;
		}

	protected:
		~MainForm()
		{
			threadRunning = false;
			if (frameCondVar) frameCondVar->notify_all(); // wake up inference thread

			if (cameraHandler) cameraHandler->releaseCamera();
			if (cameraThread != nullptr && cameraThread->IsAlive)
				cameraThread->Join(2000);
			if (inferenceThread != nullptr && inferenceThread->IsAlive)
				inferenceThread->Join(2000);

			if (components)         delete components;
			if (exerciseController) delete exerciseController;
			if (injuryRiskAssessor) delete injuryRiskAssessor;
			if (exerciseClassifier) delete exerciseClassifier;
			if (poseEstimator)      delete poseEstimator;
			if (cameraHandler)      delete cameraHandler;
			
			delete frameCondVar;
			delete queueMutex;
			delete frameQueue;
			delete frameMutex;
			delete sharedFrame;
			delete sharedFeedback;
			delete sharedQuality;
			delete sharedDetectedExercise;
			delete sharedInjuryMsg;
			delete sharedCaptureError;
			delete customSteps;
		}

	private:
		// ── Core ─────────────────────────────────────────────────────────────
		CameraHandler*      cameraHandler;
		PoseEstimator*      poseEstimator;
		ExerciseClassifier* exerciseClassifier;
		InjuryRiskAssessor* injuryRiskAssessor;
		ExerciseController* exerciseController;
		bool isRunning = false;

		// ── UI Controls ──────────────────────────────────────────────────────
		System::Windows::Forms::PictureBox^ picBoxGuide;

		// ── Session tracking ─────────────────────────────────────────────────
		double           currentFps        = 0.0;
		System::DateTime sessionStartTime;
		int              lastKnownRepCount = 0;
		UserProfile      profile;
		float            sessionCalories   = 0.0f;

		// ── Cumulative stats to survive inter-set resets ─────────────────────
		System::DateTime workoutStartTime;
		int              sessionTotalReps   = 0;
		int              sessionFullReps    = 0;
		int              sessionPartialReps = 0;
		int              sessionPoorReps    = 0;

		// ── Sets tracking ────────────────────────────────────────────────────
		int  currentSet    = 1;
		int  targetSets    = 3;
		int  targetReps    = 12;
		bool allSetsComplete = false;

		// ── Rest timer ───────────────────────────────────────────────────────
		volatile bool    isResting      = false;
		volatile bool    isPaused       = false; // Added
		int              restDurationSec = 60;
		System::DateTime restStartTime;

		// ── Mirror ───────────────────────────────────────────────────────────
		bool mirrorEnabled = true;

		// ── Model loading ────────────────────────────────────────────────────
		volatile bool    modelReady     = false;

		// ── Background thread ────────────────────────────────────────────────
		System::Threading::Thread^ cameraThread;
		System::Threading::Thread^ inferenceThread;
		volatile bool    threadRunning  = false;
		volatile bool    resetRequested = false;
		volatile bool    exerciseChangeRequested = false;
		int           pendingExerciseIndex = 0;

		// ── Capture thread error ───────────────────────────────────────────
		std::string* sharedCaptureError;
		int          consecutiveReadFails = 0;
		static constexpr int MAX_READ_FAILS = 90;  // ~3 sec at 30fps

		// Async Pipeline (Producer-Consumer)
		std::mutex*              queueMutex;
		std::condition_variable* frameCondVar;
		std::queue<cv::Mat>*     frameQueue;

		// Shared state (mutex-protected)
		std::mutex*  frameMutex;
		cv::Mat*     sharedFrame;
		bool         hasNewFrame    = false;
		int          sharedRepCount = 0;
		std::string* sharedFeedback;
		std::string* sharedQuality;
		double       sharedAngle    = 0.0;
		bool         sharedIsHold   = false;
		double       sharedFps      = 0.0;
		int          sharedFull     = 0;
		int          sharedPartial  = 0;
		int          sharedPoor     = 0;

		// ── Classifier + Injury + View shared state ──────────────────────────
		std::string* sharedDetectedExercise;
		float        sharedDetectedConf      = 0.0f;
		int          sharedDetectedClassIndex= -1;    // 0-21 raw classifier class index
		int          sharedInjuryLevel       = 0;     // 0=Safe, 1=Warning, 2=Danger
		std::string* sharedInjuryMsg;
		int          sharedViewAngle         = 0;     // 0=Unknown,1=Front,2=LeftSide,3=RightSide
		int          sharedPoseHintClass     = -1;    // ExerciseType int from pose geometry (-1=unknown)
		float        sharedPoseHintConf      = 0.0f;

		// ── Sensor Fusion Parameters ────────────────────────────────────────
		// Video classifier and pose hint are combined as follows:
		//   - If both agree (mapped exercise == pose hint): boost video confidence by pose*0.2
		//   - If both present but disagree: keep video but penalize by 0.85x
		//   - If only one is confident enough: use it alone
		static constexpr float VIDEO_CONFIDENCE_THRESHOLD = 0.35f;  // Min confidence for video
		static constexpr float POSE_CONFIDENCE_THRESHOLD  = 0.42f;  // Min confidence for pose
		static constexpr float FUSION_BOOST_FACTOR        = 0.2f;   // Weight pose in agreement
		static constexpr float DISAGREEMENT_PENALTY       = 0.85f;  // Reduce video on disagree

		// ── Theme ────────────────────────────────────────────────────────────
		static Color colBg            = Color::FromArgb(18, 18, 30);
		static Color colHeader        = Color::FromArgb(10, 36, 72);
		static Color colHeaderAccent  = Color::FromArgb(14, 44, 86);
		static Color colSidebar       = Color::FromArgb(22, 33, 54);
		static Color colCard          = Color::FromArgb(28, 42, 66);
		static Color colSidebarBorder = Color::FromArgb(40, 55, 85);
		static Color colAccent        = Color::FromArgb(0, 200, 155);
		static Color colTextPrimary   = Color::White;
		static Color colTextSecondary = Color::FromArgb(140, 155, 180);
		static Color colBtnStart      = Color::FromArgb(46, 160, 67);
		static Color colBtnStop       = Color::FromArgb(210, 60, 50);
		static Color colBtnReset      = Color::FromArgb(50, 110, 200);
		static Color colBarBg         = Color::FromArgb(35, 45, 65);
		static Color colSuccess       = Color::FromArgb(76, 175, 80);
		static Color colWarning       = Color::FromArgb(255, 193, 7);
		static Color colError         = Color::FromArgb(244, 67, 54);
		static Color colRest          = Color::FromArgb(255, 152, 0);

		 // Navigation state
	int  currentProgramIndex = -1;
	int  preWorkoutExercise  = 0;
	int  preWorkoutSets      = 3;
	int  preWorkoutReps      = 12;
	std::vector<WorkoutExerciseStep>* customSteps;
	bool isCustomProgram     = false;
	int  summaryTotalReps    = 0;
	int  summaryTotalSets    = 0;
	float summaryCalBurned   = 0.0f;
	int  summaryFull2        = 0;
	int  summaryPartial2     = 0;
	int  summaryPoor2        = 0;
	String^ summaryDuration2  = L"00:00";
	String^ summaryExerciseName = L"None";


	private: System::Windows::Forms::PictureBox^  pictureBox1;
	private: System::Windows::Forms::ComboBox^    comboBoxExercise;
	private: System::Windows::Forms::Label^       lblRepCount;
	private: System::Windows::Forms::Label^       lblFeedback;
	private: System::Windows::Forms::Button^      btnStart;
	private: System::Windows::Forms::Button^      btnStop;
	private: System::Windows::Forms::Button^      btnReset;
	private: System::Windows::Forms::Timer^       timer1;
	private: System::ComponentModel::IContainer^  components;
	private: System::Windows::Forms::Panel^       panelHeader;
	private: System::Windows::Forms::Label^       lblTitle;
	private: System::Windows::Forms::Panel^       panelSidebar;
	private: System::Windows::Forms::Label^       lblExerciseLabel;
	private: System::Windows::Forms::Label^       lblCameraLabel;
	private: System::Windows::Forms::TextBox^     txtCameraSource;
	private: System::Windows::Forms::Label^       lblWeightLabel;
	private: System::Windows::Forms::TextBox^     txtWeight;
	private: System::Windows::Forms::Label^       lblRepsLabel;
	private: System::Windows::Forms::Label^       lblTimer;
	private: System::Windows::Forms::Label^       lblAngleLabel;
	private: System::Windows::Forms::Panel^       panelAngleBarBg;
	private: System::Windows::Forms::Panel^       panelAngleBarFill;
	private: System::Windows::Forms::Label^       lblAngleValue;
	private: System::Windows::Forms::Label^       lblStatus;
	private: System::Windows::Forms::Label^       lblGpuStatus;
	private: System::Windows::Forms::Panel^       panelCameraBorder;
	private: System::Windows::Forms::ToolTip^     toolTip1;
	private: System::Windows::Forms::Label^       lblProfileInfo;
	private: System::Windows::Forms::Label^       lblGoalTarget;
	private: System::Windows::Forms::Label^       lblCalories;
	private: System::Windows::Forms::Button^      btnEditProfile;
			 // ── New controls ──
	private: System::Windows::Forms::Label^       lblSet;
	private: System::Windows::Forms::Panel^       panelRepProgressBg;
	private: System::Windows::Forms::Panel^       panelRepProgressFill;
	private: System::Windows::Forms::Label^       lblRepProgress;
	private: System::Windows::Forms::Label^       lblQualityFull;
	private: System::Windows::Forms::Label^       lblQualityPartial;
	private: System::Windows::Forms::Label^       lblQualityPoor;
	private: System::Windows::Forms::Button^      btnMirror;
	private: System::Windows::Forms::Button^      btnPause; // Added
	private: System::Windows::Forms::Label^       lblNextExercise; // Added
		 // ── Auto-detection + Injury Risk + View Angle controls ──
	private: System::Windows::Forms::Label^       lblAutoExercise;
	private: System::Windows::Forms::Label^       lblInjuryRisk;
	private: System::Windows::Forms::Label^       lblViewAngle;
		 // Screen panels
	private: System::Windows::Forms::Panel^       panelHome;
	private: System::Windows::Forms::Panel^       panelPreWorkout;
	private: System::Windows::Forms::Panel^       panelSummary;
	private: System::Windows::Forms::Panel^       panelHistory;
		 // Home screen
	private: System::Windows::Forms::Label^       lblHomeGreeting;
	private: System::Windows::Forms::Label^       lblHomeDailyGoalHdr;
	private: System::Windows::Forms::Panel^       panelGoalBarBg;
	private: System::Windows::Forms::Panel^       panelGoalBarFill;
	private: System::Windows::Forms::Label^       lblGoalProgress;
	private: System::Windows::Forms::Button^      btnQuickStart;
	private: System::Windows::Forms::Button^      btnViewHistory;
	private: System::Windows::Forms::Label^       lblProgramsHdr;
	private: System::Windows::Forms::Button^      btnProgram0;
	private: System::Windows::Forms::Button^      btnProgram1;
	private: System::Windows::Forms::Button^      btnProgram2;
	private: System::Windows::Forms::Button^      btnProgram3;
	private: System::Windows::Forms::Button^      btnProgram4;
		 // Pre-Workout screen
	private: System::Windows::Forms::Label^       lblPreWorkoutTitle;
	private: System::Windows::Forms::Label^       lblSelectedProgram;
	private: System::Windows::Forms::Label^       lblExercisePickHdr;
	private: System::Windows::Forms::FlowLayoutPanel^ flowExercisePicker;
	private: System::Windows::Forms::Label^       lblPreSetsHdr;
	private: System::Windows::Forms::NumericUpDown^ numSets;
	private: System::Windows::Forms::Label^       lblPreRepsHdr;
	private: System::Windows::Forms::NumericUpDown^ numReps;
	private: System::Windows::Forms::Label^       lblPreCamHdr;
	private: System::Windows::Forms::TextBox^     txtCameraSourcePre;
	private: System::Windows::Forms::Button^      btnStartWorkout;
	private: System::Windows::Forms::Button^      btnBackPreWorkout;
	private: System::Windows::Forms::Label^       lblCurrentExercise;
		 // Summary screen
	private: System::Windows::Forms::Label^       lblSummaryTitle;
	private: System::Windows::Forms::Label^       lblSummaryExercise;
	private: System::Windows::Forms::Label^       lblSummaryReps;
	private: System::Windows::Forms::Label^       lblSummaryDuration;
	private: System::Windows::Forms::Label^       lblSummarySets;
	private: System::Windows::Forms::Label^       lblSummaryCalories;
	private: System::Windows::Forms::Label^       lblSummaryQualityHdr;
	private: System::Windows::Forms::Label^       lblSummaryFull;
	private: System::Windows::Forms::Label^       lblSummaryPartial;
	private: System::Windows::Forms::Label^       lblSummaryPoor;
	private: System::Windows::Forms::Label^       lblSummaryMotivation;
	private: System::Windows::Forms::Button^      btnSummaryHome;
	private: System::Windows::Forms::Button^      btnSummaryAgain;
		 // Custom Program picker
	private: System::Windows::Forms::ListBox^     listCustomProgram;
	private: System::Windows::Forms::Button^      btnAddCustom;
	private: System::Windows::Forms::Button^      btnClearCustom;
		 // Dashboard Overlays
	private: System::Windows::Forms::FlowLayoutPanel^ flowWorkoutBar;
	private: System::Windows::Forms::Label^       lblDashSession;
	private: System::Windows::Forms::Label^       lblDashUser;
	private: System::Windows::Forms::Label^       lblDashRep;
	private: System::Windows::Forms::Label^       lblDashPose;
	private: System::Windows::Forms::Label^       lblDashBPM;
	private: System::Windows::Forms::Label^       lblDashTime;
	private: System::Windows::Forms::Button^      btnDashEndWorkout;
		 // History screen
	private: System::Windows::Forms::Label^       lblHistoryTitle;
	private: System::Windows::Forms::Label^       lblHistoryStats;
	private: System::Windows::Forms::ListBox^     listHistory;
	private: System::Windows::Forms::Button^      btnHistoryBack;


#pragma region Windows Form Designer generated code
		void InitializeComponent(void)
		{
			this->components = (gcnew System::ComponentModel::Container());

			this->panelHeader = (gcnew System::Windows::Forms::Panel());
			this->lblTitle = (gcnew System::Windows::Forms::Label());
			this->panelCameraBorder = (gcnew System::Windows::Forms::Panel());
			this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			this->panelSidebar = (gcnew System::Windows::Forms::Panel());
			this->lblExerciseLabel = (gcnew System::Windows::Forms::Label());
			this->comboBoxExercise = (gcnew System::Windows::Forms::ComboBox());
			this->lblCameraLabel = (gcnew System::Windows::Forms::Label());
			this->txtCameraSource = (gcnew System::Windows::Forms::TextBox());
			this->lblWeightLabel = (gcnew System::Windows::Forms::Label());
			this->txtWeight = (gcnew System::Windows::Forms::TextBox());
			this->lblRepCount = (gcnew System::Windows::Forms::Label());
			this->lblRepsLabel = (gcnew System::Windows::Forms::Label());
			this->lblTimer = (gcnew System::Windows::Forms::Label());
			this->lblFeedback = (gcnew System::Windows::Forms::Label());
			this->lblAngleLabel = (gcnew System::Windows::Forms::Label());
			this->panelAngleBarBg = (gcnew System::Windows::Forms::Panel());
			this->panelAngleBarFill = (gcnew System::Windows::Forms::Panel());
			this->lblAngleValue = (gcnew System::Windows::Forms::Label());
			this->btnStart = (gcnew System::Windows::Forms::Button());
			this->btnStop = (gcnew System::Windows::Forms::Button());
			this->btnReset = (gcnew System::Windows::Forms::Button());
			this->lblStatus = (gcnew System::Windows::Forms::Label());
			this->lblGpuStatus = (gcnew System::Windows::Forms::Label());
			this->timer1 = (gcnew System::Windows::Forms::Timer(this->components));
			this->toolTip1 = (gcnew System::Windows::Forms::ToolTip(this->components));
			this->btnEditProfile = (gcnew System::Windows::Forms::Button());
			this->lblProfileInfo = (gcnew System::Windows::Forms::Label());
			this->lblGoalTarget  = (gcnew System::Windows::Forms::Label());
			this->lblCalories    = (gcnew System::Windows::Forms::Label());
			// Guide GIF Overlay
			this->picBoxGuide    = (gcnew System::Windows::Forms::PictureBox());
			// New controls
			this->lblSet              = (gcnew System::Windows::Forms::Label());
			this->panelRepProgressBg  = (gcnew System::Windows::Forms::Panel());
			this->panelRepProgressFill= (gcnew System::Windows::Forms::Panel());
			this->lblRepProgress      = (gcnew System::Windows::Forms::Label());
			this->lblQualityFull      = (gcnew System::Windows::Forms::Label());
			this->lblQualityPartial   = (gcnew System::Windows::Forms::Label());
			this->lblQualityPoor      = (gcnew System::Windows::Forms::Label());
			this->btnMirror           = (gcnew System::Windows::Forms::Button());
			this->btnPause            = (gcnew System::Windows::Forms::Button());
			this->lblNextExercise     = (gcnew System::Windows::Forms::Label());
			this->lblAutoExercise     = (gcnew System::Windows::Forms::Label());
			this->lblInjuryRisk       = (gcnew System::Windows::Forms::Label());
			this->lblViewAngle        = (gcnew System::Windows::Forms::Label());

			// New screen panels
		this->panelHome        = (gcnew System::Windows::Forms::Panel());
		this->panelPreWorkout  = (gcnew System::Windows::Forms::Panel());
		this->panelSummary     = (gcnew System::Windows::Forms::Panel());
		this->panelHistory     = (gcnew System::Windows::Forms::Panel());
		// Home screen
		this->lblHomeGreeting    = (gcnew System::Windows::Forms::Label());
		this->lblHomeDailyGoalHdr= (gcnew System::Windows::Forms::Label());
		this->panelGoalBarBg     = (gcnew System::Windows::Forms::Panel());
		this->panelGoalBarFill   = (gcnew System::Windows::Forms::Panel());
		this->lblGoalProgress    = (gcnew System::Windows::Forms::Label());
		this->btnQuickStart      = (gcnew System::Windows::Forms::Button());
		this->btnViewHistory     = (gcnew System::Windows::Forms::Button());
		this->lblProgramsHdr     = (gcnew System::Windows::Forms::Label());
		this->btnProgram0        = (gcnew System::Windows::Forms::Button());
		this->btnProgram1        = (gcnew System::Windows::Forms::Button());
		this->btnProgram2        = (gcnew System::Windows::Forms::Button());
		this->btnProgram3        = (gcnew System::Windows::Forms::Button());
		this->btnProgram4        = (gcnew System::Windows::Forms::Button());
		// Pre-Workout screen
		this->lblPreWorkoutTitle  = (gcnew System::Windows::Forms::Label());
		this->lblSelectedProgram  = (gcnew System::Windows::Forms::Label());
		this->lblExercisePickHdr  = (gcnew System::Windows::Forms::Label());
		this->flowExercisePicker  = (gcnew System::Windows::Forms::FlowLayoutPanel());
		this->lblPreSetsHdr       = (gcnew System::Windows::Forms::Label());
		this->numSets             = (gcnew System::Windows::Forms::NumericUpDown());
		this->lblPreRepsHdr       = (gcnew System::Windows::Forms::Label());
		this->numReps             = (gcnew System::Windows::Forms::NumericUpDown());
		this->lblPreCamHdr        = (gcnew System::Windows::Forms::Label());
		this->txtCameraSourcePre  = (gcnew System::Windows::Forms::TextBox());
		this->btnStartWorkout     = (gcnew System::Windows::Forms::Button());
		this->btnBackPreWorkout   = (gcnew System::Windows::Forms::Button());
		this->lblCurrentExercise  = (gcnew System::Windows::Forms::Label());
		// Summary screen
		this->lblSummaryTitle     = (gcnew System::Windows::Forms::Label());
		this->lblSummaryExercise  = (gcnew System::Windows::Forms::Label());
		this->lblSummaryReps      = (gcnew System::Windows::Forms::Label());
		this->lblSummaryDuration  = (gcnew System::Windows::Forms::Label());
		this->lblSummarySets      = (gcnew System::Windows::Forms::Label());
		this->lblSummaryCalories  = (gcnew System::Windows::Forms::Label());
		this->lblSummaryQualityHdr= (gcnew System::Windows::Forms::Label());
		this->lblSummaryFull      = (gcnew System::Windows::Forms::Label());
		this->lblSummaryPartial   = (gcnew System::Windows::Forms::Label());
		this->lblSummaryPoor      = (gcnew System::Windows::Forms::Label());
		this->lblSummaryMotivation= (gcnew System::Windows::Forms::Label());
		this->btnSummaryHome      = (gcnew System::Windows::Forms::Button());
		this->btnSummaryAgain     = (gcnew System::Windows::Forms::Button());
		// Custom Program picker
		this->listCustomProgram   = (gcnew System::Windows::Forms::ListBox());
		this->btnAddCustom        = (gcnew System::Windows::Forms::Button());
		this->btnClearCustom      = (gcnew System::Windows::Forms::Button());
		// Dashboard Overlays
		this->flowWorkoutBar      = (gcnew System::Windows::Forms::FlowLayoutPanel());
		this->lblDashSession      = (gcnew System::Windows::Forms::Label());
		this->lblDashUser         = (gcnew System::Windows::Forms::Label());
		this->lblDashRep          = (gcnew System::Windows::Forms::Label());
		this->lblDashPose         = (gcnew System::Windows::Forms::Label());
		this->lblDashBPM          = (gcnew System::Windows::Forms::Label());
		this->lblDashTime         = (gcnew System::Windows::Forms::Label());
		this->btnDashEndWorkout   = (gcnew System::Windows::Forms::Button());
		// History screen
		this->lblHistoryTitle  = (gcnew System::Windows::Forms::Label());
		this->lblHistoryStats  = (gcnew System::Windows::Forms::Label());
		this->listHistory      = (gcnew System::Windows::Forms::ListBox());
		this->btnHistoryBack   = (gcnew System::Windows::Forms::Button());

		(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			this->panelHeader->SuspendLayout();
			this->panelCameraBorder->SuspendLayout();
			this->panelSidebar->SuspendLayout();
			this->panelAngleBarBg->SuspendLayout();
			this->SuspendLayout();

			// ═══════════════ Header ═══════════════
			this->panelHeader->BackColor = colHeader;
			this->panelHeader->Dock = System::Windows::Forms::DockStyle::Top;
			this->panelHeader->Size = System::Drawing::Size(1280, 50);
			this->panelHeader->Controls->Add(this->lblTitle);
			this->panelHeader->Controls->Add(this->btnEditProfile);

			this->lblTitle->AutoSize = false;
			this->lblTitle->Location = System::Drawing::Point(0, 0);
			this->lblTitle->Size = System::Drawing::Size(870, 52);
			this->lblTitle->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16, System::Drawing::FontStyle::Bold));
			this->lblTitle->ForeColor = colTextPrimary;
			this->lblTitle->Text = L"   AI FITNESS TRAINER";
			this->lblTitle->TextAlign = System::Drawing::ContentAlignment::MiddleLeft;

			this->btnEditProfile->BackColor = colSidebarBorder;
			this->btnEditProfile->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnEditProfile->FlatAppearance->BorderSize = 1;
			this->btnEditProfile->FlatAppearance->MouseOverBackColor = Color::FromArgb(55, 72, 100);
			this->btnEditProfile->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9));
			this->btnEditProfile->ForeColor = colTextPrimary;
			this->btnEditProfile->Location = System::Drawing::Point(906, 10);
			this->btnEditProfile->Size = System::Drawing::Size(168, 32);
			this->btnEditProfile->Text = L"\u2699  Edit Profile";
			this->btnEditProfile->UseVisualStyleBackColor = false;
			this->btnEditProfile->Cursor = System::Windows::Forms::Cursors::Hand;
			this->btnEditProfile->Click += gcnew System::EventHandler(this, &MainForm::btnEditProfile_Click);

			// ═══════════════ Camera Border (Centered 16:9) ═══════════════
			this->panelCameraBorder->BackColor = colBg;
			this->panelCameraBorder->Location = System::Drawing::Point(0, 50);
			this->panelCameraBorder->Size = System::Drawing::Size(1280, 720);
			this->panelCameraBorder->Controls->Add(this->pictureBox1);
			this->pictureBox1->BackColor = Color::Black;
			this->pictureBox1->Size = System::Drawing::Size(1280, 720);
			this->pictureBox1->Location = System::Drawing::Point(
				(this->panelCameraBorder->Width - this->pictureBox1->Width) / 2,
				0
			);
			this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::StretchImage;
			this->pictureBox1->TabStop = false;
			this->pictureBox1->Paint += gcnew System::Windows::Forms::PaintEventHandler(this, &MainForm::pictureBox1_Paint);

			// ═══════════════ Guide GIF Overlay ═══════════════
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->picBoxGuide))->BeginInit();
			
			// The invisible PictureBox simply acts as a GIF frame manager for the Paint event
			this->picBoxGuide->BackColor = Color::Transparent;
			this->picBoxGuide->Size = System::Drawing::Size(300, 300);
			this->picBoxGuide->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			this->picBoxGuide->TabStop = false;
			this->picBoxGuide->Visible = false; // We draw it manually in Paint event
			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->picBoxGuide))->EndInit();

			// ═══════════════ Sidebar (Now Bottom Action Bar) ═══════════════
			this->panelSidebar->BackColor = colSidebar;
			this->panelSidebar->Location = System::Drawing::Point(0, 770);
			this->panelSidebar->Size = System::Drawing::Size(1280, 210);
			this->panelSidebar->Controls->Add(this->lblExerciseLabel);
			this->panelSidebar->Controls->Add(this->comboBoxExercise);
			this->panelSidebar->Controls->Add(this->lblCameraLabel);
			this->panelSidebar->Controls->Add(this->txtCameraSource);
			this->panelSidebar->Controls->Add(this->lblWeightLabel);
			this->panelSidebar->Controls->Add(this->txtWeight);
			this->panelSidebar->Controls->Add(this->lblSet);
			this->panelSidebar->Controls->Add(this->lblRepCount);
			this->panelSidebar->Controls->Add(this->panelRepProgressBg);
			this->panelSidebar->Controls->Add(this->lblRepProgress);
			this->panelSidebar->Controls->Add(this->lblRepsLabel);
			this->panelSidebar->Controls->Add(this->lblTimer);
			this->panelSidebar->Controls->Add(this->lblFeedback);
			this->panelSidebar->Controls->Add(this->lblAngleLabel);
			this->panelSidebar->Controls->Add(this->panelAngleBarBg);
			this->panelSidebar->Controls->Add(this->lblAngleValue);
			this->panelSidebar->Controls->Add(this->lblQualityFull);
			this->panelSidebar->Controls->Add(this->lblQualityPartial);
			this->panelSidebar->Controls->Add(this->lblQualityPoor);
			this->panelSidebar->Controls->Add(this->btnStart);
			this->panelSidebar->Controls->Add(this->btnStop);
			this->panelSidebar->Controls->Add(this->btnPause);
			this->panelSidebar->Controls->Add(this->btnReset);
			this->panelSidebar->Controls->Add(this->btnMirror);
			this->panelSidebar->Controls->Add(this->lblNextExercise);
			this->panelSidebar->Controls->Add(this->lblStatus);
			this->panelSidebar->Controls->Add(this->lblGpuStatus);
			this->panelSidebar->Controls->Add(this->lblProfileInfo);
			this->panelSidebar->Controls->Add(this->lblGoalTarget);
			this->panelSidebar->Controls->Add(this->lblCalories);

			// ----- Exercise Selection -----
			this->lblExerciseLabel->AutoSize = true;
			this->lblExerciseLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Bold));
			this->lblExerciseLabel->ForeColor = colTextSecondary;
			this->lblExerciseLabel->Location = System::Drawing::Point(10, 8);
			this->lblExerciseLabel->Text = L"SELECT EXERCISE";

			this->comboBoxExercise->BackColor = colBarBg;
			this->comboBoxExercise->ForeColor = colTextPrimary;
			this->comboBoxExercise->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->comboBoxExercise->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			this->comboBoxExercise->DropDownStyle = System::Windows::Forms::ComboBoxStyle::DropDownList;
			this->comboBoxExercise->FormattingEnabled = true;
			this->comboBoxExercise->Items->AddRange(gcnew cli::array< System::Object^ >(23) {
				L"Bicep Curl", L"Squat", L"Push-up", L"Lunge",
				L"Shoulder Press", L"Deadlift", L"Lateral Raise", L"Plank",
				L"Tricep Pushdown", L"Pull-Up", L"Hip Thrust", L"Leg Extension",
				L"Bench Press", L"Chest Fly", L"Decline Bench Press", L"Hammer Curl",
				L"Incline Bench Press", L"Lat Pulldown", L"Leg Raises", L"Romanian Deadlift",
				L"Russian Twist", L"T-Bar Row", L"Tricep Dips"
			});
			this->comboBoxExercise->Location = System::Drawing::Point(10, 25);
			this->comboBoxExercise->Size = System::Drawing::Size(260, 28);
			this->comboBoxExercise->TabIndex = 1;
			this->comboBoxExercise->SelectedIndexChanged += gcnew System::EventHandler(this, &MainForm::comboBoxExercise_SelectedIndexChanged);

			this->lblNextExercise->AutoSize = true;
			this->lblNextExercise->Font = (gcnew System::Drawing::Font(L"Segoe UI", 8.5f, System::Drawing::FontStyle::Italic));
			this->lblNextExercise->ForeColor = colTextSecondary;
			this->lblNextExercise->Location = System::Drawing::Point(10, 56);
			this->lblNextExercise->Text = L"Next: --";
			this->lblNextExercise->Visible = false;

			// ----- Camera + Weight -----
			this->lblCameraLabel->AutoSize = true;
			this->lblCameraLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 8, System::Drawing::FontStyle::Bold));
			this->lblCameraLabel->ForeColor = colTextSecondary;
			this->lblCameraLabel->Location = System::Drawing::Point(10, 76);
			this->lblCameraLabel->Text = L"CAMERA";

			this->txtCameraSource->BackColor = colBarBg;
			this->txtCameraSource->ForeColor = colTextPrimary;
			this->txtCameraSource->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			this->txtCameraSource->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtCameraSource->Location = System::Drawing::Point(10, 92);
			this->txtCameraSource->Size = System::Drawing::Size(115, 25);
			this->txtCameraSource->Text = L"0";
			this->txtCameraSource->TabIndex = 2;
			this->toolTip1->SetToolTip(this->txtCameraSource,
				L"Camera index (0,1,2...) or URL (http/rtsp).");

			this->lblWeightLabel->AutoSize = true;
			this->lblWeightLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 8, System::Drawing::FontStyle::Bold));
			this->lblWeightLabel->ForeColor = colTextSecondary;
			this->lblWeightLabel->Location = System::Drawing::Point(135, 76);
			this->lblWeightLabel->Text = L"WEIGHT (KG)";

			this->txtWeight->BackColor = colBarBg;
			this->txtWeight->ForeColor = colTextPrimary;
			this->txtWeight->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10));
			this->txtWeight->BorderStyle = System::Windows::Forms::BorderStyle::FixedSingle;
			this->txtWeight->Location = System::Drawing::Point(135, 92);
			this->txtWeight->Size = System::Drawing::Size(130, 25);
			this->txtWeight->Text = L"0";
			this->txtWeight->TabIndex = 3;

			// ═══════════════ Stats Area ═══════════════

			// ----- Set Indicator -----
			this->lblSet->AutoSize = false;
			this->lblSet->Font = (gcnew System::Drawing::Font(L"Segoe UI", 11, System::Drawing::FontStyle::Bold));
			this->lblSet->ForeColor = colAccent;
			this->lblSet->Location = System::Drawing::Point(285, 8);
			this->lblSet->Size = System::Drawing::Size(170, 18);
			this->lblSet->Text = L"SET 1 / 3";
			this->lblSet->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;

			// ----- Rep Counter -----
			this->lblRepCount->AutoSize = false;
			this->lblRepCount->Font = (gcnew System::Drawing::Font(L"Segoe UI", 48, System::Drawing::FontStyle::Bold));
			this->lblRepCount->ForeColor = colAccent;
			this->lblRepCount->Location = System::Drawing::Point(285, 26);
			this->lblRepCount->Size = System::Drawing::Size(170, 62);
			this->lblRepCount->Text = L"0";
			this->lblRepCount->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;

			// ----- Rep Progress Bar -----
			this->panelRepProgressBg->BackColor = colBarBg;
			this->panelRepProgressBg->Location = System::Drawing::Point(290, 94);
			this->panelRepProgressBg->Size = System::Drawing::Size(130, 8);
			this->panelRepProgressBg->Controls->Add(this->panelRepProgressFill);
			this->panelRepProgressFill = (gcnew System::Windows::Forms::Panel());
			this->panelRepProgressFill->BackColor = colAccent;
			this->panelRepProgressFill->Location = System::Drawing::Point(0, 0);
			this->panelRepProgressFill->Size = System::Drawing::Size(0, 8);
			this->panelRepProgressBg->Controls->Add(this->panelRepProgressFill);

			this->lblRepProgress->AutoSize = true;
			this->lblRepProgress->Font = (gcnew System::Drawing::Font(L"Segoe UI", 8));
			this->lblRepProgress->ForeColor = colTextSecondary;
			this->lblRepProgress->Location = System::Drawing::Point(290, 104);
			this->lblRepProgress->Text = L"0 / 12";

			// ----- Labels -----
			this->lblRepsLabel->AutoSize = false;
			this->lblRepsLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
			this->lblRepsLabel->ForeColor = colTextSecondary;
			this->lblRepsLabel->Location = System::Drawing::Point(285, 116);
			this->lblRepsLabel->Size = System::Drawing::Size(170, 18);
			this->lblRepsLabel->Text = L"REPS";
			this->lblRepsLabel->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;

			this->lblTimer->AutoSize = false;
			this->lblTimer->Font = (gcnew System::Drawing::Font(L"Segoe UI", 13));
			this->lblTimer->ForeColor = colTextSecondary;
			this->lblTimer->Location = System::Drawing::Point(285, 136);
			this->lblTimer->Size = System::Drawing::Size(170, 22);
			this->lblTimer->Text = L"00:00";
			this->lblTimer->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;

			this->lblFeedback->AutoSize = false;
			this->lblFeedback->Font = (gcnew System::Drawing::Font(L"Segoe UI", 13, System::Drawing::FontStyle::Bold));
			this->lblFeedback->ForeColor = colTextSecondary;
			this->lblFeedback->Location = System::Drawing::Point(466, 8);
			this->lblFeedback->Size = System::Drawing::Size(256, 26);
			this->lblFeedback->Text = L"Ready";
			this->lblFeedback->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;

			// ═══════════════ Analytics Area ═══════════════

			// ----- Angle Bar -----
			this->lblAngleLabel->AutoSize = true;
			this->lblAngleLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 8, System::Drawing::FontStyle::Bold));
			this->lblAngleLabel->ForeColor = colTextSecondary;
			this->lblAngleLabel->Location = System::Drawing::Point(466, 42);
			this->lblAngleLabel->Text = L"JOINT ANGLE";

			this->panelAngleBarBg->BackColor = colBarBg;
			this->panelAngleBarBg->Location = System::Drawing::Point(466, 58);
			this->panelAngleBarBg->Size = System::Drawing::Size(200, 14);
			this->panelAngleBarBg->Controls->Add(this->panelAngleBarFill);
			this->panelAngleBarFill->BackColor = colAccent;
			this->panelAngleBarFill->Location = System::Drawing::Point(0, 0);
			this->panelAngleBarFill->Size = System::Drawing::Size(0, 14);

			this->lblAngleValue->AutoSize = true;
			this->lblAngleValue->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Bold));
			this->lblAngleValue->ForeColor = colTextPrimary;
			this->lblAngleValue->Location = System::Drawing::Point(672, 56);
			this->lblAngleValue->Text = L"--\x00B0";

			// ----- Rep Quality Badges -----
			auto lblQualityLabel = (gcnew System::Windows::Forms::Label());
			lblQualityLabel->AutoSize = true;
			lblQualityLabel->Font = (gcnew System::Drawing::Font(L"Segoe UI", 8, System::Drawing::FontStyle::Bold));
			lblQualityLabel->ForeColor = colTextSecondary;
			lblQualityLabel->Location = System::Drawing::Point(466, 80);
			lblQualityLabel->Text = L"REP QUALITY";
			this->panelSidebar->Controls->Add(lblQualityLabel);

			this->lblQualityFull->AutoSize = true;
			this->lblQualityFull->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Bold));
			this->lblQualityFull->ForeColor = colSuccess;
			this->lblQualityFull->Location = System::Drawing::Point(466, 96);
			this->lblQualityFull->Text = L"\u25CF 0 Full";

			this->lblQualityPartial->AutoSize = true;
			this->lblQualityPartial->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Bold));
			this->lblQualityPartial->ForeColor = colWarning;
			this->lblQualityPartial->Location = System::Drawing::Point(528, 96);
			this->lblQualityPartial->Text = L"\u25CF 0 Partial";

			this->lblQualityPoor->AutoSize = true;
			this->lblQualityPoor->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Bold));
			this->lblQualityPoor->ForeColor = colError;
			this->lblQualityPoor->Location = System::Drawing::Point(614, 96);
			this->lblQualityPoor->Text = L"\u25CF 0 Poor";

			// ═══════════════ Buttons ═══════════════

			this->btnStart->BackColor = colBtnStart;
			this->btnStart->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnStart->FlatAppearance->BorderSize = 0;
			this->btnStart->FlatAppearance->MouseOverBackColor = Color::FromArgb(66, 180, 87);
			this->btnStart->FlatAppearance->MouseDownBackColor = Color::FromArgb(36, 140, 57);
			this->btnStart->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
			this->btnStart->ForeColor = colTextPrimary;
			this->btnStart->Location = System::Drawing::Point(738, 8);
			this->btnStart->Size = System::Drawing::Size(108, 38);
			this->btnStart->Text = L"\u25B6  Start";
			this->btnStart->UseVisualStyleBackColor = false;
			this->btnStart->Cursor = System::Windows::Forms::Cursors::Hand;
			this->btnStart->Click += gcnew System::EventHandler(this, &MainForm::btnStart_Click);

			this->btnStop->BackColor = colBtnStop;
			this->btnStop->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnStop->FlatAppearance->BorderSize = 0;
			this->btnStop->FlatAppearance->MouseOverBackColor = Color::FromArgb(230, 80, 70);
			this->btnStop->FlatAppearance->MouseDownBackColor = Color::FromArgb(190, 40, 30);
			this->btnStop->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
			this->btnStop->ForeColor = colTextPrimary;
			this->btnStop->Location = System::Drawing::Point(851, 8);
			this->btnStop->Size = System::Drawing::Size(108, 38);
			this->btnStop->Text = L"\u25A0  Stop";
			this->btnStop->UseVisualStyleBackColor = false;
			this->btnStop->Cursor = System::Windows::Forms::Cursors::Hand;
			this->btnStop->Enabled = false;
			this->btnStop->Click += gcnew System::EventHandler(this, &MainForm::btnStop_Click);

			this->btnPause->BackColor = colWarning;
			this->btnPause->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnPause->FlatAppearance->BorderSize = 0;
			this->btnPause->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
			this->btnPause->ForeColor = colBg;
			this->btnPause->Location = System::Drawing::Point(738, 52);
			this->btnPause->Size = System::Drawing::Size(108, 32);
			this->btnPause->Text = L"\u23F8  Pause";
			this->btnPause->UseVisualStyleBackColor = false;
			this->btnPause->Cursor = System::Windows::Forms::Cursors::Hand;
			this->btnPause->Enabled = false;
			this->btnPause->Click += gcnew System::EventHandler(this, &MainForm::btnPause_Click);

			this->btnReset->BackColor = colBtnReset;
			this->btnReset->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnReset->FlatAppearance->BorderSize = 0;
			this->btnReset->FlatAppearance->MouseOverBackColor = Color::FromArgb(70, 130, 220);
			this->btnReset->FlatAppearance->MouseDownBackColor = Color::FromArgb(30, 90, 180);
			this->btnReset->Font = (gcnew System::Drawing::Font(L"Segoe UI", 10, System::Drawing::FontStyle::Bold));
			this->btnReset->ForeColor = colTextPrimary;
			this->btnReset->Location = System::Drawing::Point(964, 8);
			this->btnReset->Size = System::Drawing::Size(108, 38);
			this->btnReset->Text = L"\u21BA  Reset";
			this->btnReset->UseVisualStyleBackColor = false;
			this->btnReset->Cursor = System::Windows::Forms::Cursors::Hand;
			this->btnReset->Click += gcnew System::EventHandler(this, &MainForm::btnReset_Click);

			// ----- Mirror Toggle -----
			this->btnMirror->BackColor = colAccent;
			this->btnMirror->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->btnMirror->FlatAppearance->BorderSize = 0;
			this->btnMirror->FlatAppearance->MouseOverBackColor = Color::FromArgb(20, 220, 175);
			this->btnMirror->Font = (gcnew System::Drawing::Font(L"Segoe UI", 8.5f, System::Drawing::FontStyle::Bold));
			this->btnMirror->ForeColor = colBg;
			this->btnMirror->Location = System::Drawing::Point(851, 52);
			this->btnMirror->Size = System::Drawing::Size(108, 32);
			this->btnMirror->Text = L"\u2194  Mirror: ON";
			this->btnMirror->UseVisualStyleBackColor = false;
			this->btnMirror->Cursor = System::Windows::Forms::Cursors::Hand;
			this->btnMirror->Click += gcnew System::EventHandler(this, &MainForm::btnMirror_Click);

			// ═══════════════ Status ═══════════════

			this->lblStatus->AutoSize = true;
			this->lblStatus->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9));
			this->lblStatus->ForeColor = colTextSecondary;
			this->lblStatus->Location = System::Drawing::Point(738, 92);
			this->lblStatus->Text = L"\u25CF  Camera: Disconnected";

			this->lblGpuStatus->AutoSize = true;
			this->lblGpuStatus->Font = (gcnew System::Drawing::Font(L"Segoe UI", 9));
			this->lblGpuStatus->ForeColor = colTextSecondary;
			this->lblGpuStatus->Location = System::Drawing::Point(738, 112);
			this->lblGpuStatus->Text = L"\u25CF  GPU: Detecting...";

			// ── Separator + Profile ──
			auto profileSep = (gcnew System::Windows::Forms::Panel());
			profileSep->BackColor = colSidebarBorder;
			profileSep->Location  = System::Drawing::Point(10, 110);
			profileSep->Size      = System::Drawing::Size(264, 1);
			this->panelSidebar->Controls->Add(profileSep);

			this->lblProfileInfo->AutoSize  = false;
			this->lblProfileInfo->Font      = (gcnew System::Drawing::Font(L"Segoe UI", 9));
			this->lblProfileInfo->ForeColor = colTextSecondary;
			this->lblProfileInfo->Location  = System::Drawing::Point(10, 116);
			this->lblProfileInfo->Size      = System::Drawing::Size(264, 18);
			this->lblProfileInfo->Text      = L"\u25A3  User  |  BMI: --";

			this->lblGoalTarget->AutoSize  = false;
			this->lblGoalTarget->Font      = (gcnew System::Drawing::Font(L"Segoe UI", 9));
			this->lblGoalTarget->ForeColor = colAccent;
			this->lblGoalTarget->Location  = System::Drawing::Point(10, 136);
			this->lblGoalTarget->Size      = System::Drawing::Size(264, 18);
			this->lblGoalTarget->Text      = L"\u25BA  General Fitness  |  Target: 12 reps";

			this->lblCalories->AutoSize  = false;
			this->lblCalories->Font      = (gcnew System::Drawing::Font(L"Segoe UI", 9));
			this->lblCalories->ForeColor = colWarning;
			this->lblCalories->Location  = System::Drawing::Point(10, 156);
			this->lblCalories->Size      = System::Drawing::Size(264, 18);
			this->lblCalories->Text      = L"\u25B2  0.0 kcal burned";

			// ── Auto-Detection + Injury Risk ──
			auto aiSep = (gcnew System::Windows::Forms::Panel());
			aiSep->BackColor = colSidebarBorder;
			aiSep->Location  = System::Drawing::Point(466, 120);
			aiSep->Size      = System::Drawing::Size(256, 1);
			this->panelSidebar->Controls->Add(aiSep);

			this->lblAutoExercise->AutoSize  = false;
			this->lblAutoExercise->Font      = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Bold));
			this->lblAutoExercise->ForeColor = colAccent;
			this->lblAutoExercise->Location  = System::Drawing::Point(466, 126);
			this->lblAutoExercise->Size      = System::Drawing::Size(256, 18);
			this->lblAutoExercise->Text      = L"\u25BA  Auto: Waiting for classifier...";

			this->lblInjuryRisk->AutoSize  = false;
			this->lblInjuryRisk->Font      = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Bold));
			this->lblInjuryRisk->ForeColor = colSuccess;
			this->lblInjuryRisk->Location  = System::Drawing::Point(466, 146);
			this->lblInjuryRisk->Size      = System::Drawing::Size(256, 18);
			this->lblInjuryRisk->Text      = L"\u26A0  Form: Good";

			this->lblViewAngle->AutoSize  = false;
			this->lblViewAngle->Font      = (gcnew System::Drawing::Font(L"Segoe UI", 9, System::Drawing::FontStyle::Bold));
			this->lblViewAngle->ForeColor = colTextSecondary;
			this->lblViewAngle->Location  = System::Drawing::Point(466, 166);
			this->lblViewAngle->Size      = System::Drawing::Size(256, 18);
			this->lblViewAngle->Text      = L"\u25C9  View: Detecting...";

			this->panelSidebar->Controls->Add(this->lblAutoExercise);
			this->panelSidebar->Controls->Add(this->lblInjuryRisk);
			this->panelSidebar->Controls->Add(this->lblViewAngle);

			// -- Vertical column separators (horizontal layout) ----
			auto colSep1 = (gcnew System::Windows::Forms::Panel());
			colSep1->BackColor = colSidebarBorder;
			colSep1->Location  = System::Drawing::Point(279, 5);
			colSep1->Size      = System::Drawing::Size(1, 205);
			this->panelSidebar->Controls->Add(colSep1);

			auto colSep2 = (gcnew System::Windows::Forms::Panel());
			colSep2->BackColor = colSidebarBorder;
			colSep2->Location  = System::Drawing::Point(460, 5);
			colSep2->Size      = System::Drawing::Size(1, 205);
			this->panelSidebar->Controls->Add(colSep2);

			auto colSep3 = (gcnew System::Windows::Forms::Panel());
			colSep3->BackColor = colSidebarBorder;
			colSep3->Location  = System::Drawing::Point(731, 5);
			colSep3->Size      = System::Drawing::Size(1, 205);
			this->panelSidebar->Controls->Add(colSep3);

			// ═══════════════ Dashboard Overlays ═══════════════
			this->flowWorkoutBar->BackColor = Color::FromArgb(200, 20, 25, 35);
			this->flowWorkoutBar->Location = System::Drawing::Point(0, 0);
			this->flowWorkoutBar->Size = System::Drawing::Size(1280, 40);
			this->flowWorkoutBar->WrapContents = false;
			this->flowWorkoutBar->AutoScroll = true;

			this->lblDashSession->Parent = this->pictureBox1;
			this->lblDashSession->BackColor = Color::Transparent;
			this->lblDashSession->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold));
			this->lblDashSession->ForeColor = colTextPrimary;
			this->lblDashSession->Location = System::Drawing::Point(15, 45);
			this->lblDashSession->Size = System::Drawing::Size(400, 25);
			this->lblDashSession->Text = L"SESSION: HOME WORKOUT";

			this->lblDashUser->Parent = this->pictureBox1;
			this->lblDashUser->BackColor = Color::Transparent;
			this->lblDashUser->Font = (gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold));
			this->lblDashUser->ForeColor = colTextSecondary;
			this->lblDashUser->Location = System::Drawing::Point(15, 75);
			this->lblDashUser->Size = System::Drawing::Size(400, 25);
			this->lblDashUser->Text = L"USER: ...";

			this->lblDashRep->Parent = this->pictureBox1;
			this->lblDashRep->BackColor = Color::Transparent;
			this->lblDashRep->Font = (gcnew System::Drawing::Font(L"Segoe UI", 28, System::Drawing::FontStyle::Bold));
			this->lblDashRep->ForeColor = colTextPrimary;
			this->lblDashRep->Location = System::Drawing::Point(15, 105);
			this->lblDashRep->Size = System::Drawing::Size(300, 50);
			this->lblDashRep->Text = L"REP 0/0";

			this->lblDashPose->Parent = this->pictureBox1;
			this->lblDashPose->BackColor = Color::Transparent;
			this->lblDashPose->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16, System::Drawing::FontStyle::Bold));
			this->lblDashPose->ForeColor = colTextPrimary;
			this->lblDashPose->Location = System::Drawing::Point(15, 670);
			this->lblDashPose->Size = System::Drawing::Size(600, 30);
			this->lblDashPose->Text = L"POSE: WAITING...";

			this->lblDashBPM->Parent = this->pictureBox1;
			this->lblDashBPM->BackColor = Color::Transparent;
			this->lblDashBPM->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16, System::Drawing::FontStyle::Bold));
			this->lblDashBPM->ForeColor = colWarning;
			this->lblDashBPM->Location = System::Drawing::Point(850, 670);
			this->lblDashBPM->Size = System::Drawing::Size(200, 30);
			this->lblDashBPM->Text = L"\u2665 BPM: 120";
			this->lblDashBPM->TextAlign = System::Drawing::ContentAlignment::MiddleRight;

			this->lblDashTime->Parent = this->pictureBox1;
			this->lblDashTime->BackColor = Color::Transparent;
			this->lblDashTime->Font = (gcnew System::Drawing::Font(L"Segoe UI", 16, System::Drawing::FontStyle::Bold));
			this->lblDashTime->ForeColor = colTextSecondary;
			this->lblDashTime->Location = System::Drawing::Point(1050, 670);
			this->lblDashTime->Size = System::Drawing::Size(200, 30);
			this->lblDashTime->Text = L"TIME: 00:00";
			this->lblDashTime->TextAlign = System::Drawing::ContentAlignment::MiddleRight;

			this->pictureBox1->Controls->Add(this->flowWorkoutBar);
			this->pictureBox1->Controls->Add(this->lblDashSession);
			this->pictureBox1->Controls->Add(this->lblDashUser);
			this->pictureBox1->Controls->Add(this->lblDashRep);
			this->pictureBox1->Controls->Add(this->lblDashPose);
			this->pictureBox1->Controls->Add(this->lblDashBPM);
			this->pictureBox1->Controls->Add(this->lblDashTime);

			// ═══════════════ Timer / Form ═══════════════
			this->toolTip1->AutoPopDelay = 8000;
			this->toolTip1->InitialDelay = 300;
			this->timer1->Interval = 33;
			this->timer1->Tick += gcnew System::EventHandler(this, &MainForm::timer1_Tick);
			this->Shown += gcnew System::EventHandler(this, &MainForm::MainForm_Shown);

			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->BackColor = colBg;
			this->ClientSize = System::Drawing::Size(1280, 980);
			this->Controls->Add(this->panelSidebar);
			this->Controls->Add(this->panelCameraBorder);
			this->Controls->Add(this->panelHeader);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedSingle;
			this->MaximizeBox = false;
			this->Name = L"MainForm";
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterScreen;
			this->Text = L"AI Fitness Trainer";
			this->DoubleBuffered = true;

		// Add screen panels to form (all Fill, start hidden)
		panelHome->Dock       = System::Windows::Forms::DockStyle::Fill;
		panelHome->BackColor  = colBg;
		panelHome->Visible    = false;
		panelPreWorkout->Dock      = System::Windows::Forms::DockStyle::Fill;
		panelPreWorkout->BackColor = colBg;
		panelPreWorkout->Visible   = false;
		panelSummary->Dock      = System::Windows::Forms::DockStyle::Fill;
		panelSummary->BackColor = colBg;
		panelSummary->Visible   = false;
		panelHistory->Dock      = System::Windows::Forms::DockStyle::Fill;
		panelHistory->BackColor = colBg;
		panelHistory->Visible   = false;
		this->Controls->Add(panelHome);
		this->Controls->Add(panelPreWorkout);
		this->Controls->Add(panelSummary);
		this->Controls->Add(panelHistory);

		BuildHomePanel();
		BuildPreWorkoutPanel();
		BuildSummaryPanel();
		BuildHistoryPanel();




			(cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			this->panelAngleBarBg->ResumeLayout(false);
			this->panelSidebar->ResumeLayout(false);
			this->panelSidebar->PerformLayout();
			this->panelCameraBorder->ResumeLayout(false);
			this->panelHeader->ResumeLayout(false);
			this->ResumeLayout(false);
			this->PerformLayout();
		}
#pragma endregion

	// ===== NAVIGATION =====
	void ShowScreen(int id)
	{
		panelHeader->Visible       = (id == 2);
		panelCameraBorder->Visible = (id == 2);
		panelSidebar->Visible      = (id == 2);
		panelHome->Visible       = (id == 0);
		panelPreWorkout->Visible = (id == 1);
		panelSummary->Visible    = (id == 3);
		panelHistory->Visible    = (id == 4);
	}

	void UpdateGuideGif(int exIdx)
	{
		static const wchar_t* gifMap[] = {
			L"biceps-curl.gif", L"squat.gif", L"push-up.gif", L"lunge.gif",
			L"shoulders-press.gif", L"deadlift.gif", L"lateral-raise.gif", L"plank.gif",
			L"tricep pushdown.gif", L"pull-up.gif", L"hip-thrust.gif", L"leg-extension.gif",
			L"bench-press.gif", L"chest flygif.gif", L"decline bench.gif", L"hammer-curls.gif",
			L"incline-bench-press.gif", L"lat pulldown.gif", L"leg raises.gif", L"romanain deadlift.gif",
			L"russain twist.gif", L"t-bar-row.gif", L"tricep dips.gif"
		};

		if (exIdx >= 0 && exIdx < 23) {
			String^ fullPath = L"E:\\ai_fitness_trainer_v2\\gif exercise\\" + gcnew String(gifMap[exIdx]);
			try {
				if (System::IO::File::Exists(fullPath)) {
					if (picBoxGuide->Image != nullptr) {
						System::Drawing::ImageAnimator::StopAnimate(picBoxGuide->Image, nullptr);
						delete picBoxGuide->Image;
					}
					picBoxGuide->Image = Image::FromFile(fullPath);
					
					if (System::Drawing::ImageAnimator::CanAnimate(picBoxGuide->Image)) {
					    System::Drawing::ImageAnimator::Animate(picBoxGuide->Image, nullptr);
					}
					
					picBoxGuide->Visible = true;
				} else {
					picBoxGuide->Visible = false;
				}
			} catch (...) {
				picBoxGuide->Visible = false;
			}
		} else {
			picBoxGuide->Visible = false;
		}
	}

	void HighlightExerciseButton(int exIdx)
	{
		for each (System::Windows::Forms::Control^ ctrl in flowExercisePicker->Controls) {
			Button^ b = dynamic_cast<Button^>(ctrl);
			if (!b) continue;
			int idx = flowExercisePicker->Controls->IndexOf(b);
			b->BackColor = (idx == exIdx) ? colAccent : colCard;
			b->ForeColor = (idx == exIdx) ? colBg     : colTextPrimary;
		}
		preWorkoutExercise = exIdx;
		lblPreRepsHdr->Text = (exIdx == 7)
			? L"Hold (sec) / \u0E27\u0E34\u0E19\u0E32\u0E17\u0E35"
			: L"Reps / \u0E23\u0E35\u0E1E";
	}

	int ReadTodayRepsFromCSV()
	{
		try {
			String^ fp = "workout_history.csv";
			if (!System::IO::File::Exists(fp)) return 0;
			String^ today = DateTime::Now.ToString("yyyy-MM-dd");
			int total = 0;
			auto lines = System::IO::File::ReadAllLines(fp);
			for (int i = 1; i < lines->Length; i++) {
				auto cols = lines[i]->Split(L',');
				if (cols->Length > 10 && cols[0] == today) {
					int r = 0; if (Int32::TryParse(cols[10], r)) total += r;
				}
			}
			return total;
		} catch (...) { return 0; }
	}

	void RefreshHomeScreen()
	{
		String^ nm = gcnew String(profile.name);
		lblHomeGreeting->Text = L"\u0E2A\u0E27\u0E31\u0E2A\u0E14\u0E35, " + nm + L"!";
		int todayReps = ReadTodayRepsFromCSV();
		int goalReps  = profile.RecommendedReps() * profile.RecommendedSets();
		float pct = (goalReps > 0) ? (float)todayReps / goalReps : 0.0f;
		if (pct > 1.0f) pct = 1.0f;
		panelGoalBarFill->Width = (int)(panelGoalBarBg->Width * pct);
		lblGoalProgress->Text = String::Format("{0} / {1} reps today ({2}%)",
			todayReps, goalReps, (int)(pct * 100));
	}


	private:

		// ════════════════════════════════════════════════════════════════════
		//  MODEL LOADING — async on Form_Shown so UI appears immediately
		// ════════════════════════════════════════════════════════════════════
		System::Void MainForm_Shown(System::Object^ sender, System::EventArgs^ e)
		{
			System::Threading::Thread^ loadThread = gcnew System::Threading::Thread(
				gcnew System::Threading::ThreadStart(this, &MainForm::LoadModelThread));
			loadThread->IsBackground = true;
			loadThread->Start();

			// Start at Home screen
			RefreshHomeScreen();
			ShowScreen(0);
		}

		void LoadModelThread()
		{
			bool ok = poseEstimator->loadModel("models/yolov8m-pose.onnx");

			// Try video classifier first (temporal), fall back to image classifier
			if (!exerciseClassifier->loadModel("models/video_classifier.onnx"))
				exerciseClassifier->loadModel("models/exercise_classifier.onnx");

			// Marshal back to UI thread
			this->Invoke(gcnew System::Action<bool>(this, &MainForm::OnModelLoaded), ok);
		}

		void OnModelLoaded(bool ok)
		{
			if (!ok) {
				MessageBox::Show("Failed to load YOLOv8-Pose model.\nEnsure models/yolov8m-pose.onnx exists.",
					"Model Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
				lblGpuStatus->ForeColor = colError;
				lblGpuStatus->Text      = L"\u25CF  Model load failed";
				return;
			}

			modelReady = true;
			btnStart->Enabled = true;

			String^ provider = gcnew String(poseEstimator->getProviderName().c_str());
			if (poseEstimator->isGPUEnabled()) {
				lblGpuStatus->ForeColor = colSuccess;
				lblGpuStatus->Text = L"\u25CF  GPU: " + provider + L" Enabled";
			} else {
				lblGpuStatus->ForeColor = colTextSecondary;
				lblGpuStatus->Text = L"\u25CF  CPU Mode (No GPU)";
			}

			// Update classifier status
			if (exerciseClassifier->isLoaded()) {
				lblAutoExercise->Text = L"\u25BA  Auto-Detect: Ready";
				lblAutoExercise->ForeColor = colSuccess;
			} else {
				lblAutoExercise->Text = L"\u25BA  Auto-Detect: No model (train first)";
				lblAutoExercise->ForeColor = colTextSecondary;
			}
		}

		// ════════════════════════════════════════════════════════════════════
		//  BACKGROUND THREAD 1 — Camera Capture (Producer)
		// ════════════════════════════════════════════════════════════════════
		void CameraLoop()
		{
			while (threadRunning) {
				try {
					cv::Mat frame;
					if (!cameraHandler->readFrame(frame) || frame.empty()) {
						consecutiveReadFails++;
						if (consecutiveReadFails >= MAX_READ_FAILS) {
							{
								std::lock_guard<std::mutex> lock(*frameMutex);
								*sharedCaptureError = "Camera disconnected";
							}
							threadRunning = false;
							frameCondVar->notify_all();
							break;
						}
						System::Threading::Thread::Sleep(2);
						continue;
					}
					consecutiveReadFails = 0;

					if (mirrorEnabled) {
						cv::flip(frame, frame, 1);
					}

					// Push frame to queue and notify consumer
					{
						std::lock_guard<std::mutex> lk(*queueMutex);
						// Limit max queue size to 2 to prevent real-time lag
						while (frameQueue->size() >= 2) frameQueue->pop();
						frameQueue->push(frame.clone());
					}
					frameCondVar->notify_one();
				}
				catch (...) {
					System::Threading::Thread::Sleep(10);
				}
			}
		}

		// ════════════════════════════════════════════════════════════════════
		//  BACKGROUND THREAD 2 — GPU Inference + Drawing (Consumer)
		// ════════════════════════════════════════════════════════════════════
		void InferenceLoop()
		{
			using namespace std::chrono;
			auto   fpsStart  = steady_clock::now();
			int    fpsFrames = 0;
			double fps       = 0.0;

			while (threadRunning) {
				try {
					// ── Handle requests from UI thread ────────────────────────
					if (exerciseChangeRequested) {
						exerciseController->setExerciseType(
							static_cast<ExerciseType>(pendingExerciseIndex));
						exerciseChangeRequested = false;
					}
					if (resetRequested) {
						exerciseController->reset();
						resetRequested = false;
					}

					// ── Pop frame from Queue ──────────────────────────────────
					cv::Mat frame;
					{
						std::unique_lock<std::mutex> lk(*queueMutex);
						while (frameQueue->empty() && threadRunning) {
							frameCondVar->wait(lk);
						}
						if (!threadRunning && frameQueue->empty()) break;
						
						frame = frameQueue->front();
						frameQueue->pop();
					}

					// ── Check Pause State ─────────────────────────────────────
					if (isPaused) {
						// Discard frame to prevent queue overflow, save GPU power, and prevent ghost reps
						continue;
					}

					// ── Performance Optimization: GPU/CPU Downscale ───────────
					// Resize high-res feeds (e.g. 1280x720) down by half (640x360) 
					// early in the pipeline. This reduces CPU drawing load by 4x,
					// speeds up YOLO/ONNX preprocessing, and cuts memory transfer
					// payload for LockBits in MatToBitmap by 4x.
					if (frame.cols >= 1280) {
						cv::resize(frame, frame, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
					}

					// ── GPU / CPU inference ───────────────────────────────────
					auto detections = poseEstimator->runInference(frame);
					if (!detections.empty() && !isResting) {
						exerciseController->update(detections[0].keypoints);

						// ── Injury risk assessment (every frame) ─────────
						InjuryRisk risk = injuryRiskAssessor->assess(
							exerciseController->getCurrentExercise(),
							detections[0].keypoints,
							(float)frame.rows);  // #10: pass frame height for scale

						// ── View angle detection ────────────────────────
						ViewAngle va = BiomechanicsMath::detectViewAngle(
							detections[0].keypoints, (float)frame.cols);

						// -- Pose-based exercise hint (sensor fusion input) --
						PoseHint ph = BiomechanicsMath::poseExerciseHint(
							detections[0].keypoints, (float)frame.rows, (float)frame.cols);

						{
							std::lock_guard<std::mutex> lock(*frameMutex);
							sharedInjuryLevel   = (int)risk.level;
							*sharedInjuryMsg    = risk.message;
							sharedViewAngle     = (int)va;
							sharedPoseHintClass = ph.exerciseTypeInt;
							sharedPoseHintConf  = ph.conf;
						}
					}

					// ── Exercise auto-detection (rate-limited inside classifier) ──
					if (exerciseClassifier->isLoaded()) {
						ClassificationResult cls = exerciseClassifier->classify(frame);
						if (cls.classIndex >= 0) {
							{
								std::lock_guard<std::mutex> lock(*frameMutex);
								*sharedDetectedExercise  = cls.className;
								sharedDetectedConf       = cls.confidence;
								sharedDetectedClassIndex = cls.classIndex;
							}

							// Only auto-switch exercise type at high confidence (>= 0.70)
							if (cls.confidence >= 0.70f) {
								ExerciseType mapped = ExerciseController::mapClassifierToExercise(cls.classIndex);
								if (mapped != exerciseController->getCurrentExercise()) {
									exerciseController->setExerciseType(mapped);
								}
							}
						}
					}

					// ── Skeleton color ────────────────────────────────────────
					std::string fb = exerciseController->getFeedback();
					cv::Scalar skeletonColor;
					if      (fb == "Good Job!"  || fb == "Perfect Form!") skeletonColor = cv::Scalar(0, 255,   0);
					else if (fb == "Adjust Position")                      skeletonColor = cv::Scalar(0,   0, 255);
					else if (fb == "Straighten Body!")                     skeletonColor = cv::Scalar(0, 100, 255);
					else                                                   skeletonColor = cv::Scalar(0, 200, 255);

					poseEstimator->drawPose(frame, detections, skeletonColor);

					// ── Angle arc ─────────────────────────────────────────────
					double angle = exerciseController->getCurrentAngle();
					cv::Point2f p1, p2, p3;
					exerciseController->getActiveKeypoints(p1, p2, p3);
					if (angle > 0 && p2.x > 0 && p2.y > 0) {
						drawAngleArc(frame, p1, p2, p3, skeletonColor);
						std::string angleText = std::to_string((int)angle) + " deg";
						cv::putText(frame, angleText,
							cv::Point((int)p2.x + 20, (int)p2.y - 20),
							cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
					}

					// ── FPS ───────────────────────────────────────────────────
					fpsFrames++;
					auto   now        = steady_clock::now();
					double elapsedSec = duration<double>(now - fpsStart).count();
					if (elapsedSec >= 1.0) {
						fps       = fpsFrames / elapsedSec;
						fpsFrames = 0;
						fpsStart  = now;
						// #8: adapt classifier stride to actual FPS
						if (exerciseClassifier->isLoaded() && exerciseClassifier->isVideoMode()) {
							exerciseClassifier->setActualFps(fps);
						}
					}
					cv::putText(frame, "FPS: " + std::to_string((int)fps),
						cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

					// ── Publish shared state ──────────────────────────────────
					{
						std::lock_guard<std::mutex> lock(*frameMutex);
						*sharedFrame    = std::move(frame);
						hasNewFrame     = true;
						sharedRepCount  = exerciseController->getRepCount();
						*sharedFeedback = fb;
						*sharedQuality  = exerciseController->getLastRepQuality();
						sharedAngle     = angle;
						sharedIsHold    = exerciseController->isHoldExercise();
						sharedFps       = fps;
						sharedFull      = exerciseController->getFullReps();
						sharedPartial   = exerciseController->getPartialReps();
						sharedPoor      = exerciseController->getPoorReps();
					}

				} catch (const Ort::Exception& e) {
					std::lock_guard<std::mutex> lock(*frameMutex);
					*sharedCaptureError = std::string("ONNX error: ") + e.what();
					threadRunning = false;
					break;
				} catch (const cv::Exception& e) {
					std::lock_guard<std::mutex> lock(*frameMutex);
					*sharedCaptureError = std::string("OpenCV error: ") + e.what();
					threadRunning = false;
					break;
				} catch (const std::exception& e) {
					std::lock_guard<std::mutex> lock(*frameMutex);
					*sharedCaptureError = std::string("Error: ") + e.what();
					threadRunning = false;
					break;
				} catch (...) {
					std::lock_guard<std::mutex> lock(*frameMutex);
					*sharedCaptureError = "Unknown error in capture thread";
					threadRunning = false;
					break;
				}
			}
		}

		// ════════════════════════════════════════════════════════════════════
		//  UI TIMER TICK — read shared state + update controls + sets logic
		// ════════════════════════════════════════════════════════════════════
		System::Void timer1_Tick(System::Object^ sender, System::EventArgs^ e)
		{
			// ── Snapshot ──────────────────────────────────────────────────
			cv::Mat     frame;
			int         repCount = 0;
			std::string feedback, quality;
			double      angle  = 0.0;
			bool        isHold = false;
			double      fps    = 0.0;
			int         full = 0, partial = 0, poor = 0;
			std::string detectedExercise;
			float       detectedConf     = 0.0f;
			int         detectedClassIdx = -1;
			int         injuryLevel      = 0;
			std::string injuryMsg;
			int         viewAngle        = 0;
			int         poseHintClass    = -1;
			float       poseHintConf     = 0.0f;

			std::string captureError;

			{
				std::lock_guard<std::mutex> lock(*frameMutex);
				if (hasNewFrame) {
					frame       = std::move(*sharedFrame);
					hasNewFrame = false;
					repCount    = sharedRepCount;
					feedback    = *sharedFeedback;
					quality     = *sharedQuality;
					angle       = sharedAngle;
					isHold      = sharedIsHold;
					fps         = sharedFps;
					full        = sharedFull;
					partial     = sharedPartial;
					poor        = sharedPoor;
					detectedExercise = *sharedDetectedExercise;
					detectedConf     = sharedDetectedConf;
					detectedClassIdx = sharedDetectedClassIndex;
					injuryLevel      = sharedInjuryLevel;
					injuryMsg        = *sharedInjuryMsg;
					viewAngle        = sharedViewAngle;
					poseHintClass    = sharedPoseHintClass;
					poseHintConf     = sharedPoseHintConf;
				}
				captureError = *sharedCaptureError;
				
				// Wait for inference thread to acknowledge reset before reading reps
				if (resetRequested) {
					repCount = 0;
					angle = 0;
				}
			}

			// ── Check if capture thread crashed ──────────────────────────
			if (!captureError.empty() && isRunning) {
				isRunning = false;
				timer1->Stop();
				cameraHandler->releaseCamera();
				btnStart->Enabled        = true;
				btnStop->Enabled         = false;
				txtCameraSource->Enabled = true;
				txtWeight->Enabled       = true;
				lblStatus->ForeColor     = colError;
				lblStatus->Text          = L"\u25CF  Error: Thread stopped";
				MessageBox::Show(gcnew String(captureError.c_str()),
					"Capture Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
				return;
			}

			if (frame.empty()) return;

			// Display full frame — StretchImage on pictureBox handles landscape fill
			cv::Mat displayFrame = frame.clone();

			currentFps = fps;

			// ── Rest timer check ──────────────────────────────────────────
			if (isResting) {
				TimeSpan restElapsed = DateTime::Now - restStartTime;
				int remaining = restDurationSec - (int)restElapsed.TotalSeconds;
				if (remaining <= 0) {
					isResting = false;
					resetRequested = true;
					lastKnownRepCount = 0;
					sessionStartTime = DateTime::Now;
					lblRepCount->ForeColor = colAccent;
					
					if (allSetsComplete && (currentProgramIndex >= 0 || isCustomProgram)) {
						// Advance to next exercise in program
						int currentStepIdx = -1;
						int progStepsCount = isCustomProgram ? (int)customSteps->size() : PROGRAMS[currentProgramIndex].stepCount;
						
						for (int i = 0; i < progStepsCount; i++) {
							int exType = isCustomProgram ? (*customSteps)[i].exerciseType : PROGRAMS[currentProgramIndex].steps[i].exerciseType;
							if (exType == preWorkoutExercise) {
								currentStepIdx = i; break;
							}
						}

						if (currentStepIdx >= 0 && currentStepIdx < progStepsCount - 1) {
							// Setup next exercise
							allSetsComplete = false;
							currentSet = 1;
							int nextEx = isCustomProgram ? (*customSteps)[currentStepIdx + 1].exerciseType : PROGRAMS[currentProgramIndex].steps[currentStepIdx + 1].exerciseType;
							comboBoxExercise->SelectedIndex = nextEx; // triggers change
							preWorkoutExercise = nextEx;
							targetSets = isCustomProgram ? (*customSteps)[currentStepIdx + 1].sets : PROGRAMS[currentProgramIndex].steps[currentStepIdx + 1].sets;
							targetReps = isCustomProgram ? (*customSteps)[currentStepIdx + 1].reps : PROGRAMS[currentProgramIndex].steps[currentStepIdx + 1].reps;
							lblSet->Text = L"SET 1 / " + targetSets.ToString();
							lblRepsLabel->Text = (nextEx == 7) ? L"SECONDS" : L"REPS";
							
							if (currentStepIdx + 2 < progStepsCount) {
								int upcomingEx = isCustomProgram ? (*customSteps)[currentStepIdx + 2].exerciseType : PROGRAMS[currentProgramIndex].steps[currentStepIdx + 2].exerciseType;
								lblNextExercise->Text = L"Next: " + comboBoxExercise->Items[upcomingEx]->ToString();
							} else {
								lblNextExercise->Text = L"Next: Finish";
							}
							
							// Update FLOW bar HIGHLIGHT
							for (int i = 0; i < flowWorkoutBar->Controls->Count; i++) {
								Label^ lbl = dynamic_cast<Label^>(flowWorkoutBar->Controls[i]);
								if (lbl != nullptr && lbl->Text->StartsWith(L" [")) {
									int stepEquivalent = (i - 1) / 2;
									if (stepEquivalent == currentStepIdx + 1) {
										lbl->BackColor = colAccent;
										lbl->ForeColor = Color::White;
									} else {
										lbl->BackColor = Color::Transparent;
										lbl->ForeColor = colTextSecondary;
									}
								}
							}
							
							return;
						} else {
							// Program finished
							EndWorkoutAndShowSummary();
							return;
						}
					} else {
						// Normal next set
						lblRepsLabel->Text = isHold ? L"SECONDS" : L"REPS";
						lblSet->Text = L"SET " + currentSet.ToString() + L" / " + targetSets.ToString();
					}
				} else {
					// Show rest countdown
					lblRepCount->Text = remaining.ToString();
					lblRepCount->ForeColor = colRest;
					lblRepsLabel->Text = L"REST";
					if (allSetsComplete) {
						lblFeedback->Text = L"Exercise Complete! Next in " + remaining.ToString() + L"s";
					} else {
						lblFeedback->Text = L"Set Complete! Next set in " + remaining.ToString() + L"s";
					}
					lblFeedback->ForeColor = colRest;
					// Still update the frame display
					Bitmap^ bmp = MatToBitmap(displayFrame);
					if (pictureBox1->Image != nullptr) delete pictureBox1->Image;
					pictureBox1->Image = bmp;
					return;
				}
			}

			// ── Sets completion check ─────────────────────────────────────
			if (!allSetsComplete && !isHold && repCount >= targetReps && repCount > lastKnownRepCount) {
				
				// Accumulate stats for this set before resetting
				sessionTotalReps   += exerciseController->getRepCount();
				sessionFullReps    += exerciseController->getFullReps();
				sessionPartialReps += exerciseController->getPartialReps();
				sessionPoorReps    += exerciseController->getPoorReps();
				
				if (currentSet < targetSets) {
					// Start rest between sets
					currentSet++;
					isResting = true;
					restStartTime = DateTime::Now;
					System::Media::SystemSounds::Exclamation->Play();
					lastKnownRepCount = repCount;
					
					// Force a reset *now* so the UI doesn't evaluate completion again next tick
					resetRequested = true;
					
					// Display rest state
					Bitmap^ bmp = MatToBitmap(displayFrame);
					if (pictureBox1->Image != nullptr) delete pictureBox1->Image;
					pictureBox1->Image = bmp;
					return;
					} else {
					allSetsComplete = true;
					System::Media::SystemSounds::Exclamation->Play();
					lblFeedback->Text = L"All Sets Complete!";
					lblFeedback->ForeColor = colSuccess;
					
					// Force a reset *now* so inference thread stops evaluating this exercise
					resetRequested = true;
					
					if (currentProgramIndex >= 0 || isCustomProgram) {
						isResting = true;
						restStartTime = DateTime::Now;
					}
				}
			}

			// ── Pause check ───────────────────────────────────────────────
			if (isPaused) {
				lblFeedback->Text = L"Workout Paused";
				lblFeedback->ForeColor = colWarning;
				Bitmap^ bmp = MatToBitmap(displayFrame);
				if (pictureBox1->Image != nullptr) delete pictureBox1->Image;
				pictureBox1->Image = bmp;
				return;
			}

			// ── Sound on new rep ──────────────────────────────────────────
			if (repCount > lastKnownRepCount) {
				if (!isHold)
					System::Media::SystemSounds::Asterisk->Play();
				lastKnownRepCount = repCount;
			}

			// ── Display frame ─────────────────────────────────────────────
			if (picBoxGuide->Image != nullptr) {
			    System::Drawing::ImageAnimator::UpdateFrames(picBoxGuide->Image);
			}
			
			Bitmap^ bmp = MatToBitmap(displayFrame);
			if (pictureBox1->Image != nullptr) delete pictureBox1->Image;
			pictureBox1->Image = bmp; // This triggers pictureBox1_Paint automatically for the GIF overlay

			// ── Rep count ─────────────────────────────────────────────────
			lblRepCount->Text = repCount.ToString();
			lblDashRep->Text = L"REP " + repCount.ToString() + L"/" + targetReps.ToString();

			// ── Rep progress bar ──────────────────────────────────────────
			if (targetReps > 0 && !isHold) {
				int prog = Math::Min(repCount, targetReps);
				int barW = (int)((float)prog / targetReps * 130.0f);
				panelRepProgressFill->Size = System::Drawing::Size(barW, 8);
				lblRepProgress->Text = prog.ToString() + L" / " + targetReps.ToString();
				if (prog >= targetReps)
					panelRepProgressFill->BackColor = colSuccess;
				else
					panelRepProgressFill->BackColor = colAccent;
			}

			// ── Feedback label ────────────────────────────────────────────
			if (!allSetsComplete) {
				String^ feedbackStr = gcnew String(feedback.c_str());
				String^ qualityStr  = gcnew String(quality.c_str());
				if (isHold) {
					lblFeedback->Text = feedbackStr;
				} else if (qualityStr->Length > 0 && feedbackStr == "Good Job!") {
					lblFeedback->Text = feedbackStr + " (" + qualityStr + ")";
				} else {
					lblFeedback->Text = feedbackStr;
				}

				if      (feedbackStr == "Good Job!" || feedbackStr == "Perfect Form!" || feedbackStr == "Hold Steady!") {
					lblFeedback->ForeColor = colSuccess;
					lblDashPose->ForeColor = colSuccess;
				} else if (feedbackStr == "Adjust Position" || feedbackStr == "Straighten Body!") {
					lblFeedback->ForeColor = colError;
					lblDashPose->ForeColor = colError;
				} else if (feedbackStr == "Ready") {
					lblFeedback->ForeColor = colTextSecondary;
					lblDashPose->ForeColor = colTextSecondary;
				} else {
					lblFeedback->ForeColor = colWarning;
					lblDashPose->ForeColor = colWarning;
				}
				
				// Update Dashboard
				if (qualityStr->Length > 0 && feedbackStr == "Good Job!") {
					lblDashPose->Text = L"POSE: " + feedbackStr->ToUpper() + L" (" + qualityStr->ToUpper() + L")";
				} else {
					lblDashPose->Text = L"POSE: " + feedbackStr->ToUpper();
				}
			}

			// ── Quality badges ────────────────────────────────────────────
			lblQualityFull->Text    = L"\u25CF " + full.ToString()    + L" Full";
			lblQualityPartial->Text = L"\u25CF " + partial.ToString() + L" Partial";
			lblQualityPoor->Text    = L"\u25CF " + poor.ToString()    + L" Poor";

			// ── Timer ─────────────────────────────────────────────────────
			TimeSpan workoutElapsed = DateTime::Now - sessionStartTime;
			lblTimer->Text = String::Format("{0:D2}:{1:D2}",
				(int)workoutElapsed.TotalMinutes, workoutElapsed.Seconds);
			lblDashTime->Text = L"TIME: " + lblTimer->Text;
			

			// ── Angle bar ─────────────────────────────────────────────────
			if (angle > 0) {
				int barWidth = (int)((180.0 - angle) / 180.0 * 280.0);
				barWidth = Math::Max(0, Math::Min(280, barWidth));
				panelAngleBarFill->Size = System::Drawing::Size(barWidth, 14);
				lblAngleValue->Text = ((int)angle).ToString() + L"\x00B0";
				if      (angle < 60)  panelAngleBarFill->BackColor = colSuccess;
				else if (angle < 120) panelAngleBarFill->BackColor = colAccent;
				else                  panelAngleBarFill->BackColor = colWarning;
			}

			// ── Calories ──────────────────────────────────────────────────
			{
				int   exIdx = preWorkoutExercise;  // set from Pre-Workout screen
				float elSec = (float)(DateTime::Now - sessionStartTime).TotalSeconds;
				sessionCalories   = profile.EstimateCalories(exIdx, elSec);
				lblCalories->Text = L"\u25B2  " + sessionCalories.ToString("F1") + L" kcal burned";
			}

			// -- Auto-detected exercise (fused: video classifier + pose hint) -------
			{
				static const char* poseTypeNames[] = {
					"bicep curl","squat","push-up","lunge",
					"shoulder press","deadlift","lateral raise","plank",
					"tricep pushdown","pull-up","hip thrust","leg extension"
				};
				bool hasVideo = !detectedExercise.empty() && detectedConf >= VIDEO_CONFIDENCE_THRESHOLD;
				bool hasPose  = (poseHintClass >= 0 && poseHintClass < 12) && poseHintConf >= POSE_CONFIDENCE_THRESHOLD;

				if (hasVideo || hasPose) {
					String^ exName;
					float   showConf;
					Color   autoCol;
					String^ srcTag = L"";

					if (hasVideo && hasPose && detectedClassIdx >= 0) {
						ExerciseType vidMapped = ExerciseController::mapClassifierToExercise(detectedClassIdx);
						bool agree = ((int)vidMapped == poseHintClass);
						if (agree) {
							// Both sources agree -- fuse
							{ float _b = detectedConf + poseHintConf * FUSION_BOOST_FACTOR; showConf = _b < 1.0f ? _b : 1.0f; }
							exName   = gcnew String(detectedExercise.c_str());
							srcTag   = L" \u2713";   // check mark = fused
						} else {
							// Disagreement -- keep video, slight penalty
							showConf = detectedConf * DISAGREEMENT_PENALTY;
							exName   = gcnew String(detectedExercise.c_str());
							srcTag   = L" ?";
						}
					} else if (hasVideo) {
						showConf = detectedConf;
						exName   = gcnew String(detectedExercise.c_str());
					} else {
						// Only pose hint available
						showConf = poseHintConf;
						exName   = gcnew String(poseTypeNames[poseHintClass]);
						srcTag   = L" [pose]";
					}

					int confPct = (int)(showConf * 100);
					if      (showConf >= 0.70f) autoCol = colSuccess;
					else if (showConf >= 0.50f) autoCol = colAccent;
					else                        autoCol = colTextSecondary;

					lblAutoExercise->Text      = L"\u25BA  Auto: " + exName +
					                             L" (" + confPct.ToString() + L"%)" + srcTag;
					lblAutoExercise->ForeColor = autoCol;
				}
			}

			// ── Injury risk display ──────────────────────────────────────
			{
				String^ riskMsg = gcnew String(injuryMsg.c_str());
				if (injuryLevel == 0) {
					lblInjuryRisk->Text      = L"\u2714  Form: " + riskMsg;
					lblInjuryRisk->ForeColor = colSuccess;
				} else if (injuryLevel == 1) {
					lblInjuryRisk->Text      = L"\u26A0  " + riskMsg;
					lblInjuryRisk->ForeColor = colWarning;
				} else {
					lblInjuryRisk->Text      = L"\u26D4  " + riskMsg;
					lblInjuryRisk->ForeColor = colError;
				}
			}

			// ── View angle display ─────────────────────────────────────────────
			{
				int exIdx = (int)exerciseController->getCurrentExercise();
				const char* recView = BiomechanicsMath::recommendedViewLabel(exIdx);
				String^ viewTxt;
				Color   viewCol;
				switch (viewAngle) {
					case 1: viewTxt = L"\u25C9  View: Front";      viewCol = colSuccess;       break;
					case 2: viewTxt = L"\u25C9  View: Left Side";  viewCol = colAccent;        break;
					case 3: viewTxt = L"\u25C9  View: Right Side"; viewCol = colAccent;        break;
					default:viewTxt = L"\u25C9  View: Detecting...";viewCol = colTextSecondary; break;
				}
				// Show tip if wrong view for this exercise
				if (recView && recView[0] && viewAngle > 0) {
					String^ rec = gcnew String(recView);
					bool isFront = (viewAngle == 1);
					bool needFront = (rec == "Front");
					bool needSide  = (rec == "Side");
					if ((needFront && !isFront) || (needSide && isFront)) {
						viewTxt = viewTxt + L" (try " + rec + L" view)";
						viewCol = colWarning;
					}
				}
				lblViewAngle->Text      = viewTxt;
				lblViewAngle->ForeColor = viewCol;
			}
		}

		// ════════════════════════════════════════════════════════════════════
		//  BUTTON HANDLERS
		// ════════════════════════════════════════════════════════════════════

		System::Void btnStart_Click(System::Object^ sender, System::EventArgs^ e) {
			String^ sourceText = txtCameraSource->Text->Trim();
			bool cameraOpened = false;

			if (sourceText->StartsWith("http") || sourceText->StartsWith("rtsp")) {
				std::string url = msclr::interop::marshal_as<std::string>(sourceText);
				cameraOpened = cameraHandler->initCamera(url);
			} else {
				int cameraIndex = 0;
				Int32::TryParse(sourceText, cameraIndex);
				cameraOpened = cameraHandler->initCamera(cameraIndex);
			}

			if (cameraOpened) {
				isRunning      = true;
				threadRunning = true;
				resetRequested = false;
				exerciseChangeRequested = false;
				hasNewFrame    = false;
				isResting = false;
				allSetsComplete = false;
				consecutiveReadFails = 0;
				*sharedCaptureError = "";

				// Sets configuration from profile
				targetReps = profile.RecommendedReps();
				targetSets = profile.RecommendedSets();
				restDurationSec = profile.RecommendedRestSec();
				currentSet = 1;

				lblSet->Text = L"SET 1 / " + targetSets.ToString();
				lblRepProgress->Text = L"0 / " + targetReps.ToString();
				panelRepProgressFill->Size = System::Drawing::Size(0, 8);

				cameraThread = gcnew System::Threading::Thread(
					gcnew System::Threading::ThreadStart(this, &MainForm::CameraLoop));
				cameraThread->IsBackground = true;
				cameraThread->Name = L"CameraThread";
				cameraThread->Start();

				inferenceThread = gcnew System::Threading::Thread(
					gcnew System::Threading::ThreadStart(this, &MainForm::InferenceLoop));
				inferenceThread->IsBackground = true;
				inferenceThread->Name = L"InferenceThread";
				inferenceThread->Start();

				timer1->Start();

				btnStart->Enabled        = false;
				btnStop->Enabled         = true;
				btnPause->Enabled        = true;
				txtCameraSource->Enabled = false;
				txtWeight->Enabled       = false;

				sessionStartTime  = DateTime::Now;
				lastKnownRepCount = exerciseController->getRepCount();
				sessionCalories   = 0.0f;
				lblCalories->Text = L"\u25B2  0.0 kcal burned";

				lblStatus->ForeColor = colSuccess;
				lblStatus->Text = L"\u25CF  Camera: Connected";
			} else {
				MessageBox::Show(
					"Could not open camera.\n\n"
					"Local camera: enter index (0, 1, 2...)\n"
					"iPad/IP camera: enter URL (http/rtsp)",
					"Camera Error", MessageBoxButtons::OK, MessageBoxIcon::Warning);
			}
		}

		System::Void btnStop_Click(System::Object^ sender, System::EventArgs^ e) {
			EndWorkoutAndShowSummary();
		}

		System::Void btnReset_Click(System::Object^ sender, System::EventArgs^ e) {
			if (isRunning) {
				resetRequested = true;
			} else {
				exerciseController->reset();
			}
			isResting = false;
			isPaused = false;
			btnPause->Text = L"\u23F8  Pause";
			btnPause->BackColor = colWarning;
			allSetsComplete = false;
			currentSet      = 1;
			lastKnownRepCount = 0;
			lblRepCount->Text      = L"0";
			lblRepCount->ForeColor = colAccent;
			lblFeedback->Text      = L"Ready";
			lblFeedback->ForeColor = colTextSecondary;
			lblAngleValue->Text    = L"--\x00B0";
			lblTimer->Text         = L"00:00";
			lblRepsLabel->Text     = L"REPS";
			panelAngleBarFill->Size     = System::Drawing::Size(0, 14);
			panelRepProgressFill->Size  = System::Drawing::Size(0, 8);
			lblSet->Text        = L"SET 1 / " + targetSets.ToString();
			lblRepProgress->Text = L"0 / " + targetReps.ToString();
			lblQualityFull->Text    = L"\u25CF 0 Full";
			lblQualityPartial->Text = L"\u25CF 0 Partial";
			lblQualityPoor->Text    = L"\u25CF 0 Poor";
			if (isRunning) sessionStartTime = DateTime::Now;
		}

		System::Void btnPause_Click(System::Object^ sender, System::EventArgs^ e) {
			if (!isRunning) return;
			isPaused = !isPaused;
			if (isPaused) {
				btnPause->Text = L"\u25B6  Resume";
				btnPause->BackColor = colSuccess;
				lblStatus->Text = L"\u23F8  Workout Paused";
				lblStatus->ForeColor = colWarning;
				timer1->Stop(); // Stop UI timer
			} else {
				btnPause->Text = L"\u23F8  Pause";
				btnPause->BackColor = colWarning;
				lblStatus->Text = L"\u25CF  Camera: Active";
				lblStatus->ForeColor = colSuccess;
				timer1->Start(); // Resume UI timer
			}
		}

		System::Void comboBoxExercise_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e) {
			int index = comboBoxExercise->SelectedIndex;

			// Thread-safe: tell background thread to change exercise
			if (isRunning) {
				pendingExerciseIndex = index;
				exerciseChangeRequested = true;
			} else {
				exerciseController->setExerciseType(static_cast<ExerciseType>(index));
			}

			lastKnownRepCount = 0;
			currentSet = 1;
			allSetsComplete = false;
			isResting = false;
			lblRepCount->Text      = L"0";
			lblRepCount->ForeColor = colAccent;
			lblFeedback->Text      = L"Ready";
			lblFeedback->ForeColor = colTextSecondary;
			lblAngleValue->Text    = L"--\x00B0";
			panelAngleBarFill->Size    = System::Drawing::Size(0, 14);
			panelRepProgressFill->Size = System::Drawing::Size(0, 8);
			lblSet->Text        = L"SET 1 / " + targetSets.ToString();
			lblRepProgress->Text = L"0 / " + targetReps.ToString();

			if (static_cast<ExerciseType>(index) == ExerciseType::Plank) {
				lblRepsLabel->Text = L"SECONDS";
			} else {
				lblRepsLabel->Text = L"REPS";
			}
		}

		System::Void btnMirror_Click(System::Object^ sender, System::EventArgs^ e) {
			mirrorEnabled = !mirrorEnabled;
			if (mirrorEnabled) {
				btnMirror->BackColor = colAccent;
				btnMirror->ForeColor = colBg;
				btnMirror->Text = L"\u2194  Mirror: ON";
			} else {
				btnMirror->BackColor = colSidebarBorder;
				btnMirror->ForeColor = colTextPrimary;
				btnMirror->Text = L"\u2194  Mirror: OFF";
			}
		}

	private:


	// ===== BUILD HOME PANEL =====
	void BuildHomePanel()
	{
		using namespace System::Drawing;
		System::Drawing::Font^ f24b = gcnew System::Drawing::Font(L"Segoe UI", 24, FontStyle::Bold);
		System::Drawing::Font^ f16b = gcnew System::Drawing::Font(L"Segoe UI", 16, FontStyle::Bold);
		System::Drawing::Font^ f12b = gcnew System::Drawing::Font(L"Segoe UI", 12, FontStyle::Bold);
		System::Drawing::Font^ f11  = gcnew System::Drawing::Font(L"Segoe UI", 11);
		System::Drawing::Font^ f10b = gcnew System::Drawing::Font(L"Segoe UI", 10, FontStyle::Bold);
		System::Drawing::Font^ f9   = gcnew System::Drawing::Font(L"Segoe UI", 9);
		System::Drawing::Font^ f9b  = gcnew System::Drawing::Font(L"Segoe UI", 9, FontStyle::Bold);

		// Greeting
		lblHomeGreeting->AutoSize  = false;
		lblHomeGreeting->Font      = f24b;
		lblHomeGreeting->ForeColor = colTextPrimary;
		lblHomeGreeting->Location  = System::Drawing::Point(40, 30);
		lblHomeGreeting->Size      = System::Drawing::Size(800, 40);
		lblHomeGreeting->Text      = L"Hello!";

		// Daily Goal header
		lblHomeDailyGoalHdr->AutoSize  = false;
		lblHomeDailyGoalHdr->Font      = f10b;
		lblHomeDailyGoalHdr->ForeColor = colAccent;
		lblHomeDailyGoalHdr->Location  = System::Drawing::Point(40, 82);
		lblHomeDailyGoalHdr->Size      = System::Drawing::Size(400, 20);
		lblHomeDailyGoalHdr->Text      = L"TODAY'S GOAL / \u0E40\u0E1B\u0E49\u0E32\u0E2B\u0E21\u0E32\u0E22\u0E27\u0E31\u0E19\u0E19\u0E35\u0E49";

		// Goal progress bar
		panelGoalBarBg->BackColor = colBarBg;
		panelGoalBarBg->Location  = System::Drawing::Point(40, 108);
		panelGoalBarBg->Size      = System::Drawing::Size(600, 16);
		panelGoalBarFill->BackColor = colAccent;
		panelGoalBarFill->Location  = System::Drawing::Point(0, 0);
		panelGoalBarFill->Size      = System::Drawing::Size(0, 16);
		panelGoalBarBg->Controls->Add(panelGoalBarFill);

		lblGoalProgress->AutoSize  = true;
		lblGoalProgress->Font      = f9;
		lblGoalProgress->ForeColor = colTextSecondary;
		lblGoalProgress->Location  = System::Drawing::Point(650, 110);
		lblGoalProgress->Text      = L"0 / 0 reps";

		// Action buttons row
		btnQuickStart->BackColor = colBtnStart;
		btnQuickStart->FlatStyle = FlatStyle::Flat;
		btnQuickStart->FlatAppearance->BorderSize = 0;
		btnQuickStart->Font      = gcnew System::Drawing::Font(L"Segoe UI", 12, FontStyle::Bold);
		btnQuickStart->ForeColor = colTextPrimary;
		btnQuickStart->Location  = System::Drawing::Point(40, 142);
		btnQuickStart->Size      = System::Drawing::Size(220, 50);
		btnQuickStart->Text      = L"\u25B6  Quick Start";
		btnQuickStart->Cursor    = Cursors::Hand;
		btnQuickStart->UseVisualStyleBackColor = false;
		btnQuickStart->Click += gcnew System::EventHandler(this, &MainForm::btnQuickStart_Click);

		btnViewHistory->BackColor = colSidebarBorder;
		btnViewHistory->FlatStyle = FlatStyle::Flat;
		btnViewHistory->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
		btnViewHistory->FlatAppearance->BorderSize = 0;
		btnViewHistory->Font      = gcnew System::Drawing::Font(L"Segoe UI", 11, FontStyle::Bold);
		btnViewHistory->ForeColor = colTextPrimary;
		btnViewHistory->Location  = System::Drawing::Point(276, 142);
		btnViewHistory->Size      = System::Drawing::Size(200, 50);
		btnViewHistory->Text      = L"History / \x0E1B\x0E23\x0E30\x0E27\x0E31\x0E15\x0E34";
		btnViewHistory->Cursor    = System::Windows::Forms::Cursors::Hand;
		btnViewHistory->UseVisualStyleBackColor = false;
		btnViewHistory->Click += gcnew System::EventHandler(this, &MainForm::btnViewHistory_Click);

		// Programs header
		lblProgramsHdr->AutoSize  = false;
		lblProgramsHdr->Font      = gcnew System::Drawing::Font(L"Segoe UI", 10, FontStyle::Bold);
		lblProgramsHdr->ForeColor = colAccent;
		lblProgramsHdr->Location  = System::Drawing::Point(40, 212);
		lblProgramsHdr->Size      = System::Drawing::Size(700, 22);
		lblProgramsHdr->Text      = L"WORKOUT PROGRAMS / \x0E42\x0E1B\x0E23\x0E41\x0E01\x0E23\x0E21\x0E2D\x0E2D\x0E01\x0E01\x0E33\x0E25\x0E31\x0E07\x0E01\x0E32\x0E22";

		// Program card buttons (2 rows: 3 + 2)
		cli::array<System::Windows::Forms::Button^>^ progBtns = gcnew cli::array<System::Windows::Forms::Button^>(5) {
			btnProgram0, btnProgram1, btnProgram2, btnProgram3, btnProgram4
		};
		cli::array<System::EventHandler^>^ progHandlers = gcnew cli::array<System::EventHandler^>(5) {
			gcnew System::EventHandler(this, &MainForm::btnProgram0_Click),
			gcnew System::EventHandler(this, &MainForm::btnProgram1_Click),
			gcnew System::EventHandler(this, &MainForm::btnProgram2_Click),
			gcnew System::EventHandler(this, &MainForm::btnProgram3_Click),
			gcnew System::EventHandler(this, &MainForm::btnProgram4_Click)
		};
		int cardW = 200, cardH = 90, gap = 16, startX = 40, startY = 242;
		for (int i = 0; i < 5; i++) {
			int row = i / 3, col = i % 3;
			progBtns[i]->BackColor = colCard;
			progBtns[i]->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			progBtns[i]->FlatAppearance->BorderColor = colSidebarBorder;
			progBtns[i]->FlatAppearance->BorderSize  = 1;
			progBtns[i]->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(40, 54, 80);
			progBtns[i]->Font      = gcnew System::Drawing::Font(L"Segoe UI", 9, FontStyle::Bold);
			progBtns[i]->ForeColor = colTextPrimary;
			progBtns[i]->Location  = System::Drawing::Point(startX + col*(cardW+gap), startY + row*(cardH+gap));
			progBtns[i]->Size      = System::Drawing::Size(cardW, cardH);
			progBtns[i]->TextAlign = System::Drawing::ContentAlignment::MiddleCenter;
			progBtns[i]->UseVisualStyleBackColor = false;
			progBtns[i]->Cursor    = System::Windows::Forms::Cursors::Hand;
			// Build card text: EN name + TH name + desc
			System::String^ nameEn = gcnew System::String(PROGRAMS[i].nameEn);
			System::String^ nameTh = gcnew System::String(PROGRAMS[i].nameTh);
			System::String^ desc   = gcnew System::String(PROGRAMS[i].desc);
			progBtns[i]->Text = nameEn + L"\n" + nameTh + L"\n" + desc;
			progBtns[i]->Click += progHandlers[i];
			panelHome->Controls->Add(progBtns[i]);
		}

		// Add all home controls
		panelHome->Controls->Add(lblHomeGreeting);
		panelHome->Controls->Add(lblHomeDailyGoalHdr);
		panelHome->Controls->Add(panelGoalBarBg);
		panelHome->Controls->Add(lblGoalProgress);
		panelHome->Controls->Add(btnQuickStart);
		panelHome->Controls->Add(btnViewHistory);
		panelHome->Controls->Add(lblProgramsHdr);
	}

	// ===== BUILD PRE-WORKOUT PANEL =====
	void BuildPreWorkoutPanel()
	{
		using namespace System::Drawing;
		System::Drawing::Font^ f18b = gcnew System::Drawing::Font(L"Segoe UI", 18, FontStyle::Bold);
		System::Drawing::Font^ f10b = gcnew System::Drawing::Font(L"Segoe UI", 10, FontStyle::Bold);
		System::Drawing::Font^ f10  = gcnew System::Drawing::Font(L"Segoe UI", 10);
		System::Drawing::Font^ f9   = gcnew System::Drawing::Font(L"Segoe UI", 9);
		System::Drawing::Font^ f9i  = gcnew System::Drawing::Font(L"Segoe UI", 9, FontStyle::Italic);
		System::Drawing::Font^ f12b = gcnew System::Drawing::Font(L"Segoe UI", 12, FontStyle::Bold);

		// Back button
		btnBackPreWorkout->BackColor = colSidebarBorder;
		btnBackPreWorkout->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
		btnBackPreWorkout->FlatAppearance->BorderSize = 0;
		btnBackPreWorkout->Font      = f10b;
		btnBackPreWorkout->ForeColor = colTextPrimary;
		btnBackPreWorkout->Location  = System::Drawing::Point(20, 16);
		btnBackPreWorkout->Size      = System::Drawing::Size(100, 32);
		btnBackPreWorkout->Text      = L"\x2190  Back";
		btnBackPreWorkout->Cursor    = System::Windows::Forms::Cursors::Hand;
		btnBackPreWorkout->UseVisualStyleBackColor = false;
		btnBackPreWorkout->Click += gcnew System::EventHandler(this, &MainForm::btnBackPreWorkout_Click);

		// Title
		lblPreWorkoutTitle->AutoSize  = false;
		lblPreWorkoutTitle->Font      = f18b;
		lblPreWorkoutTitle->ForeColor = colTextPrimary;
		lblPreWorkoutTitle->Location  = System::Drawing::Point(40, 60);
		lblPreWorkoutTitle->Size      = System::Drawing::Size(700, 34);
		lblPreWorkoutTitle->Text      = L"Setup Workout / \x0E15\x0E31\x0E49\x0E07\x0E04\x0E48\x0E32\x0E01\x0E32\x0E23\x0E2D\x0E2D\x0E01\x0E01\x0E33\x0E25\x0E31\x0E07\x0E01\x0E32\x0E22";

		// Selected program label
		lblSelectedProgram->AutoSize  = false;
		lblSelectedProgram->Font      = f9i;
		lblSelectedProgram->ForeColor = colTextSecondary;
		lblSelectedProgram->Location  = System::Drawing::Point(40, 100);
		lblSelectedProgram->Size      = System::Drawing::Size(700, 20);
		lblSelectedProgram->Text      = L"Quick Start";

		// Exercise picker header
		lblExercisePickHdr->AutoSize  = false;
		lblExercisePickHdr->Font      = f10b;
		lblExercisePickHdr->ForeColor = colAccent;
		lblExercisePickHdr->Location  = System::Drawing::Point(40, 130);
		lblExercisePickHdr->Size      = System::Drawing::Size(700, 22);
		lblExercisePickHdr->Text      = L"Choose Exercise / \x0E40\x0E25\x0E37\x0E2D\x0E01\x0E17\x0E48\x0E32\x0E2D\x0E2D\x0E01\x0E01\x0E33\x0E25\x0E31\x0E07\x0E01\x0E32\x0E22";

		// Exercise picker flow panel
		flowExercisePicker->BackColor  = colBg;
		flowExercisePicker->Location   = System::Drawing::Point(40, 158);
		flowExercisePicker->Size       = System::Drawing::Size(425, 240);
		flowExercisePicker->AutoScroll = true;
		flowExercisePicker->WrapContents = true;

		static const wchar_t* exNames[] = {
			L"Bicep Curl", L"Squat", L"Push-up", L"Lunge",
			L"Shoulder Press", L"Deadlift", L"Lateral Raise", L"Plank",
			L"Tricep Pushdown", L"Pull-Up", L"Hip Thrust", L"Leg Extension",
			L"Bench Press", L"Chest Fly", L"Decline Bench Press", L"Hammer Curl",
			L"Incline Bench Press", L"Lat Pulldown", L"Leg Raises", L"Romanian Deadlift",
			L"Russian Twist", L"T-Bar Row", L"Tricep Dips"
		};
		for (int i = 0; i < 23; i++) {
			System::Windows::Forms::Button^ b = gcnew System::Windows::Forms::Button();
			b->BackColor = colCard;
			b->ForeColor = colTextPrimary;
			b->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			b->FlatAppearance->BorderColor = colSidebarBorder;
			b->FlatAppearance->BorderSize  = 1;
			b->Font      = gcnew System::Drawing::Font(L"Segoe UI", 8.5f, System::Drawing::FontStyle::Bold);
			b->Size      = System::Drawing::Size(125, 36);
			b->Margin    = System::Windows::Forms::Padding(3);
			b->Text      = gcnew System::String(exNames[i]);
			b->Cursor    = System::Windows::Forms::Cursors::Hand;
			b->UseVisualStyleBackColor = false;
			b->Tag       = i.ToString();
			b->Click += gcnew System::EventHandler(this, &MainForm::exercisePick_Click);
			flowExercisePicker->Controls->Add(b);
		}

		// Custom Program ListBox
		listCustomProgram->BackColor = colBarBg;
		listCustomProgram->ForeColor = colTextPrimary;
		listCustomProgram->Font = gcnew System::Drawing::Font(L"Segoe UI", 11);
		listCustomProgram->Location = System::Drawing::Point(480, 158);
		listCustomProgram->Size = System::Drawing::Size(360, 240);

		btnAddCustom->BackColor = colSidebarBorder;
		btnAddCustom->ForeColor = colTextPrimary;
		btnAddCustom->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
		btnAddCustom->Location = System::Drawing::Point(480, 420);
		btnAddCustom->Size = System::Drawing::Size(175, 32);
		btnAddCustom->Text = L"\u2795 Add to Queue";
		btnAddCustom->Click += gcnew System::EventHandler(this, &MainForm::btnAddCustom_Click);

		btnClearCustom->BackColor = colError;
		btnClearCustom->ForeColor = colBg;
		btnClearCustom->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
		btnClearCustom->Location = System::Drawing::Point(665, 420);
		btnClearCustom->Size = System::Drawing::Size(175, 32);
		btnClearCustom->Text = L"\u2716 Clear Queue";
		btnClearCustom->Click += gcnew System::EventHandler(this, &MainForm::btnClearCustom_Click);

		// Sets row
		lblPreSetsHdr->AutoSize  = false;
		lblPreSetsHdr->Font      = f10b;
		lblPreSetsHdr->ForeColor = colTextSecondary;
		lblPreSetsHdr->Location  = System::Drawing::Point(40, 420);
		lblPreSetsHdr->Size      = System::Drawing::Size(90, 22);
		lblPreSetsHdr->Text      = L"Sets / \x0E40\x0E0B\x0E47\x0E15";

		numSets->BackColor = colBarBg;
		numSets->ForeColor = colTextPrimary;
		numSets->Font      = gcnew System::Drawing::Font(L"Segoe UI", 11);
		numSets->Location  = System::Drawing::Point(40, 446);
		numSets->Size      = System::Drawing::Size(90, 28);
		numSets->Minimum   = 1;
		numSets->Maximum   = 10;
		numSets->Value     = 3;

		// Reps row
		lblPreRepsHdr->AutoSize  = false;
		lblPreRepsHdr->Font      = f10b;
		lblPreRepsHdr->ForeColor = colTextSecondary;
		lblPreRepsHdr->Location  = System::Drawing::Point(140, 420);
		lblPreRepsHdr->Size      = System::Drawing::Size(90, 22);
		lblPreRepsHdr->Text      = L"Reps / \x0E23\x0E35\x0E1E";

		numReps->BackColor = colBarBg;
		numReps->ForeColor = colTextPrimary;
		numReps->Font      = gcnew System::Drawing::Font(L"Segoe UI", 11);
		numReps->Location  = System::Drawing::Point(140, 446);
		numReps->Size      = System::Drawing::Size(90, 28);
		numReps->Minimum   = 1;
		numReps->Maximum   = 300;
		numReps->Value     = 12;

		// Camera source row
		lblPreCamHdr->AutoSize  = false;
		lblPreCamHdr->Font      = f10b;
		lblPreCamHdr->ForeColor = colTextSecondary;
		lblPreCamHdr->Location  = System::Drawing::Point(240, 420);
		lblPreCamHdr->Size      = System::Drawing::Size(225, 22);
		lblPreCamHdr->Text      = L"Camera Source / \x0E01\x0E25\x0E49\x0E2D\x0E07";

		txtCameraSourcePre->BackColor    = colBarBg;
		txtCameraSourcePre->ForeColor    = colTextPrimary;
		txtCameraSourcePre->Font         = gcnew System::Drawing::Font(L"Segoe UI", 11);
		txtCameraSourcePre->BorderStyle  = System::Windows::Forms::BorderStyle::FixedSingle;
		txtCameraSourcePre->Location     = System::Drawing::Point(240, 446);
		txtCameraSourcePre->Size         = System::Drawing::Size(225, 28);
		txtCameraSourcePre->Text         = L"0";

		// Start Workout button
		btnStartWorkout->BackColor = colBtnStart;
		btnStartWorkout->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
		btnStartWorkout->FlatAppearance->BorderSize = 0;
		btnStartWorkout->FlatAppearance->MouseOverBackColor = System::Drawing::Color::FromArgb(66, 180, 87);
		btnStartWorkout->Font      = gcnew System::Drawing::Font(L"Segoe UI", 13, FontStyle::Bold);
		btnStartWorkout->ForeColor = colTextPrimary;
		btnStartWorkout->Location  = System::Drawing::Point(40, 490);
		btnStartWorkout->Size      = System::Drawing::Size(800, 56);
		btnStartWorkout->Text      = L"\x25B6  Start Workout / \x0E40\x0E23\x0E34\x0E48\x0E21\x0E2D\x0E2D\x0E01\x0E01\x0E33\x0E25\x0E31\x0E07\x0E01\x0E32\x0E22";
		btnStartWorkout->Cursor    = System::Windows::Forms::Cursors::Hand;
		btnStartWorkout->UseVisualStyleBackColor = false;
		btnStartWorkout->Click += gcnew System::EventHandler(this, &MainForm::btnStartWorkout_Click);

		// lblCurrentExercise for workout screen sidebar
		lblCurrentExercise->AutoSize  = false;
		lblCurrentExercise->Font      = gcnew System::Drawing::Font(L"Segoe UI", 10, FontStyle::Bold);
		lblCurrentExercise->ForeColor = colAccent;
		lblCurrentExercise->Location  = System::Drawing::Point(10, 8);
		lblCurrentExercise->Size      = System::Drawing::Size(260, 20);
		lblCurrentExercise->Text      = L"Bicep Curl";
		panelSidebar->Controls->Add(lblCurrentExercise);

		// Add pre-workout controls to panel
		panelPreWorkout->Controls->Add(btnBackPreWorkout);
		panelPreWorkout->Controls->Add(lblPreWorkoutTitle);
		panelPreWorkout->Controls->Add(lblSelectedProgram);
		panelPreWorkout->Controls->Add(lblExercisePickHdr);
		panelPreWorkout->Controls->Add(flowExercisePicker);
		panelPreWorkout->Controls->Add(listCustomProgram);
		panelPreWorkout->Controls->Add(btnAddCustom);
		panelPreWorkout->Controls->Add(btnClearCustom);
		panelPreWorkout->Controls->Add(lblPreSetsHdr);
		panelPreWorkout->Controls->Add(numSets);
		panelPreWorkout->Controls->Add(lblPreRepsHdr);
		panelPreWorkout->Controls->Add(numReps);
		panelPreWorkout->Controls->Add(lblPreCamHdr);
		panelPreWorkout->Controls->Add(txtCameraSourcePre);
		panelPreWorkout->Controls->Add(btnStartWorkout);
	}

	// ===== BUILD SUMMARY PANEL =====
	void BuildSummaryPanel()
	{
		using namespace System::Drawing;
		System::Drawing::Font^ f22b = gcnew System::Drawing::Font(L"Segoe UI", 22, FontStyle::Bold);
		System::Drawing::Font^ f16b = gcnew System::Drawing::Font(L"Segoe UI", 16, FontStyle::Bold);
		System::Drawing::Font^ f13b = gcnew System::Drawing::Font(L"Segoe UI", 13, FontStyle::Bold);
		System::Drawing::Font^ f12  = gcnew System::Drawing::Font(L"Segoe UI", 12);
		System::Drawing::Font^ f11b = gcnew System::Drawing::Font(L"Segoe UI", 11, FontStyle::Bold);
		System::Drawing::Font^ f11  = gcnew System::Drawing::Font(L"Segoe UI", 11);

		lblSummaryTitle->AutoSize  = false;
		lblSummaryTitle->Font      = f22b;
		lblSummaryTitle->ForeColor = colAccent;
		lblSummaryTitle->Location  = System::Drawing::Point(60, 50);
		lblSummaryTitle->Size      = System::Drawing::Size(800, 44);
		lblSummaryTitle->Text      = L"WORKOUT COMPLETE! / \x0E2D\x0E2D\x0E01\x0E01\x0E33\x0E25\x0E31\x0E07\x0E01\x0E32\x0E22\x0E40\x0E2A\x0E23\x0E47\x0E08\x0E41\x0E25\x0E49\x0E27!";

		lblSummaryExercise->AutoSize  = false;
		lblSummaryExercise->Font      = f16b;
		lblSummaryExercise->ForeColor = colTextPrimary;
		lblSummaryExercise->Location  = System::Drawing::Point(60, 108);
		lblSummaryExercise->Size      = System::Drawing::Size(700, 30);
		lblSummaryExercise->Text      = L"Exercise";

		// Stats grid (2 col)
		lblSummaryReps->AutoSize  = false;
		lblSummaryReps->Font      = f12;
		lblSummaryReps->ForeColor = colTextPrimary;
		lblSummaryReps->Location  = System::Drawing::Point(60, 160);
		lblSummaryReps->Size      = System::Drawing::Size(380, 26);
		lblSummaryReps->Text      = L"Total Reps / \x0E23\x0E35\x0E1E\x0E17\x0E31\x0E49\x0E07\x0E2B\x0E21\x0E14:  0";

		lblSummaryDuration->AutoSize  = false;
		lblSummaryDuration->Font      = f12;
		lblSummaryDuration->ForeColor = colTextPrimary;
		lblSummaryDuration->Location  = System::Drawing::Point(460, 160);
		lblSummaryDuration->Size      = System::Drawing::Size(380, 26);
		lblSummaryDuration->Text      = L"Duration / \x0E40\x0E27\x0E25\x0E32:  00:00";

		lblSummarySets->AutoSize  = false;
		lblSummarySets->Font      = f12;
		lblSummarySets->ForeColor = colTextPrimary;
		lblSummarySets->Location  = System::Drawing::Point(60, 194);
		lblSummarySets->Size      = System::Drawing::Size(380, 26);
		lblSummarySets->Text      = L"Sets / \x0E40\x0E0B\x0E47\x0E15:  0";

		lblSummaryCalories->AutoSize  = false;
		lblSummaryCalories->Font      = f12;
		lblSummaryCalories->ForeColor = colWarning;
		lblSummaryCalories->Location  = System::Drawing::Point(460, 194);
		lblSummaryCalories->Size      = System::Drawing::Size(380, 26);
		lblSummaryCalories->Text      = L"Calories / \x0E41\x0E04\x0E25\x0E2D\x0E23\x0E35\x0E48:  0 kcal";

		// Quality
		lblSummaryQualityHdr->AutoSize  = false;
		lblSummaryQualityHdr->Font      = f11b;
		lblSummaryQualityHdr->ForeColor = colAccent;
		lblSummaryQualityHdr->Location  = System::Drawing::Point(60, 250);
		lblSummaryQualityHdr->Size      = System::Drawing::Size(600, 24);
		lblSummaryQualityHdr->Text      = L"Rep Quality / \x0E04\x0E38\x0E13\x0E20\x0E32\x0E1E\x0E01\x0E32\x0E23\x0E17\x0E33\x0E0B\x0E49\x0E33";

		lblSummaryFull->AutoSize  = false;
		lblSummaryFull->Font      = f11b;
		lblSummaryFull->ForeColor = colSuccess;
		lblSummaryFull->Location  = System::Drawing::Point(60, 284);
		lblSummaryFull->Size      = System::Drawing::Size(220, 24);
		lblSummaryFull->Text      = L"\x2705 Full Range:  0";

		lblSummaryPartial->AutoSize  = false;
		lblSummaryPartial->Font      = f11b;
		lblSummaryPartial->ForeColor = colWarning;
		lblSummaryPartial->Location  = System::Drawing::Point(300, 284);
		lblSummaryPartial->Size      = System::Drawing::Size(220, 24);
		lblSummaryPartial->Text      = L"\x26A0 Partial:  0";

		lblSummaryPoor->AutoSize  = false;
		lblSummaryPoor->Font      = f11b;
		lblSummaryPoor->ForeColor = colError;
		lblSummaryPoor->Location  = System::Drawing::Point(540, 284);
		lblSummaryPoor->Size      = System::Drawing::Size(220, 24);
		lblSummaryPoor->Text      = L"\x274C Poor Form:  0";

		lblSummaryMotivation->AutoSize  = false;
		lblSummaryMotivation->Font      = gcnew System::Drawing::Font(L"Segoe UI", 13, FontStyle::Bold | FontStyle::Italic);
		lblSummaryMotivation->ForeColor = colAccent;
		lblSummaryMotivation->Location  = System::Drawing::Point(60, 330);
		lblSummaryMotivation->Size      = System::Drawing::Size(700, 30);
		lblSummaryMotivation->Text      = L"Great work!";

		// Buttons
		btnSummaryHome->BackColor = colSidebarBorder;
		btnSummaryHome->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
		btnSummaryHome->FlatAppearance->BorderSize = 0;
		btnSummaryHome->Font      = gcnew System::Drawing::Font(L"Segoe UI", 12, FontStyle::Bold);
		btnSummaryHome->ForeColor = colTextPrimary;
		btnSummaryHome->Location  = System::Drawing::Point(60, 390);
		btnSummaryHome->Size      = System::Drawing::Size(260, 52);
		btnSummaryHome->Text      = L"Back to Home / \x0E01\x0E25\x0E31\x0E1A\x0E2B\x0E19\x0E49\x0E32\x0E2B\x0E25\x0E31\x0E01";
		btnSummaryHome->Cursor    = System::Windows::Forms::Cursors::Hand;
		btnSummaryHome->UseVisualStyleBackColor = false;
		btnSummaryHome->Click += gcnew System::EventHandler(this, &MainForm::btnSummaryHome_Click);

		btnSummaryAgain->BackColor = colBtnStart;
		btnSummaryAgain->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
		btnSummaryAgain->FlatAppearance->BorderSize = 0;
		btnSummaryAgain->Font      = gcnew System::Drawing::Font(L"Segoe UI", 12, FontStyle::Bold);
		btnSummaryAgain->ForeColor = colTextPrimary;
		btnSummaryAgain->Location  = System::Drawing::Point(340, 390);
		btnSummaryAgain->Size      = System::Drawing::Size(280, 52);
		btnSummaryAgain->Text      = L"Do Another Set / \x0E17\x0E33\x0E2D\x0E35\x0E01\x0E04\x0E23\x0E31\x0E49\x0E07";
		btnSummaryAgain->Cursor    = System::Windows::Forms::Cursors::Hand;
		btnSummaryAgain->UseVisualStyleBackColor = false;
		btnSummaryAgain->Click += gcnew System::EventHandler(this, &MainForm::btnSummaryAgain_Click);

		panelSummary->Controls->Add(lblSummaryTitle);
		panelSummary->Controls->Add(lblSummaryExercise);
		panelSummary->Controls->Add(lblSummaryReps);
		panelSummary->Controls->Add(lblSummaryDuration);
		panelSummary->Controls->Add(lblSummarySets);
		panelSummary->Controls->Add(lblSummaryCalories);
		panelSummary->Controls->Add(lblSummaryQualityHdr);
		panelSummary->Controls->Add(lblSummaryFull);
		panelSummary->Controls->Add(lblSummaryPartial);
		panelSummary->Controls->Add(lblSummaryPoor);
		panelSummary->Controls->Add(lblSummaryMotivation);
		panelSummary->Controls->Add(btnSummaryHome);
		panelSummary->Controls->Add(btnSummaryAgain);
	}

	// ===== BUILD HISTORY PANEL =====
	void BuildHistoryPanel()
	{
		using namespace System::Drawing;
		System::Drawing::Font^ f18b = gcnew System::Drawing::Font(L"Segoe UI", 18, FontStyle::Bold);
		System::Drawing::Font^ f10b = gcnew System::Drawing::Font(L"Segoe UI", 10, FontStyle::Bold);
		System::Drawing::Font^ f9   = gcnew System::Drawing::Font(L"Segoe UI", 9);
		System::Drawing::Font^ f10  = gcnew System::Drawing::Font(L"Segoe UI", 10);

		btnHistoryBack->BackColor = colSidebarBorder;
		btnHistoryBack->FlatStyle = FlatStyle::Flat;
		btnHistoryBack->FlatAppearance->BorderSize = 0;
		btnHistoryBack->Font      = f10b;
		btnHistoryBack->ForeColor = colTextPrimary;
		btnHistoryBack->Location  = System::Drawing::Point(20, 16);
		btnHistoryBack->Size      = System::Drawing::Size(100, 32);
		btnHistoryBack->Text      = L"\u2190  Back";
		btnHistoryBack->Cursor    = Cursors::Hand;
		btnHistoryBack->UseVisualStyleBackColor = false;
		btnHistoryBack->Click += gcnew System::EventHandler(this, &MainForm::btnHistoryBack_Click);

		lblHistoryTitle->AutoSize  = false;
		lblHistoryTitle->Font      = f18b;
		lblHistoryTitle->ForeColor = colTextPrimary;
		lblHistoryTitle->Location  = System::Drawing::Point(40, 60);
		lblHistoryTitle->Size      = System::Drawing::Size(700, 34);
		lblHistoryTitle->Text      = L"History / \u0E1B\u0E23\u0E30\u0E27\u0E31\u0E15\u0E34\u0E01\u0E32\u0E23\u0E2D\u0E2D\u0E01\u0E01\u0E33\u0E25\u0E31\u0E07\u0E01\u0E32\u0E22";

		lblHistoryStats->AutoSize  = false;
		lblHistoryStats->Font      = f9;
		lblHistoryStats->ForeColor = colTextSecondary;
		lblHistoryStats->Location  = System::Drawing::Point(40, 104);
		lblHistoryStats->Size      = System::Drawing::Size(800, 20);
		lblHistoryStats->Text      = L"No workouts yet";

		listHistory->BackColor  = colBarBg;
		listHistory->ForeColor  = colTextPrimary;
		listHistory->Font       = f10;
		listHistory->BorderStyle = System::Windows::Forms::BorderStyle::None;
		listHistory->Location   = System::Drawing::Point(40, 136);
		listHistory->Size       = System::Drawing::Size(1000, 580);
		listHistory->SelectionMode = System::Windows::Forms::SelectionMode::None;

		panelHistory->Controls->Add(btnHistoryBack);
		panelHistory->Controls->Add(lblHistoryTitle);
		panelHistory->Controls->Add(lblHistoryStats);
		panelHistory->Controls->Add(listHistory);
	}

	// ===== POPULATE HISTORY LIST =====
	void PopulateHistory()
	{
		listHistory->Items->Clear();
		try {
			String^ fp = "workout_history.csv";
			if (!System::IO::File::Exists(fp)) {
				lblHistoryStats->Text = L"No workouts yet";
				return;
			}
			auto lines = System::IO::File::ReadAllLines(fp);
			int totalWO = 0, totalReps = 0; double totalCal = 0.0;
			// Reverse order (newest first)
			for (int i = lines->Length - 1; i >= 1; i--) {
				auto c = lines[i]->Split(L',');
				if (c->Length < 13) continue;
				totalWO++;
				int r = 0; Int32::TryParse(c[10], r); totalReps += r;
				double cal = 0.0; Double::TryParse(c[12], cal); totalCal += cal;
				int dur = 0; Int32::TryParse(c[11], dur);
				String^ row = String::Format("{0}  {1,-20}  {2} reps  {3:F0} kcal  {4:D2}:{5:D2}",
					c[0], c[7], r, cal, dur/60, dur%60);
				listHistory->Items->Add(row);
			}
			lblHistoryStats->Text = String::Format(
				L"Total: {0} workouts  |  {1} reps  |  {2:F0} kcal",
				totalWO, totalReps, totalCal);
		} catch (...) { lblHistoryStats->Text = L"Error reading history"; }
	}

		void drawAngleArc(cv::Mat& frame, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3,
			cv::Scalar color)
		{
			double a1 = std::atan2(p1.y - p2.y, p1.x - p2.x) * 180.0 / CV_PI;
			double a2 = std::atan2(p3.y - p2.y, p3.x - p2.x) * 180.0 / CV_PI;
			double startAngle = (std::min)(a1, a2);
			double endAngle   = (std::max)(a1, a2);
			if (endAngle - startAngle > 180.0) {
				std::swap(startAngle, endAngle);
				endAngle += 360.0;
			}
			cv::ellipse(frame, (cv::Point)p2, cv::Size(30, 30), 0,
				startAngle, endAngle, color, 2);
		}

		void UpdateProfileDisplay()
		{
			float   bmi    = profile.GetBMI();
			String^ bmiStr = (bmi > 0) ? bmi.ToString("F1") : L"--";
			String^ bmiCat = (bmi > 0) ? profile.GetBMICategory() : L"";
			lblProfileInfo->Text = L"\u25A3  " + profile.name
				+ L"  |  BMI: " + bmiStr + L"  " + bmiCat;

			targetReps = profile.RecommendedReps();
			targetSets = profile.RecommendedSets();
			restDurationSec = profile.RecommendedRestSec();

			lblGoalTarget->Text = L"\u25BA  "
				+ UserProfile::GoalName(profile.goalIndex)
				+ L"  |  " + targetSets.ToString() + L" sets \u00D7 "
				+ targetReps.ToString() + L" reps";

			lblSet->Text = L"SET 1 / " + targetSets.ToString();
			lblRepProgress->Text = L"0 / " + targetReps.ToString();
		}

		System::Void btnEditProfile_Click(System::Object^ sender, System::EventArgs^ e)
		{
			ProfileForm^ pf = gcnew ProfileForm(profile, true);
			if (pf->ShowDialog(this) == System::Windows::Forms::DialogResult::OK) {
				profile = pf->profile;
				UpdateProfileDisplay();
			}
		}

	// ===== NAVIGATION HANDLERS =====
	System::Void btnQuickStart_Click(System::Object^ sender, System::EventArgs^ e) {
		currentProgramIndex = -1;
		lblSelectedProgram->Text = L"Quick Start";
		HighlightExerciseButton(preWorkoutExercise);
		numSets->Value = profile.RecommendedSets();
		numReps->Value = profile.RecommendedReps();
		ShowScreen(1);
	}

	System::Void btnViewHistory_Click(System::Object^ sender, System::EventArgs^ e) {
		PopulateHistory();
		ShowScreen(4);
	}

	void SelectProgram(int idx) {
		currentProgramIndex = idx;
		preWorkoutExercise = PROGRAMS[idx].steps[0].exerciseType;
		preWorkoutSets     = PROGRAMS[idx].steps[0].sets;
		preWorkoutReps     = PROGRAMS[idx].steps[0].reps;
		String^ nm = gcnew String(PROGRAMS[idx].nameEn);
		String^ th = gcnew String(PROGRAMS[idx].nameTh);
		lblSelectedProgram->Text = nm + " / " + th;
		HighlightExerciseButton(preWorkoutExercise);
		numSets->Value = preWorkoutSets;
		numReps->Value = preWorkoutReps;
		ShowScreen(1);
	}

	System::Void btnProgram0_Click(System::Object^ sender, System::EventArgs^ e) {
		SelectProgram(0);
	}

	System::Void btnProgram1_Click(System::Object^ sender, System::EventArgs^ e) {
		SelectProgram(1);
	}

	System::Void btnProgram2_Click(System::Object^ sender, System::EventArgs^ e) {
		SelectProgram(2);
	}

	System::Void btnProgram3_Click(System::Object^ sender, System::EventArgs^ e) {
		SelectProgram(3);
	}

	System::Void btnProgram4_Click(System::Object^ sender, System::EventArgs^ e) {
		SelectProgram(4);
	}

	System::Void exercisePick_Click(System::Object^ sender, System::EventArgs^ e) {
		Button^ b = safe_cast<Button^>(sender);
		int idx = 0;
		Int32::TryParse(safe_cast<String^>(b->Tag), idx);
		HighlightExerciseButton(idx);
	}

	System::Void btnAddCustom_Click(System::Object^ sender, System::EventArgs^ e) {
		WorkoutExerciseStep step;
		step.exerciseType = preWorkoutExercise;
		step.sets = (int)numSets->Value;
		step.reps = (int)numReps->Value;
		customSteps->push_back(step);
		
		static const wchar_t* exNames[] = {
			L"Bicep Curl", L"Squat", L"Push-up", L"Lunge",
			L"Shoulder Press", L"Deadlift", L"Lateral Raise", L"Plank",
			L"Tricep Pushdown", L"Pull-Up", L"Hip Thrust", L"Leg Extension",
			L"Bench Press", L"Chest Fly", L"Decline Bench Press", L"Hammer Curl",
			L"Incline Bench Press", L"Lat Pulldown", L"Leg Raises", L"Romanian Deadlift",
			L"Russian Twist", L"T-Bar Row", L"Tricep Dips"
		};
		String^ item = gcnew String(exNames[preWorkoutExercise]) +
			L" - " + step.sets.ToString() + L" sets \xD7 " + step.reps.ToString() + L" reps";
		listCustomProgram->Items->Add(item);
	}

	System::Void btnClearCustom_Click(System::Object^ sender, System::EventArgs^ e) {
		customSteps->clear();
		listCustomProgram->Items->Clear();
	}

	System::Void btnBackPreWorkout_Click(System::Object^ sender, System::EventArgs^ e) {
		ShowScreen(0);
	}

	System::Void btnHistoryBack_Click(System::Object^ sender, System::EventArgs^ e) {
		ShowScreen(0);
	}

	System::Void btnSummaryHome_Click(System::Object^ sender, System::EventArgs^ e) {
		RefreshHomeScreen();
		ShowScreen(0);
	}

	System::Void btnSummaryAgain_Click(System::Object^ sender, System::EventArgs^ e) {
		HighlightExerciseButton(preWorkoutExercise);
		ShowScreen(1);
	}

	System::Void btnStartWorkout_Click(System::Object^ sender, System::EventArgs^ e) {
		// Gather pre-workout settings
		if (currentProgramIndex < 0 && (int)customSteps->size() == 0) {
			WorkoutExerciseStep step;
			step.exerciseType = preWorkoutExercise;
			step.sets = (int)numSets->Value;
			step.reps = (int)numReps->Value;
			customSteps->push_back(step);
			isCustomProgram = true;
		} else if ((int)customSteps->size() > 0) {
			isCustomProgram = true;
			preWorkoutExercise = (*customSteps)[0].exerciseType;
			preWorkoutSets = (*customSteps)[0].sets;
			preWorkoutReps = (*customSteps)[0].reps;
		} else {
			isCustomProgram = false;
			preWorkoutSets = (int)numSets->Value;
			preWorkoutReps = (int)numReps->Value;
		}
		String^ sourceText = txtCameraSourcePre->Text->Trim();

		// Set exercise type
		exerciseController->setExerciseType(static_cast<ExerciseType>(preWorkoutExercise));
		comboBoxExercise->SelectedIndex = preWorkoutExercise;  // keep in sync

		// Update workout screen exercise label
		static const wchar_t* exNames[] = {
			L"Bicep Curl",L"Squat",L"Push-up",L"Lunge",
			L"Shoulder Press",L"Deadlift",L"Lateral Raise",L"Plank",
			L"Tricep Pushdown",L"Pull-Up",L"Hip Thrust",L"Leg Extension",
			L"Bench Press",L"Chest Fly",L"Decline Bench Press",L"Hammer Curl",
			L"Incline Bench Press",L"Lat Pulldown",L"Leg Raises",L"Romanian Deadlift",
			L"Russian Twist",L"T-Bar Row",L"Tricep Dips"
		};
		summaryExerciseName = gcnew String(exNames[preWorkoutExercise]);
		lblCurrentExercise->Text = summaryExerciseName;

		// Populate Dashboard FLOW bar
		flowWorkoutBar->Controls->Clear();
		Label^ flowTitle = gcnew Label();
		flowTitle->Text = L"FLOW";
		flowTitle->Font = gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold);
		flowTitle->ForeColor = colTextPrimary;
		flowTitle->AutoSize = true;
		flowTitle->Margin = System::Windows::Forms::Padding(15, 8, 5, 0);
		flowWorkoutBar->Controls->Add(flowTitle);

		int stepCount = isCustomProgram ? (int)customSteps->size() : PROGRAMS[currentProgramIndex].stepCount;
		for (int i = 0; i < stepCount; i++) {
			int exIdx = isCustomProgram ? (*customSteps)[i].exerciseType : PROGRAMS[currentProgramIndex].steps[i].exerciseType;
			Label^ stepLbl = gcnew Label();
			stepLbl->Text = L" [" + (gcnew String(exNames[exIdx]))->ToUpper() + L"] ";
			stepLbl->Font = gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold);
			stepLbl->AutoSize = true;
			stepLbl->Margin = System::Windows::Forms::Padding(5, 8, 5, 0);
			if (i == 0) {
				stepLbl->ForeColor = Color::White;
				stepLbl->BackColor = colAccent; // Active
			} else {
				stepLbl->ForeColor = colTextSecondary;
				stepLbl->BackColor = Color::Transparent;
			}
			flowWorkoutBar->Controls->Add(stepLbl);
			
			if (i < stepCount - 1) {
				Label^ arrowLbl = gcnew Label();
				arrowLbl->Text = L"\u2192";
				arrowLbl->Font = gcnew System::Drawing::Font(L"Segoe UI", 12, System::Drawing::FontStyle::Bold);
				arrowLbl->ForeColor = colTextSecondary;
				arrowLbl->AutoSize = true;
				arrowLbl->Margin = System::Windows::Forms::Padding(0, 8, 0, 0);
				flowWorkoutBar->Controls->Add(arrowLbl);
			}
		}

		// Open camera
		bool cameraOpened = false;
		if (sourceText->StartsWith("http") || sourceText->StartsWith("rtsp")) {
			std::string url = msclr::interop::marshal_as<std::string>(sourceText);
			cameraOpened = cameraHandler->initCamera(url);
		} else {
			int idx2 = 0; Int32::TryParse(sourceText, idx2);
			cameraOpened = cameraHandler->initCamera(idx2);
		}

		if (!cameraOpened) {
			MessageBox::Show("Could not open camera.\nCheck Camera Source field.", "Camera Error",
				MessageBoxButtons::OK, MessageBoxIcon::Warning);
			return;
		}

		isRunning    = true;
		threadRunning = true;
		resetRequested = false;
		exerciseChangeRequested = false;
		hasNewFrame  = false;
		isResting    = false;
		allSetsComplete = false;
		consecutiveReadFails = 0;
		*sharedCaptureError = "";

		targetReps = preWorkoutReps;
		targetSets = preWorkoutSets;
		restDurationSec = profile.RecommendedRestSec();
		currentSet = 1;

		lblSet->Text         = L"SET 1 / " + targetSets.ToString();
		lblRepProgress->Text  = L"0 / " + targetReps.ToString();
		panelRepProgressFill->Size = System::Drawing::Size(0, 8);
		lblRepsLabel->Text = (preWorkoutExercise == 7) ? L"SECONDS" : L"REPS";

		cameraThread = gcnew System::Threading::Thread(
			gcnew System::Threading::ThreadStart(this, &MainForm::CameraLoop));
		cameraThread->IsBackground = true;
		cameraThread->Name = L"CameraThread";
		cameraThread->Start();

		inferenceThread = gcnew System::Threading::Thread(
			gcnew System::Threading::ThreadStart(this, &MainForm::InferenceLoop));
		inferenceThread->IsBackground = true;
		inferenceThread->Name = L"InferenceThread";
		inferenceThread->Start();

		timer1->Start();

		btnStart->Enabled = false;
		btnStop->Enabled  = true;
		btnPause->Enabled = true;
		lblNextExercise->Visible = (currentProgramIndex >= 0);

		sessionStartTime  = DateTime::Now;
		workoutStartTime  = DateTime::Now;
		sessionTotalReps   = 0;
		sessionFullReps    = 0;
		sessionPartialReps = 0;
		sessionPoorReps    = 0;
		lastKnownRepCount = exerciseController->getRepCount();
		sessionCalories   = 0.0f;
		lblCalories->Text = L"\u25B2  0.0 kcal burned";

		lblStatus->ForeColor = colSuccess;
		lblStatus->Text = L"\u25CF  Camera: Connected";

		UpdateGuideGif(preWorkoutExercise);
		ShowScreen(2);  // Go to workout screen
	}

	System::Void pictureBox1_Paint(System::Object^ sender, System::Windows::Forms::PaintEventArgs^ e) {
		if (picBoxGuide->Image != nullptr) {
			try {
				// Setup transparency matrix (0.5 Opacity)
				float opacity = 0.55f;
				System::Drawing::Imaging::ColorMatrix^ matrix = gcnew System::Drawing::Imaging::ColorMatrix();
				matrix->Matrix33 = opacity; // Alpha channel

				System::Drawing::Imaging::ImageAttributes^ attributes = gcnew System::Drawing::Imaging::ImageAttributes();
				attributes->SetColorMatrix(matrix,
					System::Drawing::Imaging::ColorMatrixFlag::Default,
					System::Drawing::Imaging::ColorAdjustType::Bitmap);

				// Ensure proportional scaling for the GIF within a 350x350 box at bottom right
				int boxW = 400; int boxH = 400;
				int x = pictureBox1->Width - boxW - 20;
				int y = pictureBox1->Height - boxH - 20;

				Image^ gifImage = picBoxGuide->Image;
				float imgRatio = (float)gifImage->Width / (float)gifImage->Height;
				int drawW = boxW; int drawH = boxH;
				if (gifImage->Width > gifImage->Height) {
					drawH = (int)(boxW / imgRatio);
					y += (boxH - drawH) / 2; // Center vertically
				} else {
					drawW = (int)(boxH * imgRatio);
					x += (boxW - drawW) / 2; // Center horizontally
				}

				System::Drawing::Rectangle destRect(x, y, drawW, drawH);
				
				e->Graphics->DrawImage(gifImage, destRect, 
					0, 0, gifImage->Width, gifImage->Height, 
					System::Drawing::GraphicsUnit::Pixel, attributes);

				delete attributes;
				delete matrix;
			} catch (...) {}
		}
	}

	void EndWorkoutAndShowSummary()
	{
		if (!isRunning) return;
		isRunning = false;
		timer1->Stop();
		threadRunning = false;
		if (frameCondVar) frameCondVar->notify_all();

		if (cameraThread != nullptr && cameraThread->IsAlive) {
			cameraThread->Join(2000);
			cameraThread = nullptr;
		}
		if (inferenceThread != nullptr && inferenceThread->IsAlive) {
			inferenceThread->Join(2000);
			inferenceThread = nullptr;
		}
		cameraHandler->releaseCamera();
		
		btnStart->Enabled = true;
		btnStop->Enabled  = false;
		btnPause->Enabled = false;
		lblNextExercise->Visible = false;
		lblStatus->ForeColor = colTextSecondary;
		lblStatus->Text = L"\u25CF  Camera: Disconnected";

		// Capture final stats
		TimeSpan dur = DateTime::Now - workoutStartTime;
		if (!allSetsComplete && exerciseController) {
			sessionTotalReps   += exerciseController->getRepCount();
			sessionFullReps    += exerciseController->getFullReps();
			sessionPartialReps += exerciseController->getPartialReps();
			sessionPoorReps    += exerciseController->getPoorReps();
		}
		int totalReps    = sessionTotalReps;
		int fullR  = sessionFullReps;
		int partR  = sessionPartialReps;
		int poorR  = sessionPoorReps;
		int setsD  = System::Math::Min(currentSet, targetSets);

		// Save to CSV
		if (totalReps > 0) {
			double wt = profile.weightKg;
			saveToCSV(summaryExerciseName, wt, totalReps, (int)dur.TotalSeconds,
				fullR, partR, poorR, currentFps);
		}

		// Populate summary panel
		lblSummaryExercise->Text  = summaryExerciseName;
		lblSummaryReps->Text      = String::Format(L"Total Reps / \u0E23\u0E35\u0E1E:  {0}", totalReps);
		lblSummaryDuration->Text  = String::Format(L"Duration / \u0E40\u0E27\u0E25\u0E32:  {0:D2}:{1:D2}",
			(int)dur.TotalMinutes, dur.Seconds);
		lblSummarySets->Text      = String::Format(L"Sets / \u0E40\u0E0B\u0E47\u0E15:  {0}", setsD);
		lblSummaryCalories->Text  = String::Format(L"Calories / \u0E41\u0E04\u0E25\u0E2D\u0E23\u0E35\u0E48:  {0:F1} kcal", sessionCalories);
		lblSummaryFull->Text      = String::Format(L"\u2705 Full Range:  {0}", fullR);
		lblSummaryPartial->Text   = String::Format(L"\u26A0 Partial:  {0}", partR);
		lblSummaryPoor->Text      = String::Format(L"\u274C Poor Form:  {0}", poorR);

		// Motivational message
		int totQ = fullR + partR + poorR;
		float fp = (totQ > 0) ? (float)fullR / totQ : 0.0f;
		if      (fp >= 0.8f) lblSummaryMotivation->Text = L"";
		else if (fp >= 0.5f) lblSummaryMotivation->Text = L"";
		else                 lblSummaryMotivation->Text = L"";

		ShowScreen(3);
	}

		void saveToCSV(String^ exercise, double weight, int reps, int durationSec,
			int full, int partial, int poor, double avgFps)
		{
			try {
				String^ filePath = "workout_history.csv";
				bool isNew = !System::IO::File::Exists(filePath);
				System::IO::StreamWriter^ sw = gcnew System::IO::StreamWriter(filePath, true);
				if (isNew) {
					sw->WriteLine("Date,Time,Name,Gender,Goal,Level,BMI,Exercise,Weight_KG,"
						"Sets,Reps,Duration_Sec,Calories_kcal,Full,Partial,Poor,Avg_FPS");
				}
				sw->WriteLine(String::Format(
					"{0},{1},{2},{3},{4},{5},{6:F1},{7},{8:F1},{9},{10},{11},{12:F1},{13},{14},{15},{16:F1}",
					DateTime::Now.ToString("yyyy-MM-dd"),
					DateTime::Now.ToString("HH:mm:ss"),
					profile.name,
					UserProfile::GenderName(profile.genderIndex),
					UserProfile::GoalName(profile.goalIndex),
					UserProfile::LevelName(profile.levelIndex),
					profile.GetBMI(),
					exercise, weight,
					Math::Min(currentSet, targetSets),
					reps, durationSec, sessionCalories,
					full, partial, poor, avgFps));
				sw->Close();
				delete sw;
			} catch (Exception^) {}
		}

		Bitmap^ MatToBitmap(cv::Mat& mat) {
			if (mat.empty()) return nullptr;
			PixelFormat fmt;
			if      (mat.channels() == 3) fmt = PixelFormat::Format24bppRgb;
			else if (mat.channels() == 1) fmt = PixelFormat::Format8bppIndexed;
			else return nullptr;

			// Allocate a new Bitmap directly
			Bitmap^ bmp = gcnew Bitmap(mat.cols, mat.rows, fmt);
			
			// Lock the bits of the Bitmap to allow direct memory access
			System::Drawing::Rectangle rect(0, 0, mat.cols, mat.rows);
			System::Drawing::Imaging::BitmapData^ bmpData = bmp->LockBits(rect, System::Drawing::Imaging::ImageLockMode::WriteOnly, fmt);

			// Copy the data block from cv::Mat to Bitmap using fast memory copy
			size_t dataSize = mat.step[0] * mat.rows;
			std::memcpy((void*)bmpData->Scan0, mat.data, dataSize);

			// Unlock bits
			bmp->UnlockBits(bmpData);

			// Handle grayscale palette if needed
			if (fmt == PixelFormat::Format8bppIndexed) {
				ColorPalette^ pal = bmp->Palette;
				for (int i = 0; i < 256; i++) pal->Entries[i] = Color::FromArgb(i, i, i);
				bmp->Palette = pal;
			}
			
			// Return without cloning, preventing massive GC pressure
			return bmp;
		}
	};
}
