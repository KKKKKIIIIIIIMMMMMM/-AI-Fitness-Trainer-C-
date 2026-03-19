#pragma once

#include "UserProfile.h"

namespace ai_fitness_trainer {

	using namespace System;
	using namespace System::Windows::Forms;
	using namespace System::Drawing;

	// ─────────────────────────────────────────────────────────────────────────
	//  ProfileForm  –  first-run wizard & edit-profile dialog
	// ─────────────────────────────────────────────────────────────────────────
	public ref class ProfileForm : public Form
	{
	public:
		UserProfile profile;

		ProfileForm(UserProfile existing, bool editMode)
		{
			profile    = existing;
			_editMode  = editMode;
			InitializeComponent();
			if (editMode || existing.isValid)
				LoadProfileIntoUI();
			else
				UpdateBMI(nullptr, nullptr);
		}

	private:
		bool _editMode;

		// Controls
		TextBox^ txtName;
		TextBox^ txtAge;
		TextBox^ txtWeight;
		TextBox^ txtHeight;
		Label^   lblBMIValue;
		Label^   lblBMICat;
		cli::array<RadioButton^>^ rbGenders;
		cli::array<RadioButton^>^ rbGoals;
		cli::array<RadioButton^>^ rbLevels;
		Button^ btnSave;

		// ── Theme colours ─────────────────────────────────────────────────────
		static Color cBg      = Color::FromArgb( 18,  18,  30);
		static Color cHeader  = Color::FromArgb( 12,  42,  82);
		static Color cCard    = Color::FromArgb( 24,  36,  58);
		static Color cBorder  = Color::FromArgb( 40,  55,  85);
		static Color cAccent  = Color::FromArgb(  0, 180, 140);
		static Color cText    = Color::White;
		static Color cTextSec = Color::FromArgb(160, 170, 190);
		static Color cInput   = Color::FromArgb( 35,  45,  65);
		static Color cGreen   = Color::FromArgb( 46, 160,  67);

		// ── UI helpers ────────────────────────────────────────────────────────
		void SectionLabel(String^ text, int y) {
			auto lbl = gcnew Label();
			lbl->AutoSize = false;
			lbl->Font     = gcnew Drawing::Font(L"Segoe UI", 8, FontStyle::Bold);
			lbl->ForeColor = cAccent;
			lbl->Location  = Point(30, y);
			lbl->Size      = Drawing::Size(420, 18);
			lbl->Text      = text;
			this->Controls->Add(lbl);

			auto line = gcnew Panel();
			line->BackColor = cBorder;
			line->Location  = Point(30, y + 20);
			line->Size      = Drawing::Size(420, 1);
			this->Controls->Add(line);
		}

		Label^ FieldLabel(String^ text, int x, int y) {
			auto lbl = gcnew Label();
			lbl->AutoSize  = true;
			lbl->ForeColor = cTextSec;
			lbl->Font      = gcnew Drawing::Font(L"Segoe UI", 9);
			lbl->Location  = Point(x, y + 5);
			lbl->Text      = text;
			this->Controls->Add(lbl);
			return lbl;
		}

		TextBox^ InputBox(int x, int y, int w) {
			auto tb = gcnew TextBox();
			tb->BackColor   = cInput;
			tb->ForeColor   = cText;
			tb->BorderStyle = BorderStyle::FixedSingle;
			tb->Font        = gcnew Drawing::Font(L"Segoe UI", 10);
			tb->Location    = Point(x, y);
			tb->Size        = Drawing::Size(w, 26);
			this->Controls->Add(tb);
			return tb;
		}

		// ── InitializeComponent ───────────────────────────────────────────────
		void InitializeComponent()
		{
			this->SuspendLayout();
			this->BackColor = cBg;
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedSingle;
			this->MaximizeBox = false;
			this->StartPosition = FormStartPosition::CenterScreen;
			this->Text = _editMode ? L"Edit Profile" : L"AI Fitness Trainer \u2014 Profile Setup";
			this->Font = gcnew Drawing::Font(L"Segoe UI", 9);

			// ── Header ──────────────────────────────────────────────────────
			auto panelHeader = gcnew Panel();
			panelHeader->BackColor = cHeader;
			panelHeader->Dock = DockStyle::Top;
			panelHeader->Size = Drawing::Size(480, 60);

			auto lblTitle = gcnew Label();
			lblTitle->AutoSize  = false;
			lblTitle->Dock      = DockStyle::Fill;
			lblTitle->Font      = gcnew Drawing::Font(L"Segoe UI", 13, FontStyle::Bold);
			lblTitle->ForeColor = cText;
			lblTitle->TextAlign = ContentAlignment::MiddleLeft;
			lblTitle->Text = _editMode
				? L"   Edit Your Profile"
				: L"   Welcome! Set up your profile";
			panelHeader->Controls->Add(lblTitle);
			this->Controls->Add(panelHeader);

			int y = 76;

			// ── PERSONAL INFORMATION ─────────────────────────────────────────
			SectionLabel(L"PERSONAL INFORMATION", y);
			y += 30;

			FieldLabel(L"Name", 30, y);
			txtName = InputBox(110, y, 340);
			txtName->Text = L"User";
			y += 38;

			FieldLabel(L"Age", 30, y);
			txtAge = InputBox(110, y, 70);
			txtAge->Text = L"25";

			FieldLabel(L"Weight (kg)", 210, y);
			txtWeight = InputBox(310, y, 70);
			txtWeight->Text = L"70";
			y += 38;

			FieldLabel(L"Height (cm)", 30, y);
			txtHeight = InputBox(110, y, 70);
			txtHeight->Text = L"170";

			// BMI live display
			lblBMIValue = gcnew Label();
			lblBMIValue->AutoSize  = false;
			lblBMIValue->Font      = gcnew Drawing::Font(L"Segoe UI", 11, FontStyle::Bold);
			lblBMIValue->ForeColor = cAccent;
			lblBMIValue->Location  = Point(210, y);
			lblBMIValue->Size      = Drawing::Size(100, 28);
			lblBMIValue->Text      = L"BMI: --";
			this->Controls->Add(lblBMIValue);

			lblBMICat = gcnew Label();
			lblBMICat->AutoSize  = false;
			lblBMICat->Font      = gcnew Drawing::Font(L"Segoe UI", 8.5f);
			lblBMICat->ForeColor = cTextSec;
			lblBMICat->Location  = Point(316, y + 6);
			lblBMICat->Size      = Drawing::Size(134, 18);
			lblBMICat->Text      = L"";
			this->Controls->Add(lblBMICat);

			txtWeight->TextChanged += gcnew EventHandler(this, &ProfileForm::UpdateBMI);
			txtHeight->TextChanged += gcnew EventHandler(this, &ProfileForm::UpdateBMI);
			y += 48;

			// ── GENDER ──────────────────────────────────────────────────────
			SectionLabel(L"GENDER", y);
			y += 30;

			auto genderPanel = gcnew Panel();
			genderPanel->BackColor = Color::Transparent;
			genderPanel->Location  = Point(30, y);
			genderPanel->Size      = Drawing::Size(420, 28);

			cli::array<String^>^ genderLabels = { L"Male", L"Female" };
			rbGenders = gcnew cli::array<RadioButton^>(2);
			for (int i = 0; i < 2; i++) {
				auto rb = gcnew RadioButton();
				rb->ForeColor = cText;
				rb->Font      = gcnew Drawing::Font(L"Segoe UI", 9);
				rb->Text      = genderLabels[i];
				rb->Location  = Point(i * 140, 0);
				rb->Size      = Drawing::Size(135, 24);
				genderPanel->Controls->Add(rb);
				rbGenders[i] = rb;
			}
			rbGenders[0]->Checked = true;
			this->Controls->Add(genderPanel);
			y += 38;

			// ── FITNESS GOAL ─────────────────────────────────────────────────
			SectionLabel(L"FITNESS GOAL", y);
			y += 30;

			auto goalPanel = gcnew Panel();
			goalPanel->BackColor = Color::Transparent;
			goalPanel->Location  = Point(30, y);
			goalPanel->Size      = Drawing::Size(420, 58);

			cli::array<String^>^ goalLabels = {
				L"Weight Loss", L"Muscle Building", L"Endurance",
				L"General Fitness", L"Rehabilitation"
			};
			rbGoals = gcnew cli::array<RadioButton^>(5);
			for (int i = 0; i < 5; i++) {
				auto rb = gcnew RadioButton();
				rb->ForeColor = cText;
				rb->Font      = gcnew Drawing::Font(L"Segoe UI", 9);
				rb->Text      = goalLabels[i];
				rb->Location  = Point((i % 3) * 140, (i / 3) * 30);
				rb->Size      = Drawing::Size(135, 24);
				goalPanel->Controls->Add(rb);
				rbGoals[i] = rb;
			}
			rbGoals[3]->Checked = true;
			this->Controls->Add(goalPanel);
			y += 68;

			// ── FITNESS LEVEL ────────────────────────────────────────────────
			SectionLabel(L"FITNESS LEVEL", y);
			y += 30;

			auto levelPanel = gcnew Panel();
			levelPanel->BackColor = Color::Transparent;
			levelPanel->Location  = Point(30, y);
			levelPanel->Size      = Drawing::Size(420, 28);

			cli::array<String^>^ levelLabels = { L"Beginner", L"Intermediate", L"Advanced" };
			rbLevels = gcnew cli::array<RadioButton^>(3);
			for (int i = 0; i < 3; i++) {
				auto rb = gcnew RadioButton();
				rb->ForeColor = cText;
				rb->Font      = gcnew Drawing::Font(L"Segoe UI", 9);
				rb->Text      = levelLabels[i];
				rb->Location  = Point(i * 140, 0);
				rb->Size      = Drawing::Size(135, 24);
				levelPanel->Controls->Add(rb);
				rbLevels[i] = rb;
			}
			rbLevels[0]->Checked = true;
			this->Controls->Add(levelPanel);
			y += 48;

			// ── SAVE BUTTON ──────────────────────────────────────────────────
			btnSave = gcnew Button();
			btnSave->BackColor = cGreen;
			btnSave->FlatStyle = FlatStyle::Flat;
			btnSave->FlatAppearance->BorderSize = 0;
			btnSave->Font      = gcnew Drawing::Font(L"Segoe UI", 11, FontStyle::Bold);
			btnSave->ForeColor = cText;
			btnSave->Location  = Point(30, y);
			btnSave->Size      = Drawing::Size(420, 44);
			btnSave->Text      = _editMode ? L"Save Changes" : L"Start Training";
			btnSave->UseVisualStyleBackColor = false;
			btnSave->Cursor    = Cursors::Hand;
			btnSave->Click    += gcnew EventHandler(this, &ProfileForm::OnSave);
			this->Controls->Add(btnSave);

			this->ClientSize = Drawing::Size(480, y + 60);
			this->ResumeLayout(false);
		}

		void LoadProfileIntoUI()
		{
			txtName->Text   = profile.name;
			txtAge->Text    = profile.age.ToString();
			txtWeight->Text = profile.weightKg.ToString("F1");
			txtHeight->Text = profile.heightCm.ToString("F0");
			for (int i = 0; i < 2; i++) rbGenders[i]->Checked = (i == profile.genderIndex);
			for (int i = 0; i < 5; i++) rbGoals[i]->Checked   = (i == profile.goalIndex);
			for (int i = 0; i < 3; i++) rbLevels[i]->Checked  = (i == profile.levelIndex);
			UpdateBMI(nullptr, nullptr);
		}

		void UpdateBMI(Object^, EventArgs^)
		{
			float w = 0.0f, h = 0.0f;
			Single::TryParse(txtWeight->Text, w);
			Single::TryParse(txtHeight->Text, h);
			if (w <= 0 || h <= 0) { lblBMIValue->Text = L"BMI: --"; lblBMICat->Text = L""; return; }

			float hm  = h / 100.0f;
			float bmi = w / (hm * hm);
			lblBMIValue->Text = L"BMI: " + bmi.ToString("F1");

			String^ cat; Color col;
			if      (bmi < 18.5f) { cat = L"Underweight"; col = Color::FromArgb( 33, 150, 243); }
			else if (bmi < 25.0f) { cat = L"Normal";      col = cAccent; }
			else if (bmi < 30.0f) { cat = L"Overweight";  col = Color::FromArgb(255, 193,   7); }
			else                  { cat = L"Obese";        col = Color::FromArgb(244,  67,  54); }
			lblBMICat->ForeColor = col;
			lblBMICat->Text      = cat;
		}

		void OnSave(Object^, EventArgs^)
		{
			float w = 0.0f, h = 0.0f;
			int   a = 0;
			if (!Single::TryParse(txtWeight->Text, w) || w < 1 || w > 500) {
				MessageBox::Show(L"Please enter a valid weight (1\u2013500 kg).", L"Validation",
					MessageBoxButtons::OK, MessageBoxIcon::Warning); return;
			}
			if (!Single::TryParse(txtHeight->Text, h) || h < 50 || h > 300) {
				MessageBox::Show(L"Please enter a valid height (50\u2013300 cm).", L"Validation",
					MessageBoxButtons::OK, MessageBoxIcon::Warning); return;
			}
			if (!Int32::TryParse(txtAge->Text, a) || a < 5 || a > 120) {
				MessageBox::Show(L"Please enter a valid age (5\u2013120).", L"Validation",
					MessageBoxButtons::OK, MessageBoxIcon::Warning); return;
			}

			String^ nm = txtName->Text->Trim();
			if (nm->Length == 0) nm = L"User";

			profile.name      = nm;
			profile.age       = a;
			profile.weightKg  = w;
			profile.heightCm  = h;
			profile.isValid   = true;

			for (int i = 0; i < 2; i++) if (rbGenders[i]->Checked) { profile.genderIndex = i; break; }
			for (int i = 0; i < 5; i++) if (rbGoals[i]->Checked)   { profile.goalIndex   = i; break; }
			for (int i = 0; i < 3; i++) if (rbLevels[i]->Checked)  { profile.levelIndex  = i; break; }

			profile.Save();
			this->DialogResult = System::Windows::Forms::DialogResult::OK;
			this->Close();
		}
	};

} // namespace ai_fitness_trainer
