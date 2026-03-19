#pragma once

namespace ai_fitness_trainer {

	using namespace System;
	using namespace System::IO;

	// ─────────────────────────────────────────────────────────────────────────
	//  UserProfile  –  persisted user data and goal logic
	// ─────────────────────────────────────────────────────────────────────────
	public value class UserProfile
	{
	public:
		String^  name;
		int      age;
		float    weightKg;
		float    heightCm;
		int      genderIndex;  // 0=Male  1=Female
		int      goalIndex;    // 0=Weight Loss  1=Muscle  2=Endurance  3=General  4=Rehab
		int      levelIndex;   // 0=Beginner  1=Intermediate  2=Advanced
		bool     isValid;

		// ── Factory ──────────────────────────────────────────────────────────
		static UserProfile Default() {
			UserProfile p;
			p.name        = L"User";
			p.age         = 25;
			p.weightKg    = 70.0f;
			p.heightCm    = 170.0f;
			p.genderIndex = 0;
			p.goalIndex   = 3;
			p.levelIndex  = 0;
			p.isValid     = false;
			return p;
		}

		// ── BMI ──────────────────────────────────────────────────────────────
		float GetBMI() {
			float h = heightCm / 100.0f;
			return (h > 0) ? weightKg / (h * h) : 0.0f;
		}

		String^ GetBMICategory() {
			float bmi = GetBMI();
			if (bmi < 18.5f) return L"Underweight";
			if (bmi < 25.0f) return L"Normal";
			if (bmi < 30.0f) return L"Overweight";
			return L"Obese";
		}

		// ── Display names ─────────────────────────────────────────────────────
		static String^ GoalName(int idx) {
			cli::array<String^>^ g = {
				L"Weight Loss", L"Muscle Building", L"Endurance",
				L"General Fitness", L"Rehabilitation"
			};
			return (idx >= 0 && idx < 5) ? g[idx] : g[3];
		}

		static String^ LevelName(int idx) {
			cli::array<String^>^ l = { L"Beginner", L"Intermediate", L"Advanced" };
			return (idx >= 0 && idx < 3) ? l[idx] : l[0];
		}

		static String^ GenderName(int idx) {
			return (idx == 1) ? L"Female" : L"Male";
		}

		// ── Goal-driven recommendations ───────────────────────────────────────
		int RecommendedReps() {
			cli::array<int>^ r = { 15, 10, 20, 12, 8 };
			return (goalIndex >= 0 && goalIndex < 5) ? r[goalIndex] : 12;
		}

		int RecommendedSets() {
			cli::array<int>^ s = { 4, 4, 3, 3, 2 };
			return (goalIndex >= 0 && goalIndex < 5) ? s[goalIndex] : 3;
		}

		int RecommendedRestSec() {
			cli::array<int>^ r = { 30, 90, 20, 60, 120 };
			return (goalIndex >= 0 && goalIndex < 5) ? r[goalIndex] : 60;
		}

		// ── Calorie estimation ────────────────────────────────────────────────
		static float ExerciseMET(int exType) {
			cli::array<float>^ m = { 3.0f, 5.0f, 3.8f, 4.0f, 3.5f, 6.0f, 3.0f, 4.0f };
			return (exType >= 0 && exType < 8) ? m[exType] : 4.0f;
		}

		float EstimateCalories(int exType, float durationSec) {
			float base = ExerciseMET(exType) * weightKg * (durationSec / 3600.0f);
			// Women burn ~10% fewer calories on average for the same MET
			if (genderIndex == 1) base *= 0.9f;
			return base;
		}

		// ── Persistence ───────────────────────────────────────────────────────
		bool Save() {
			try {
				cli::array<String^>^ lines = {
					L"name="   + name,
					L"age="    + age.ToString(),
					L"weight=" + weightKg.ToString("F2"),
					L"height=" + heightCm.ToString("F2"),
					L"gender=" + genderIndex.ToString(),
					L"goal="   + goalIndex.ToString(),
					L"level="  + levelIndex.ToString(),
					L"valid=1"
				};
				File::WriteAllLines(L"profile.dat", lines);
				return true;
			} catch (Exception^) { return false; }
		}

		static UserProfile Load() {
			UserProfile p = Default();
			try {
				if (!File::Exists(L"profile.dat")) return p;
				cli::array<String^>^ lines = File::ReadAllLines(L"profile.dat");
				for each (String^ line in lines) {
					int eq = line->IndexOf('=');
					if (eq < 0) continue;
					String^ key = line->Substring(0, eq);
					String^ val = line->Substring(eq + 1);
					int   iv = 0;
					float fv = 0.0f;
					if      (key == L"name")   p.name = val;
					else if (key == L"age")    { Int32::TryParse(val, iv);  p.age         = iv; }
					else if (key == L"weight") { Single::TryParse(val, fv); p.weightKg    = fv; }
					else if (key == L"height") { Single::TryParse(val, fv); p.heightCm    = fv; }
					else if (key == L"gender") { Int32::TryParse(val, iv);  p.genderIndex = iv; }
					else if (key == L"goal")   { Int32::TryParse(val, iv);  p.goalIndex   = iv; }
					else if (key == L"level")  { Int32::TryParse(val, iv);  p.levelIndex  = iv; }
					else if (key == L"valid")  p.isValid = (val == L"1");
				}
			} catch (Exception^) {}
			return p;
		}
	};

} // namespace ai_fitness_trainer
