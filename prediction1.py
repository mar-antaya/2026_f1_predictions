import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

qualifying_2026 = pd.DataFrame({
    "Driver": [
        "RUS", "ANT", "HAD", "LEC", "PIA", "NOR", "HAM",
        "LAW", "LIN", "BOR", "HUL", "BEA", "OCO", "GAS",
        "ALB", "COL", "ALO", "PER", "BOT", "VER", "SAI", "STR"
    ],
    "QualifyingTime (s)": [
        78.518,   # RUS  P1  pole
        78.811,   # ANT  P2  +0.293
        79.303,   # HAD  P3  +0.785
        79.327,   # LEC  P4  +0.809
        79.380,   # PIA  P5  +0.862
        79.475,   # NOR  P6  +0.957
        79.478,   # HAM  P7  +0.960
        79.994,   # LAW  P8  +1.476
        81.247,   # LIN  P9  +2.729
        80.221,   # BOR  P10
        80.303,   # HUL  P11
        80.311,   # BEA  P12
        80.491,   # OCO  P13
        80.501,   # GAS  P14
        80.941,   # ALB  P15
        81.270,   # COL  P16
        81.969,   # ALO  P17
        82.605,   # PER  P18
        83.244,   # BOT  P19
        82.500,   # VER  P20 
        83.000,   # SAI  P21 
        85.000,   # STR  P22
    ],
    "GridPosition": list(range(1, 23))
})


team_points_2025 = {
    "McLaren":       800,   
    "Mercedes":      459,
    "Red Bull":      426,
    "Ferrari":       382,
    "Williams":      137,
    "Aston Martin":   80,
    "Haas":           73,
    "Racing Bulls":   92,
    "Audi":           68,   
    "Alpine":         22,
    "Cadillac":        5,   
}
max_pts = max(team_points_2025.values())
team_performance_score = {t: p / max_pts for t, p in team_points_2025.items()}

driver_to_team = {
    "RUS": "Mercedes",  "ANT": "Mercedes",
    "VER": "Red Bull",  "HAD": "Red Bull",
    "LEC": "Ferrari",   "HAM": "Ferrari",
    "NOR": "McLaren",   "PIA": "McLaren",
    "ALB": "Williams",  "SAI": "Williams",
    "ALO": "Aston Martin", "STR": "Aston Martin",
    "BEA": "Haas",      "OCO": "Haas",
    "LAW": "Racing Bulls", "LIN": "Racing Bulls",
    "HUL": "Audi",      "BOR": "Audi",
    "GAS": "Alpine",    "COL": "Alpine",
    "PER": "Cadillac",  "BOT": "Cadillac",
}

qualifying_2026["Team"] = qualifying_2026["Driver"].map(driver_to_team)
qualifying_2026["TeamPerformanceScore"] = qualifying_2026["Team"].map(team_performance_score)

reg_change_boost = {
    "Mercedes":      1.15,  
    "Ferrari":       1.05,   
    "Red Bull":      0.95,   
    "McLaren":       1.00,   
    "Williams":      0.80,
    "Aston Martin":  0.70,   
    "Haas":          0.85,
    "Racing Bulls":  0.88,
    "Audi":          0.83,
    "Alpine":        0.80,
    "Cadillac":      0.70,
}
qualifying_2026["RegChangeBoost"] = qualifying_2026["Team"].map(reg_change_boost)
qualifying_2026["AdjustedTeamScore"] = (
    qualifying_2026["TeamPerformanceScore"] * qualifying_2026["RegChangeBoost"]
)

rain_probability = 0.15
temperature = 22.0
qualifying_2026["RainProbability"] = rain_probability
qualifying_2026["Temperature"] = temperature

qualifying_2026["GridPenalty (s)"] = (qualifying_2026["GridPosition"] - 1) * 0.15
pole_time = qualifying_2026["QualifyingTime (s)"].min()
qualifying_2026["GapFromPole (s)"] = qualifying_2026["QualifyingTime (s)"] - pole_time


QUALI_TO_RACE_FACTOR = 1.055
merged = qualifying_2026.copy()
merged["TargetLapTime (s)"] = merged["QualifyingTime (s)"] * QUALI_TO_RACE_FACTOR


feature_cols = [
    "QualifyingTime (s)",    # strong positive correlation with race time
    "GapFromPole (s)",       # gap to pole, reinforces qualifying hierarchy
    "AdjustedTeamScore",     # team strength (higher = better), negative constraint
    "GridPenalty (s)",       # grid spot penalty
    "RainProbability",       # weather
    "Temperature",           # weather
]

X = merged[feature_cols].copy()
y = merged["TargetLapTime (s)"].copy()

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

QUALI_SCALE = 3.0
X_scaled = X_imputed.copy()
X_scaled[:, 0] *= QUALI_SCALE   
X_scaled[:, 1] *= QUALI_SCALE   

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42
)

model = XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=0.5,       
    max_delta_step=1,      
    random_state=42,
    monotone_constraints=(1, 1, -1, 1, 1, 0),  
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
merged["PredictedRaceTime (s)"] = model.predict(X_scaled)

final_results = merged.sort_values(
    by=["PredictedRaceTime (s)", "GridPosition"]
).reset_index(drop=True)


podium = final_results.head(3)
print("\n" + "="*65)
print("   🏆  PREDICTED PODIUM — 2026 AUSTRALIAN GRAND PRIX")
print("="*65)
print(f"  🥇  P1: {podium.iloc[0]['Driver']} ({podium.iloc[0]['Team']})")
print(f"  🥈  P2: {podium.iloc[1]['Driver']} ({podium.iloc[1]['Team']})")
print(f"  🥉  P3: {podium.iloc[2]['Driver']} ({podium.iloc[2]['Team']})")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"\n  Model MAE (test set): {mae:.3f} seconds")
print("="*65)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("2026 F1 Australian GP — Race Winner Prediction", fontsize=14, fontweight="bold")

importance = model.feature_importances_
feature_labels = [
    "Qualifying Time",
    "Gap From Pole",
    "Adjusted Team Score",
    "Grid Penalty",
    "Rain Probability",
    "Temperature",
]
sorted_idx = np.argsort(importance)
colors = ["#e10600" if f == "Qualifying Time" else "#1f77b4" for f in [feature_labels[i] for i in sorted_idx]]
axes[0].barh([feature_labels[i] for i in sorted_idx], importance[sorted_idx], color=colors)
axes[0].set_xlabel("Feature Importance Score")
axes[0].set_title("Feature Importance\n(red = Qualifying Time)")
axes[0].axvline(x=0, color="black", linewidth=0.5)
