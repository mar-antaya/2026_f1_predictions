import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import requests
import os
import warnings
warnings.filterwarnings("ignore")

china_quali = {
    "ANT": (24.003, 27.664, 40.387, 92.064,  1),
    "RUS": (24.012, 27.783, 40.491, 92.286,  2),
    "HAM": (24.080, 27.696, 40.535, 92.415,  3),
    "LEC": (24.022, 27.660, 40.650, 92.428,  4),
    "PIA": (24.120, 27.729, 40.493, 92.550,  5),
    "NOR": (23.995, 27.747, 40.748, 92.608,  6),
    "GAS": (24.099, 27.788, 40.900, 92.873,  7),
    "VER": (24.280, 27.975, 40.613, 93.002,  8),
    "HAD": (24.465, 27.933, 40.659, 93.121,  9),
    "BEA": (24.234, 27.843, 40.931, 93.292, 10),
    "HUL": (24.558, 27.937, 40.743, 93.238, 11),
    "COL": (24.254, 28.078, 40.947, 93.279, 12),
    "OCO": (24.335, 28.041, 41.028, 93.404, 13),
    "LAW": (24.339, 28.117, 40.911, 93.367, 14),
    "LIN": (24.319, 28.181, 40.903, 93.403, 15),
    "BOR": (24.539, 28.145, 40.796, 93.480, 16),
    "SAI": (24.465, 28.669, 41.183, 94.317, 17),
    "ALB": (24.526, 28.694, 41.370, 94.590, 18),
    "ALO": (24.782, 28.723, 41.698, 95.203, 19),
    "BOT": (24.949, 28.972, 41.515, 95.436, 20),
    "STR": (24.953, 29.144, 41.838, 95.935, 21),
    "PER": (25.703, 29.246, 41.611, 96.560, 22),
}

df = pd.DataFrame.from_dict(
    china_quali, orient="index",
    columns=["S1", "S2", "S3", "ChinaQuali_s", "ChinaGrid"]
)
df.index.name = "Driver"
df = df.reset_index()

df["UltimateLap_s"] = df["S1"] + df["S2"] + df["S3"]
pole = df["ChinaQuali_s"].min()
df["ChinaGapFromPole_s"] = (df["ChinaQuali_s"] - pole).round(3)

aus_grid = {
    "RUS": 1,  "ANT": 2,  "HAD": 3,  "LEC": 4,  "PIA": 5,
    "NOR": 6,  "HAM": 7,  "LAW": 8,  "LIN": 9,  "BOR": 10,
    "OCO": 11, "HUL": 12, "ALB": 13, "GAS": 14, "COL": 15,
    "BEA": 16, "ALO": 17, "PER": 18, "BOT": 19, "VER": 20,
    "SAI": 21, "STR": 22,
}
df["AusGrid"] = df["Driver"].map(aus_grid)

aus_finish_pos = {
    "RUS": 1,  "ANT": 2,  "LEC": 3,  "HAM": 4,
    "NOR": 5,  "VER": 6,  "BEA": 7,  "LIN": 8,
    "BOR": 9,  "GAS": 10, "OCO": 11, "ALB": 12,
    "LAW": 13, "COL": 14, "SAI": 15, "PER": 16,
    "STR": 17,
    "ALO": 20, "BOT": 20, "HAD": 20,
    "PIA": 21, "HUL": 21,
}
df["AusFinishPos"] = df["Driver"].map(aus_finish_pos)

SPRINT_RACING_LAPS = 15
sprint_gaps = {
    "RUS": 0.000, "LEC": 0.674, "HAM": 2.554, "NOR": 4.433,
    "ANT": 5.688, "PIA": 6.809, "LAW": 10.900, "BEA": 11.271,
    "VER": 11.619, "OCO": 13.887, "GAS": 14.780, "SAI": 15.753,
    "BOR": 15.858, "COL": 16.393, "HAD": 16.430, "ALB": 20.014,
    "ALO": 21.599, "STR": 21.971, "PER": 28.241,
    "HUL": 35.0, "BOT": 35.0, "LIN": 35.0,
}

sprint_penalties = {"ANT": 10.0, "PER": 5.0}
sprint_adj = {
    d: max(gap - sprint_penalties.get(d, 0), 0)
    for d, gap in sprint_gaps.items()
}
df["SprintGapPerLap_s"] = df["Driver"].map(
    {d: round(g / SPRINT_RACING_LAPS, 4) for d, g in sprint_adj.items()}
)

team_points = {
    "Mercedes": 55, "Ferrari": 40, "McLaren": 18, "Red Bull": 8,
    "Haas": 7, "Racing Bulls": 6, "Audi": 2, "Alpine": 1,
    "Williams": 1, "Cadillac": 1, "Aston Martin": 1,
}
max_pts = max(team_points.values())
team_score = {t: max(p, 0.5) / max_pts for t, p in team_points.items()}

driver_to_team = {
    "RUS": "Mercedes", "ANT": "Mercedes", "HAM": "Ferrari",  "LEC": "Ferrari",
    "NOR": "McLaren",  "PIA": "McLaren",  "VER": "Red Bull", "HAD": "Red Bull",
    "BEA": "Haas",     "OCO": "Haas",     "LAW": "Racing Bulls", "LIN": "Racing Bulls",
    "HUL": "Audi",     "BOR": "Audi",     "GAS": "Alpine",   "COL": "Alpine",
    "SAI": "Williams", "ALB": "Williams", "PER": "Cadillac", "BOT": "Cadillac",
    "ALO": "Aston Martin", "STR": "Aston Martin",
}
df["Team"]      = df["Driver"].map(driver_to_team)
df["TeamScore"] = df["Team"].map(team_score)

API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
lat, lon = 31.3389, 121.2198
weather_url = (
    f"http://api.openweathermap.org/data/2.5/forecast"
    f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
)
rain_probability = 0.10
temperature      = 16.0
try:
    r = requests.get(weather_url, timeout=5)
    data = r.json()
    slot = next(
        (f for f in data.get("list", [])
         if f["dt_txt"] == "2026-03-15 07:00:00"), None
    )
    if slot:
        rain_probability = slot.get("pop", 0.10)
        temperature      = slot["main"]["temp"]
        print(f"Weather: {temperature:.1f}°C, rain {rain_probability:.0%}")
    else:
        print("Weather slot not found")
except Exception as e:
    print(f"Weather API unavailable — using defaults ({temperature}°C, {rain_probability:.0%})")

df["RainProbability"] = rain_probability
df["Temperature"]     = temperature

aus_pos_norm    = (df["AusFinishPos"]     - 1) / 21
sprint_gap_norm = df["SprintGapPerLap_s"] / df["SprintGapPerLap_s"].max()

df["RaceScore"] = (0.5 * aus_pos_norm + 0.5 * sprint_gap_norm).round(4)

feature_cols = [
    "UltimateLap_s",       # China quali: best S1+S2+S3 — pure car pace
    "ChinaGapFromPole_s",  # China quali: gap to pole — single lap delta
    "ChinaGrid",           # China qualifying position
    "AusGrid",             # Australia qualifying position — cross-circuit pace
    "TeamScore",           # 2026 constructor standings
    "RainProbability",     # weather
    "Temperature",         # weather
]

X = df[feature_cols].copy()
y = df["RaceScore"]

imputer   = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    random_state=42,
)

loo        = LeaveOneOut()
loo_errors = []
for train_idx, test_idx in loo.split(X_imputed):
    X_tr, X_te = X_imputed[train_idx], X_imputed[test_idx]
    y_tr, y_te = y.iloc[train_idx],    y.iloc[test_idx]
    model.fit(X_tr, y_tr)
    loo_errors.append(abs(model.predict(X_te)[0] - y_te.iloc[0]))

loo_mae = np.mean(loo_errors)

model.fit(X_imputed, y)
df["PredictedScore"] = model.predict(X_imputed)

final = df.sort_values("PredictedScore").reset_index(drop=True)

print("\n" + "=" * 68)
print("   PREDICTED FINISHING ORDER — 2026 CHINESE GRAND PRIX")
print("   Target: Australia finish pos + China sprint gap/lap")
print("=" * 68)
print(f"  {'P':<4}{'DRV':<6}{'Team'}")
print("  " + "─" * 20)
for i, row in final.iterrows():
    print(f"  {i+1:<4}{row['Driver']:<6}{row['Team']}")

podium = final.head(3)
print(f"\n  {'='*45}")
print(f"  🥇 P1: {podium.iloc[0]['Driver']} ({podium.iloc[0]['Team']})")
print(f"  🥈 P2: {podium.iloc[1]['Driver']} ({podium.iloc[1]['Team']})")
print(f"  🥉 P3: {podium.iloc[2]['Driver']} ({podium.iloc[2]['Team']})")
print(f"\n  Leave One out MAE: {loo_mae:.4f}")
print(f"  Weather: {temperature:.1f}°C | Rain: {rain_probability:.0%}")
print(f"  {'='*45}")

print("\nFeature importances:")
for feat, imp in sorted(zip(feature_cols, model.feature_importances_),
                         key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<22} {imp:.4f}  {bar}")