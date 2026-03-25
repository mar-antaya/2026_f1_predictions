import pandas as pd
import numpy as np

quali_data = {
    "ANT": (24.003, 27.664, 40.387, 92.064, 1),
    "RUS": (24.012, 27.783, 40.491, 92.286, 2),
    "HAM": (24.080, 27.696, 40.535, 92.415, 3),
    "LEC": (24.022, 27.660, 40.650, 92.428, 4),
    "PIA": (24.120, 27.729, 40.493, 92.550, 5),
    "NOR": (23.995, 27.747, 40.748, 92.608, 6),
    "GAS": (24.099, 27.788, 40.900, 92.873, 7),
    "VER": (24.280, 27.975, 40.613, 93.002, 8),
    "BEA": (24.234, 27.843, 40.931, 93.292, 9),
    "HAD": (24.465, 27.933, 40.659, 93.121, 10),
    "HUL": (24.558, 27.937, 40.743, 93.238, 11),
    "COL": (24.254, 28.078, 40.947, 93.279, 12),
    "LAW": (24.339, 28.117, 40.911, 93.367, 13),
    "LIN": (24.319, 28.181, 40.903, 93.403, 14),
    "OCO": (24.335, 28.041, 41.028, 93.404, 15),
    "BOR": (24.539, 28.145, 40.796, 93.480, 16),
    "SAI": (24.465, 28.669, 41.183, 94.317, 17),
    "ALB": (24.526, 28.694, 41.370, 94.590, 18),
    "ALO": (24.782, 28.723, 41.698, 95.203, 19),
    "BOT": (24.949, 28.972, 41.515, 95.436, 20),
    "STR": (24.953, 29.144, 41.838, 95.935, 21),
    "PER": (25.703, 29.246, 41.611, 96.560, 22),
}

df = pd.DataFrame.from_dict(
    quali_data, orient='index',
    columns=['S1', 'S2', 'S3', 'ActualQuali_s', 'GridPos']
)
df.index.name = 'Driver'
df = df.reset_index()

df['UltimateLap_s'] = df['S1'] + df['S2'] + df['S3']
df['LeftOnTable_s'] = (df['ActualQuali_s'] - df['UltimateLap_s']).round(3)

SOFT_TO_MEDIUM_DELTA = 0.7
SPRINT_POLE_S        = 91.8
pace_reference       = SPRINT_POLE_S + SOFT_TO_MEDIUM_DELTA

best_ultimate         = df['UltimateLap_s'].min()
df['UltimateDelta_s'] = (df['UltimateLap_s'] - best_ultimate).round(3)
df['EstimatedPace_s'] = (pace_reference + df['UltimateDelta_s']).round(3)

df_sorted = df.sort_values('EstimatedPace_s').reset_index(drop=True)

print("  Single Lap Pace for 2026 Chinese GP")
print("  This measures raw car performance delta between drivers")
print(f"  {'P':<4}{'DRV':<6}{'Ultimate(s)':<13}{'Δ to ANT':<13}{'Est Pace(s)'}")
for i, r in df_sorted.iterrows():   
    delta = f"+{r['UltimateDelta_s']:.3f}s" if r['UltimateDelta_s'] > 0 else " REF"
    print(f"  {i+1:<4}{r['Driver']:<6}{r['UltimateLap_s']:<13.3f}{delta:<13}{r['EstimatedPace_s']:.3f}s")