# Mar's 2026 F1 Race Predictions

> If this helped you, star the repo — it helps more people find it

## Scripts

### prediction1.py — Australian GP
Predicts race finish times and podium for the 2026 Australian Grand Prix.
- XGBoost regressor trained on qualifying times, grid positions, team scores, regulation change boosts, and weather
- Qualifying-to-race pace conversion with monotone constraints for logical feature relationships

### prediction2.py — Chinese GP
Predicts full finishing order for the 2026 Chinese Grand Prix.
- Blends Australia race results with China sprint race gaps as the target variable
- XGBoost with Leave-One-Out cross-validation across 7 features (sector times, grid positions, team score, weather)
- Live weather forecast from the OpenWeatherMap API with sensible fallback defaults

### racepace.py — Single Lap Pace Analysis
Pure pace comparison for the Chinese GP — no ML, just sector time analysis.
- Computes each driver's "ultimate lap" (best S1 + S2 + S3) and gap to the fastest car
- Estimates race pace from sprint pole reference with tire compound deltas

## Setup

```bash
pip install pandas numpy scikit-learn xgboost requests
```

For weather data in `prediction2.py`, create a `.env` file:

```bash
cp .env.example .env
# add your OpenWeatherMap API key to .env
```

Or export directly:

```bash
export OPENWEATHERMAP_API_KEY=your_key_here
```

The script works without a key — it falls back to default weather values.

## Usage

```bash
python prediction1.py    # Australian GP prediction
python prediction2.py    # Chinese GP prediction
python racepace.py       # Single lap pace analysis
```
