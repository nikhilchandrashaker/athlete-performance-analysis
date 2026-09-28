# What Drives Athletic Performance?

A statistical study of 15,000 college athletes across eight sports (Athletics, Badminton, Basketball, Cricket, Football, Swimming, Tennis, Volleyball), asking which physical and training characteristics are most strongly associated with performance.

## Contents

| File | Description |
|---|---|
| `Athletic_Performance_Research_Paper.docx` | 11-page research paper: abstract, methods, results, discussion, limitations |
| `Athletic_Performance_Presentation.pptx` | 13-slide presentation of the same findings |
| `athlete_profile_dashboard.html` | Interactive athlete profile tool (single self-contained file) |
| `code/analysis.py` | Statistical analysis; writes `results.json` |
| `code/figures.py` | Generates the seven figures used in the paper and slides |
| `code/make_dashboard_data.py` | Computes the data embedded in the dashboard |

The dataset itself (`student_sports_performance_Dataset.csv`) is not included; place it next to the scripts to run them.

## Key findings

- **Explosive power leads.** Sprint speed (Spearman rho = 0.59), muscular strength (0.55) and jumping ability (0.52) had the strongest associations with performance tier, followed by endurance (0.30) and flexibility (0.21).
- **Training dose is weakly associated.** Duration, intensity and weekly frequency each showed rho of about 0.10-0.14.
- **Resting heart rate carries little signal** (rho = -0.11) and is essentially uncorrelated with endurance or sprint speed.
- **Predictors do not differ by sport.** The same top three predictors appeared in all eight sports, and mean performance rank did not differ across sports (ANOVA F = 0.30, p = .95). The hypothesis that predictors differ by sport was not supported.
- **Prediction.** A random forest classified performance tier with 54.8% test accuracy (54.5% mean 5-fold cross-validated) versus a 20% chance baseline. Errors fall mostly between adjacent tiers.

## Using the dashboard

Open `athlete_profile_dashboard.html` in any modern browser. It works offline and needs no server or install.

1. Adjust the ten sliders (they start at the median athlete).
2. Read the predicted tier (Low to Excellent).
3. Compare per-metric percentiles against the full sample; the red marker shows the median Excellent-tier athlete.

Running time and heart rate are scored so that lower is better. The tier estimate comes from a linear model of standardized measurements (in-sample R^2 of about 0.83 on tier rank). Sport is intentionally not an input, since sport did not change the predictor pattern.

## Reproducing the analysis

Requires Python 3 with `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` and `seaborn`.

```bash
python code/analysis.py              # results.json
python code/figures.py               # figures/*.png (run after analysis.py)
python code/make_dashboard_data.py   # dash.json (values embedded in the dashboard)
```

## Limitations

- **Observational, cross-sectional data.** Results are associations, not evidence that training or any trait causes better performance.
- **Narrow population.** College-aged athletes (17-25) described as China-based; results may not generalize to elite, younger or other populations.
- **Unusual uniformity.** The identical predictor patterns across sports and the very weak heart-rate signal are atypical of real-world athlete data. The performance label may reflect a generalized fitness composite rather than sport-specific competitive success.
- **In-sample dashboard model.** The dashboard's tier estimate is not validated on held-out data, and it is not a substitute for professional assessment.
