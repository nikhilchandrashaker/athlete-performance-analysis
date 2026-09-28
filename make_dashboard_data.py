"""Computes the compact data block embedded in athlete_profile_dashboard.html
(percentile tables, tier medians, linear-model coefficients and tier cut points)."""
import pandas as pd, numpy as np, json
from sklearn.linear_model import LinearRegression

df = pd.read_csv('student_sports_performance_Dataset.csv')
order = ['Low', 'Moderate', 'High', 'Very High', 'Excellent']
df['r'] = df.performance_category.map({p: i for i, p in enumerate(order)})
V = ['sprint_speed_kmh', 'muscular_strength_kg', 'jumping_ability_cm', 'endurance_min', 'flexibility_cm',
     'running_time_sec', 'training_duration_min', 'activity_intensity', 'exercise_frequency_per_week', 'heart_rate_bpm']
mu, sd = df[V].mean(), df[V].std()
Z = (df[V] - mu) / sd
lr = LinearRegression().fit(Z, df.r)
s = lr.predict(Z)
cuts = [float(np.quantile(s, q)) for q in (.2, .4, .6, .8)]
q = np.arange(0, 101, 2)
D = {'V': V, 'mu': mu.round(4).tolist(), 'sd': sd.round(4).tolist(), 'coef': lr.coef_.round(4).tolist(),
     'b': float(lr.intercept_), 'cuts': [round(c, 4) for c in cuts],
     'q': {v: np.quantile(df[v], q / 100).round(2).tolist() for v in V},
     'tier': {p: df[df.performance_category == p][V].median().round(2).tolist() for p in order},
     'lo': df[V].min().tolist(), 'hi': df[V].max().tolist(), 'med': df[V].median().round(2).tolist()}
json.dump(D, open('dash.json', 'w'), separators=(',', ':'))
print('R^2 (in-sample):', lr.score(Z, df.r))
