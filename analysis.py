import pandas as pd, numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

df = pd.read_csv('student_sports_performance_Dataset.csv')

perf_order = ['Low','Moderate','High','Very High','Excellent']
df['perf_rank'] = df['performance_category'].map({p:i for i,p in enumerate(perf_order)})

numeric_vars = ['age','height_cm','weight_kg','bmi','running_time_sec','sprint_speed_kmh',
                'jumping_ability_cm','flexibility_cm','muscular_strength_kg','endurance_min',
                'heart_rate_bpm','training_duration_min','activity_intensity','exercise_frequency_per_week']

results = {}

# 1. Descriptive stats by performance category
desc = df.groupby('performance_category')[numeric_vars].mean().reindex(perf_order)
results['descriptives_by_performance'] = desc.round(2).to_dict()

# 2. Correlation with perf_rank (Spearman, since ordinal)
corrs = {}
for v in numeric_vars:
    rho, p = stats.spearmanr(df[v], df['perf_rank'])
    corrs[v] = {'spearman_r': round(rho,4), 'p_value': p}
results['correlations_with_performance'] = corrs

# 3. Kruskal-Wallis across performance groups for each var
kw = {}
for v in numeric_vars:
    groups = [df[df['performance_category']==p][v] for p in perf_order]
    h, p = stats.kruskal(*groups)
    kw[v] = {'H': round(h,2), 'p_value': p}
results['kruskal_wallis'] = kw

# 4. Speed-Strength-Jump correlations
ss = df[['sprint_speed_kmh','muscular_strength_kg','jumping_ability_cm','running_time_sec']].corr(method='spearman')
results['speed_strength_corr'] = ss.round(3).to_dict()

# 5. Heart rate / endurance / running performance
hr = df[['heart_rate_bpm','endurance_min','running_time_sec','sprint_speed_kmh']].corr(method='spearman')
results['heart_rate_corr'] = hr.round(3).to_dict()

# 6. Training dose correlations
train_vars = ['training_duration_min','activity_intensity','exercise_frequency_per_week']
tc = {}
for v in train_vars:
    rho,p = stats.spearmanr(df[v], df['perf_rank'])
    tc[v] = {'spearman_r': round(rho,4),'p_value': p}
results['training_dose_corr'] = tc

# 7. Sport category differences - mean profile per sport (z-scored)
sport_profile = df.groupby('sport_category')[numeric_vars].mean()
z_profile = (sport_profile - sport_profile.mean())/sport_profile.std()
results['sport_profiles_z'] = z_profile.round(2).to_dict()

# ANOVA of performance rank across sports
groups_sport = [df[df['sport_category']==s]['perf_rank'] for s in df['sport_category'].unique()]
f,p = stats.f_oneway(*groups_sport)
results['sport_perf_anova'] = {'F': round(f,3), 'p_value': p}

# 8. Predictive modeling: Random Forest classification of performance_category
X = df[numeric_vars]
y = df['performance_category']
le = LabelEncoder()
y_enc = le.fit_transform(y)  # alphabetical; we'll map back
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
acc = accuracy_score(y_test, pred)
cv_scores = cross_val_score(rf, X, y_enc, cv=5)

importances = dict(zip(numeric_vars, rf.feature_importances_))
importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

results['rf_model'] = {
    'test_accuracy': round(acc,4),
    'cv_mean_accuracy': round(cv_scores.mean(),4),
    'cv_std': round(cv_scores.std(),4),
    'feature_importances': {k: round(v,4) for k,v in importances.items()},
    'classes': list(le.classes_)
}

cm = confusion_matrix(y_test, pred)
results['confusion_matrix'] = cm.tolist()
results['confusion_matrix_labels'] = list(le.classes_)

# 9. Multinomial logistic regression (standardized) for interpretable coefficients, ordinal target
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
logreg = LogisticRegression(max_iter=2000)
logreg.fit(Xs, df['perf_rank'])
# average abs coefficient across classes as importance proxy
coef_importance = np.mean(np.abs(logreg.coef_), axis=0)
logreg_importance = dict(sorted(zip(numeric_vars, coef_importance), key=lambda x: -x[1]))
results['logreg_importance'] = {k: round(float(v),4) for k,v in logreg_importance.items()}

# 10. Gender & age basic descriptives
results['gender_counts'] = df['gender'].value_counts().to_dict()
results['age_stats'] = {'mean': round(df['age'].mean(),2), 'min': int(df['age'].min()), 'max': int(df['age'].max())}
results['sport_counts'] = df['sport_category'].value_counts().to_dict()
results['n'] = len(df)

with open('results.json','w') as f:
    json.dump(results, f, indent=2, default=str)

print("DONE")
print(json.dumps(results['rf_model'], indent=2))
print("Top correlations:")
for k,v in sorted(corrs.items(), key=lambda x: -abs(x[1]['spearman_r']))[:8]:
    print(k, v)
