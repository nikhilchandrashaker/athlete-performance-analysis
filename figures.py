import os; os.makedirs('figures', exist_ok=True)
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix

plt.rcParams.update({'font.size': 12, 'font.family':'DejaVu Sans'})

# palette: Ocean/performance-science feel
PRIMARY = '#065A82'
SECOND = '#1C7293'
ACCENT = '#F96167'
NEUTRAL = '#6B7B8C'

df = pd.read_csv('student_sports_performance_Dataset.csv')
perf_order = ['Low','Moderate','High','Very High','Excellent']
df['performance_category'] = pd.Categorical(df['performance_category'], categories=perf_order, ordered=True)
df['perf_rank'] = df['performance_category'].cat.codes

numeric_vars = ['age','height_cm','weight_kg','bmi','running_time_sec','sprint_speed_kmh',
                'jumping_ability_cm','flexibility_cm','muscular_strength_kg','endurance_min',
                'heart_rate_bpm','training_duration_min','activity_intensity','exercise_frequency_per_week']

# --- Fig 1: Correlation heatmap of key vars ---
key_vars = ['sprint_speed_kmh','running_time_sec','muscular_strength_kg','jumping_ability_cm',
            'endurance_min','flexibility_cm','heart_rate_bpm','training_duration_min',
            'activity_intensity','exercise_frequency_per_week','perf_rank']
corr = df[key_vars].corr(method='spearman')
labels = ['Sprint speed','Running time','Muscular strength','Jumping ability','Endurance',
          'Flexibility','Heart rate','Training duration','Activity intensity','Exercise frequency','Performance rank']
fig, ax = plt.subplots(figsize=(9,7.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.6,
            xticklabels=labels, yticklabels=labels, cbar_kws={'label':"Spearman's rho"}, ax=ax,
            linewidths=0.5, linecolor='white')
plt.xticks(rotation=45, ha='right')
plt.title('Correlation Structure of Physical, Physiological, and Training Variables', fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('figures/fig1_correlation_heatmap.png', dpi=200)
plt.close()

# --- Fig 2: Boxplots of top 4 predictors by performance category ---
top_vars = [('sprint_speed_kmh','Sprint Speed (km/h)'), ('muscular_strength_kg','Muscular Strength (kg)'),
            ('jumping_ability_cm','Jumping Ability (cm)'), ('endurance_min','Endurance (min)')]
fig, axes = plt.subplots(2,2, figsize=(11,8.5))
for ax, (var,label) in zip(axes.flat, top_vars):
    sns.boxplot(data=df, x='performance_category', y=var, order=perf_order, ax=ax,
                palette=[NEUTRAL,'#4A90A4',SECOND,PRIMARY,'#0D3B54'], showfliers=False)
    ax.set_xlabel('')
    ax.set_ylabel(label)
    ax.tick_params(axis='x', rotation=20)
fig.suptitle('Physical Attributes Across Performance Categories', fontsize=15, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig('figures/fig2_boxplots_top_predictors.png', dpi=200, bbox_inches='tight')
plt.close()

# --- Fig 3: Feature importance (RF) ---
import json
r = json.load(open('results.json'))
fi = r['rf_model']['feature_importances']
names_map = {'sprint_speed_kmh':'Sprint speed','muscular_strength_kg':'Muscular strength','jumping_ability_cm':'Jumping ability',
             'endurance_min':'Endurance','running_time_sec':'Running time','flexibility_cm':'Flexibility',
             'height_cm':'Height','training_duration_min':'Training duration','weight_kg':'Weight',
             'activity_intensity':'Activity intensity','heart_rate_bpm':'Heart rate','bmi':'BMI',
             'exercise_frequency_per_week':'Exercise frequency','age':'Age'}
items = sorted(fi.items(), key=lambda x: x[1])
labels_ = [names_map[k] for k,v in items]
vals = [v for k,v in items]
colors = [PRIMARY if v>=0.1 else (SECOND if v>=0.04 else NEUTRAL) for v in vals]
fig, ax = plt.subplots(figsize=(9,7))
bars = ax.barh(labels_, vals, color=colors)
ax.set_xlabel('Relative Importance (Random Forest)')
ax.set_title('What Predicts Performance Category?\nFeature Importance from Random Forest Classifier', fontsize=13, fontweight='bold')
for b,v in zip(bars, vals):
    ax.text(v+0.003, b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=10)
ax.set_xlim(0, max(vals)*1.2)
sns.despine(left=True, bottom=False)
plt.tight_layout()
plt.savefig('figures/fig3_feature_importance.png', dpi=200)
plt.close()

# --- Fig 4: Sport-specific top predictor heatmap (shows uniformity) ---
predictors = ['sprint_speed_kmh','muscular_strength_kg','jumping_ability_cm','endurance_min','flexibility_cm']
pred_labels = ['Sprint speed','Muscular strength','Jumping ability','Endurance','Flexibility']
sports = sorted(df['sport_category'].unique())
mat = np.zeros((len(sports), len(predictors)))
for i,s in enumerate(sports):
    g = df[df['sport_category']==s]
    for j,p in enumerate(predictors):
        rho,_ = stats.spearmanr(g[p], g['perf_rank'])
        mat[i,j] = rho
fig, ax = plt.subplots(figsize=(8.5,6.5))
sns.heatmap(mat, annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=pred_labels, yticklabels=sports,
            cbar_kws={'label':"Spearman's rho with performance"}, ax=ax, linewidths=0.5, linecolor='white', vmin=0.15, vmax=0.65)
plt.title('Predictor–Performance Correlations Are Consistent Across Sports', fontsize=12.5, fontweight='bold', pad=12)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig('figures/fig4_sport_predictor_heatmap.png', dpi=200)
plt.close()

# --- Fig 5: Training dose effects (small but positive) ---
train_vars = [('training_duration_min','Training Duration\n(min/session)'), ('activity_intensity','Activity Intensity\n(0-10 scale)'),
              ('exercise_frequency_per_week','Exercise Frequency\n(days/week)')]
fig, axes = plt.subplots(1,3, figsize=(13,4.3))
for ax,(var,label) in zip(axes, train_vars):
    means = df.groupby('performance_category')[var].mean().reindex(perf_order)
    sems = df.groupby('performance_category')[var].sem().reindex(perf_order)
    ax.bar(perf_order, means, yerr=sems, color=SECOND, capsize=4, edgecolor='white')
    ax.set_title(label, fontsize=11)
    ax.tick_params(axis='x', rotation=25)
    ax.set_ylim(bottom=means.min()*0.85)
fig.suptitle('Training Variables Show Weak Positive Associations With Performance', fontsize=13.5, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('figures/fig5_training_dose.png', dpi=200, bbox_inches='tight')
plt.close()

# --- Fig 6: Confusion matrix ---
cm = np.array(r['confusion_matrix'])
cm_labels = r['confusion_matrix_labels']
# reorder to perf_order
order_idx = [cm_labels.index(p) for p in perf_order]
cm_ord = cm[np.ix_(order_idx, order_idx)]
cm_norm = cm_ord / cm_ord.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.5,6.5))
sns.heatmap(cm_norm, annot=True, fmt='.0%', cmap='Blues', xticklabels=perf_order, yticklabels=perf_order,
            cbar_kws={'label':'Proportion of actual class'}, ax=ax, linewidths=0.5, linecolor='white')
plt.xlabel('Predicted category')
plt.ylabel('Actual category')
plt.title('Random Forest Classification Accuracy by Performance Category', fontsize=12.5, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('figures/fig6_confusion_matrix.png', dpi=200)
plt.close()

# --- Fig 7: Speed-Strength-Jump scatter relationships ---
fig, axes = plt.subplots(1,3, figsize=(13,4.3))
pairs = [('sprint_speed_kmh','muscular_strength_kg','Sprint Speed (km/h)','Muscular Strength (kg)'),
         ('sprint_speed_kmh','jumping_ability_cm','Sprint Speed (km/h)','Jumping Ability (cm)'),
         ('muscular_strength_kg','jumping_ability_cm','Muscular Strength (kg)','Jumping Ability (cm)')]
sample = df.sample(2000, random_state=42)
for ax,(x,y,xl,yl) in zip(axes, pairs):
    ax.scatter(sample[x], sample[y], s=6, alpha=0.35, color=PRIMARY)
    rho,_ = stats.spearmanr(df[x], df[y])
    z = np.polyfit(df[x], df[y], 1)
    xs = np.linspace(df[x].min(), df[x].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=ACCENT, linewidth=2)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.set_title(f"rho = {rho:.2f}", fontsize=11)
fig.suptitle('The Speed–Strength–Power Connection', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('figures/fig7_speed_strength_scatter.png', dpi=200, bbox_inches='tight')
plt.close()

print("Figures done")
import os
print(os.listdir('figures'))
