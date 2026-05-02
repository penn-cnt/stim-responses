"""
inter_ML.py - Machine Learning for SOZ Prediction using Stimulation Response Features

This script performs patient-level Leave-One-Out Cross-Validation (LOOCV) to predict
seizure onset zone (SOZ) channels using:
  1. Baseline spike rate (24hr baseline)
  2. Change in spike rate when nearby channels are stimulated
  3. Change in nearby channels when this channel is stimulated

Key Features:
  - SMOTE applied WITHIN inner CV folds during grid search to prevent data leakage
  - Patient-level LOOCV for unbiased generalization estimates
  - DeLong's test for comparing AUC between models
  - Bootstrap confidence intervals for all metrics
  - Patient-specific ranking metrics (Top-1, Top-3, MRR, Macro-AP)

Usage:
  python inter_ML.py
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
from os.path import join as ospj
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
import matplotlib.colors as mcolors

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn import FunctionSampler

# Try to load custom style, fall back to default if not found
try:
    mpl.style.use('figures.mplstyle')
except:
    pass
MM_TO_IN = 1/25.4

# Project imports
from config import CONFIG
sys.path.append(CONFIG.tools_dir)
from iEEG_helper_functions import *


# Define color scheme and brain region categories
COLORS = {
    'teal': '#99D7CF',
    'yellow': '#E7CE95',
    'brown': '#8B4513',
    'gray': '#808080'
}

RESULTS_DIR = ospj(CONFIG.results_dir, "ML-results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_prepare_data():
    """Load the pre-merged de-identified analysis dataset."""
    print("Loading data...")
    analysis_data = pd.read_csv(ospj(CONFIG.dataset_dir, 'analysis_data_inter.csv'))
    print(f"Data shape: {analysis_data.shape}")
    return analysis_data

def create_ml_features(analysis_data):
    """
    Create ML feature dataframe with one row per channel per subject.

    Features:
    1. SOZ status (target)
    2. 24hr baseline spike rate
    3. Average change when stimulating nearby (<=40mm) to this recording channel
    4. Average change in nearby channels when THIS channel is stimulated
    """
    print("Creating ML features...")

    ml_data = []
    unique_channels = analysis_data[['subject_id', 'recording_ch_name']].drop_duplicates()
    n_total = len(unique_channels)

    for idx, (_, row) in enumerate(unique_channels.iterrows()):
        if (idx + 1) % 100 == 0:
            print(f"  Processing channel {idx + 1}/{n_total}...")

        subject_id = row['subject_id']
        rec_ch = row['recording_ch_name']

        ch_data = analysis_data[
            (analysis_data['subject_id'] == subject_id) &
            (analysis_data['recording_ch_name'] == rec_ch)
        ]

        features = {
            'subject_id': subject_id,
            'channel': rec_ch
        }

        # Feature 1: SOZ status (target)
        soz_status = ch_data['soz_rec_ch'].unique()
        features['soz_rec_ch'] = soz_status[0] if len(soz_status) == 1 else None

        # Feature 2: 24hr baseline spike rate
        baseline_rate = ch_data['24hr_baseline'].unique()
        features['baseline_spike_rate'] = baseline_rate[0] if len(baseline_rate) == 1 else None

        # Feature 3: Average change when stimulating nearby (<=40mm) to this recording channel
        nearby_stim_data = ch_data[ch_data['distance_mm'] <= 40]
        features['change_w_nearby_stim'] = nearby_stim_data['change_during_to_inter'].mean() if len(nearby_stim_data) > 0 else None

        # Feature 4: Average change in nearby channels when THIS channel is stimulated
        stim_by_this_ch_data = analysis_data[
            (analysis_data['subject_id'] == subject_id) &
            (analysis_data['stim_ch_first'] == rec_ch) &
            (analysis_data['distance_mm'] <= 40)
        ]
        features['change_in_nearby_w_stim_from_this_ch'] = stim_by_this_ch_data['change_during_to_inter'].mean() if len(stim_by_this_ch_data) > 0 else None

        ml_data.append(features)

    ml_df = pd.DataFrame(ml_data)

    print(f"\n{'='*80}")
    print(f"ML FEATURE DATAFRAME SUMMARY")
    print(f"{'='*80}")
    print(f"Total channels: {len(ml_df)}")
    print(f"Total subjects: {ml_df['subject_id'].nunique()}")
    print(f"\nFeature completeness:")
    print(f"  SOZ status available: {ml_df['soz_rec_ch'].notna().sum()} ({100*ml_df['soz_rec_ch'].notna().sum()/len(ml_df):.1f}%)")
    print(f"  Baseline spike rate available: {ml_df['baseline_spike_rate'].notna().sum()} ({100*ml_df['baseline_spike_rate'].notna().sum()/len(ml_df):.1f}%)")
    print(f"  Change w/ nearby stim available: {ml_df['change_w_nearby_stim'].notna().sum()} ({100*ml_df['change_w_nearby_stim'].notna().sum()/len(ml_df):.1f}%)")
    print(f"  Change when this ch stims nearby available: {ml_df['change_in_nearby_w_stim_from_this_ch'].notna().sum()} ({100*ml_df['change_in_nearby_w_stim_from_this_ch'].notna().sum()/len(ml_df):.1f}%)")

    print(f"\n{'='*80}")
    print("First 10 rows:")
    print(ml_df.head(10))

    print(f"\n{'='*80}")
    print("Feature statistics:")
    print(ml_df[['baseline_spike_rate', 'change_w_nearby_stim', 'change_in_nearby_w_stim_from_this_ch']].describe())

    return ml_df


# ============================================================================
# CONFIGURATION
# ============================================================================

full_features = ['baseline_spike_rate', 'change_w_nearby_stim', 'change_in_nearby_w_stim_from_this_ch']
baseline_features = ['baseline_spike_rate']
id_cols = ['subject_id', 'channel', 'soz_rec_ch']

rf_param_grid = {
    'n_estimators': [50, 300, 500, 1000],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'random_state': [42]
}

N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# ============================================================================
# DELONG'S TEST FOR COMPARING AUC
# ============================================================================

def compute_midrank(x):
    """Compute midranks for DeLong test."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong AUC variance calculation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])

    aucs = np.zeros(k, dtype=float)
    score = np.zeros([k, m + n], dtype=float)

    for r in range(k):
        aucs[r] = (np.sum(tz[r, :m]) - tx[r, :].sum()) / (m * n)
        score[r, :m] = (tz[r, :m] - tx[r, :]) / n
        score[r, m:] = 1.0 - (tz[r, m:] - ty[r, :]) / m

    return aucs, score

def delong_test(y_true, y_pred1, y_pred2):
    """
    Perform DeLong's test comparing two AUC values.
    Returns: auc1, auc2, z_statistic, p_value
    """
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)

    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())

    predictions_sorted = np.vstack([y_pred1, y_pred2])[:, order]

    aucs, scores = fastDeLong(predictions_sorted, label_1_count)

    m = label_1_count
    n = len(y_true) - m

    cov = np.cov(scores)

    contrast = np.array([1, -1])
    var_diff = contrast @ cov @ contrast

    if var_diff <= 0:
        return aucs[0], aucs[1], 0, 1.0

    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return aucs[0], aucs[1], z, p_value

# ============================================================================
# PATIENT-LEVEL RANKING METRICS
# ============================================================================

def compute_patient_ranking_metrics(patient_results):
    """
    Compute Top-1, Top-3, MRR, and macro-AP from patient-level predictions.

    patient_results: list of dicts with 'y_true', 'y_prob', 'patient_id'
    """
    top1_hits = []
    top3_hits = []
    reciprocal_ranks = []
    patient_aps = []

    for pr in patient_results:
        y_true = np.array(pr['y_true'])
        y_prob = np.array(pr['y_prob'])

        if y_true.sum() == 0:
            continue

        ranks = np.argsort(-y_prob)
        soz_positions = np.where(y_true[ranks] == 1)[0]

        if len(soz_positions) > 0:
            best_soz_rank = soz_positions[0] + 1

            top1_hits.append(1 if best_soz_rank == 1 else 0)
            top3_hits.append(1 if best_soz_rank <= 3 else 0)
            reciprocal_ranks.append(1.0 / best_soz_rank)

            if len(np.unique(y_true)) == 2:
                patient_aps.append(average_precision_score(y_true, y_prob))

    return {
        'top1': np.mean(top1_hits) if top1_hits else 0,
        'top3': np.mean(top3_hits) if top3_hits else 0,
        'mrr': np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
        'macro_ap': np.mean(patient_aps) if patient_aps else 0,
        'n_patients': len(top1_hits)
    }

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_patient_metrics(patient_results, n_bootstrap=1000, seed=42):
    """Compute bootstrap CIs for all metrics at patient level."""
    rng = np.random.RandomState(seed)
    n_patients = len(patient_results)

    boot_auc = []
    boot_ap = []
    boot_top1 = []
    boot_top3 = []
    boot_mrr = []
    boot_macro_ap = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_patients, size=n_patients, replace=True)
        boot_patients = [patient_results[i] for i in idx]

        all_y_true = np.concatenate([p['y_true'] for p in boot_patients])
        all_y_prob = np.concatenate([p['y_prob'] for p in boot_patients])

        if len(np.unique(all_y_true)) == 2:
            boot_auc.append(roc_auc_score(all_y_true, all_y_prob))
            boot_ap.append(average_precision_score(all_y_true, all_y_prob))

        ranking = compute_patient_ranking_metrics(boot_patients)
        boot_top1.append(ranking['top1'])
        boot_top3.append(ranking['top3'])
        boot_mrr.append(ranking['mrr'])
        boot_macro_ap.append(ranking['macro_ap'])

    def ci(arr):
        return np.percentile(arr, [2.5, 97.5])

    return {
        'auc_ci': ci(boot_auc),
        'ap_ci': ci(boot_ap),
        'top1_ci': ci(boot_top1),
        'top3_ci': ci(boot_top3),
        'mrr_ci': ci(boot_mrr),
        'macro_ap_ci': ci(boot_macro_ap)
    }

# ============================================================================
# MAIN LOOCV FUNCTION WITH PATIENT-LEVEL TRACKING
# ============================================================================
def _prefix_param_grid_for_pipeline(param_grid, step_name="classifier"):
    """Ensure params target the pipeline's classifier step (classifier__param)."""
    if param_grid is None:
        return None
    prefixed = {}
    for k, v in param_grid.items():
        if "__" in k:
            prefixed[k] = v
        else:
            prefixed[f"{step_name}__{k}"] = v
    return prefixed


def _safe_smote_resample(X, y, random_state=42, k_max=5):
    """
    Apply SMOTE safely per split:
    - If too few minority samples, return (X, y) unchanged
    - Otherwise set k_neighbors = min(k_max, minority_count - 1)
    """
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return X, y

    counts = np.bincount(y)
    if len(counts) < 2:
        return X, y

    minority_count = counts.min()
    if minority_count < 2:
        return X, y

    k = min(k_max, minority_count - 1)
    sm = SMOTE(random_state=random_state, k_neighbors=k)
    return sm.fit_resample(X, y)

def run_loocv_model_full(ml_df, feature_cols, classifier, param_grid,
                         model_name="Model", use_grid_search=True,
                         per_fold_params=None):
    """
    Run patient-level LOOCV with full metrics and patient-level tracking.

    SMOTE is applied WITHIN each training fold to prevent data leakage.
    Also collects feature importances and best params across folds.

    Parameters:
    -----------
    per_fold_params : list of dicts, optional
        If provided, skips grid search and uses these pre-determined params
        per fold (keyed by patient_id). Used to reuse full model's optimal
        hyperparameters when training the baseline model.
    """

    ml_clean = ml_df[feature_cols + id_cols].copy().dropna()
    ml_clean['soz_rec_ch'] = ml_clean['soz_rec_ch'].astype(int)
    unique_patients = ml_clean['subject_id'].unique()
    n_patients = len(unique_patients)

    patient_results = []
    feature_importances_list = []
    best_params_per_fold = {}

    per_fold_params_lookup = {}
    if per_fold_params is not None:
        per_fold_params_lookup = {entry['patient_id']: entry['params']
                                  for entry in per_fold_params}

    safe_smote = FunctionSampler(
        func=lambda X, y: _safe_smote_resample(X, y, random_state=42, k_max=5)
    )

    param_grid_prefixed = _prefix_param_grid_for_pipeline(param_grid, step_name="classifier")

    print(f"  Running LOOCV on {n_patients} subjects...")
    for i, test_patient in enumerate(unique_patients):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Subject {i + 1}/{n_patients}...")
        train_data = ml_clean[ml_clean['subject_id'] != test_patient]
        test_data = ml_clean[ml_clean['subject_id'] == test_patient]

        if len(test_data) == 0:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data['soz_rec_ch'].values.astype(int)
        g_train = train_data['subject_id'].values

        X_test  = test_data[feature_cols].values
        y_test  = test_data['soz_rec_ch'].values.astype(int)

        if len(np.unique(y_train)) < 2:
            continue

        pipe = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('smote', safe_smote),
            ('classifier', classifier)
        ])

        if test_patient in per_fold_params_lookup:
            best_clf_params = per_fold_params_lookup[test_patient]
            classifier.set_params(**best_clf_params)
            pipe = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('smote', safe_smote),
                ('classifier', classifier)
            ])
            model = pipe
            model.fit(X_train, y_train)
            best_params_per_fold[test_patient] = best_clf_params

        elif use_grid_search and param_grid_prefixed is not None:
            n_train_groups = len(np.unique(g_train))
            if n_train_groups < 2:
                continue

            n_splits = min(5, n_train_groups)

            grid_search = GridSearchCV(
                pipe, param_grid_prefixed, cv=n_splits,
                scoring='average_precision', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train, groups=g_train)
            model = grid_search.best_estimator_

            raw_best = grid_search.best_params_
            clean_params = {k.replace('classifier__', ''): v
                            for k, v in raw_best.items()}
            best_params_per_fold[test_patient] = clean_params

        else:
            model = pipe
            model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        try:
            clf = model.named_steps['classifier']
            if hasattr(clf, 'feature_importances_'):
                feature_importances_list.append(clf.feature_importances_)
        except:
            pass

        patient_results.append({
            'patient_id': test_patient,
            'y_true': y_test,
            'y_prob': y_prob
        })

    all_y_true = np.concatenate([p['y_true'] for p in patient_results])
    all_y_prob = np.concatenate([p['y_prob'] for p in patient_results])

    auc_roc = roc_auc_score(all_y_true, all_y_prob)
    auc_prc = average_precision_score(all_y_true, all_y_prob)
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
    prec, rec, _ = precision_recall_curve(all_y_true, all_y_prob)

    ranking_metrics = compute_patient_ranking_metrics(patient_results)

    print(f"  Computing bootstrap CIs for {model_name}...")
    bootstrap_cis = bootstrap_patient_metrics(patient_results, N_BOOTSTRAP, RANDOM_SEED)

    feature_importance_stats = None
    if feature_importances_list:
        fi_array = np.array(feature_importances_list)
        feature_importance_stats = {
            'feature_names': feature_cols,
            'mean': np.mean(fi_array, axis=0),
            'std': np.std(fi_array, axis=0),
            'all_importances': fi_array
        }

    return {
        'name': model_name,
        'auc_roc': auc_roc,
        'auc_prc': auc_prc,
        'fpr': fpr,
        'tpr': tpr,
        'prec': prec,
        'rec': rec,
        'y_true': all_y_true,
        'y_prob': all_y_prob,
        'baseline_prevalence': all_y_true.mean(),
        'patient_results': patient_results,
        'ranking': ranking_metrics,
        'bootstrap_cis': bootstrap_cis,
        'feature_importances': feature_importance_stats,
        'best_params_per_fold': best_params_per_fold
    }

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def bootstrap_curve_ci(patient_results, n_bootstrap=500, seed=42):
    """
    Bootstrap confidence intervals for ROC and PRC curves.
    Returns interpolated curves at common x-axis points for plotting shaded regions.
    """
    rng = np.random.RandomState(seed)
    n_patients = len(patient_results)

    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    tprs = []
    precs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_patients, size=n_patients, replace=True)
        boot_patients = [patient_results[i] for i in idx]

        all_y_true = np.concatenate([p['y_true'] for p in boot_patients])
        all_y_prob = np.concatenate([p['y_prob'] for p in boot_patients])

        if len(np.unique(all_y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        prec, rec, _ = precision_recall_curve(all_y_true, all_y_prob)
        rec_rev = rec[::-1]
        prec_rev = prec[::-1]
        interp_prec = np.interp(mean_recall, rec_rev, prec_rev)
        precs.append(interp_prec)

    tprs = np.array(tprs)
    precs = np.array(precs)

    return {
        'mean_fpr': mean_fpr,
        'tpr_lower': np.percentile(tprs, 2.5, axis=0),
        'tpr_upper': np.percentile(tprs, 97.5, axis=0),
        'mean_recall': mean_recall,
        'prec_lower': np.percentile(precs, 2.5, axis=0),
        'prec_upper': np.percentile(precs, 97.5, axis=0),
    }


def plot_comparison_with_ci(full_model, baseline_model, delong_p, title_suffix="", save_dir=None):
    """Create clean ROC and PRC comparison plots with shaded CI regions and DeLong's p-value annotated."""

    print("  Computing curve confidence intervals (this may take a moment)...")

    full_curve_ci = bootstrap_curve_ci(full_model['patient_results'], n_bootstrap=500, seed=RANDOM_SEED)
    base_curve_ci = bootstrap_curve_ci(baseline_model['patient_results'], n_bootstrap=500, seed=RANDOM_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    full_color = '#2c7bb6'
    baseline_color = '#d7191c'

    ax = axes[0]

    full_ci = full_model['bootstrap_cis']['auc_ci']
    base_ci = baseline_model['bootstrap_cis']['auc_ci']

    ax.fill_between(full_curve_ci['mean_fpr'],
                    full_curve_ci['tpr_lower'],
                    full_curve_ci['tpr_upper'],
                    color=full_color, alpha=0.2)
    ax.fill_between(base_curve_ci['mean_fpr'],
                    base_curve_ci['tpr_lower'],
                    base_curve_ci['tpr_upper'],
                    color=baseline_color, alpha=0.2)

    ax.plot(full_model['fpr'], full_model['tpr'], color=full_color, linewidth=2.5,
            label=f"Full: AUC={full_model['auc_roc']:.3f} [{full_ci[0]:.3f}-{full_ci[1]:.3f}]")
    ax.plot(baseline_model['fpr'], baseline_model['tpr'], color=baseline_color, linewidth=2.5,
            label=f"Baseline: AUC={baseline_model['auc_roc']:.3f} [{base_ci[0]:.3f}-{base_ci[1]:.3f}]")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Chance')

    sig_str = f"DeLong p = {delong_p:.4f}"
    if delong_p < 0.001:
        sig_str += " ***"
    elif delong_p < 0.01:
        sig_str += " **"
    elif delong_p < 0.05:
        sig_str += " *"
    ax.text(0.55, 0.15, sig_str, transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve{title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, frameon=True, fancybox=False, edgecolor='gray')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[1]
    prevalence = full_model['baseline_prevalence']

    full_ap_ci = full_model['bootstrap_cis']['ap_ci']
    base_ap_ci = baseline_model['bootstrap_cis']['ap_ci']

    ax.fill_between(full_curve_ci['mean_recall'],
                    full_curve_ci['prec_lower'],
                    full_curve_ci['prec_upper'],
                    color=full_color, alpha=0.2)
    ax.fill_between(base_curve_ci['mean_recall'],
                    base_curve_ci['prec_lower'],
                    base_curve_ci['prec_upper'],
                    color=baseline_color, alpha=0.2)

    ax.plot(full_model['rec'], full_model['prec'], color=full_color, linewidth=2.5,
            label=f"Full: AP={full_model['auc_prc']:.3f} [{full_ap_ci[0]:.3f}-{full_ap_ci[1]:.3f}]")
    ax.plot(baseline_model['rec'], baseline_model['prec'], color=baseline_color, linewidth=2.5,
            label=f"Baseline: AP={baseline_model['auc_prc']:.3f} [{base_ap_ci[0]:.3f}-{base_ap_ci[1]:.3f}]")
    ax.axhline(y=prevalence, color='k', linestyle='--', linewidth=1, alpha=0.5,
               label=f'No-skill (prevalence={prevalence:.2f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve{title_suffix}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, frameon=True, fancybox=False, edgecolor='gray')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_dir:
        save_path = ospj(save_dir, 'roc_prc_comparison.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    return fig


def plot_patient_details(full_model, baseline_model, model_name="Model", save_dir=None):
    """
    Create patient-specific detail plots showing:
    1. Per-patient AUC comparison (Full vs Baseline)
    2. Best SOZ rank distribution
    3. Patient-level probability distributions for SOZ vs non-SOZ channels
    """

    full_pr = full_model['patient_results']
    base_pr = baseline_model['patient_results']

    patient_aucs_full = []
    patient_aucs_base = []
    patient_ids = []

    for fp, bp in zip(full_pr, base_pr):
        if len(np.unique(fp['y_true'])) == 2:
            patient_ids.append(fp['patient_id'])
            patient_aucs_full.append(roc_auc_score(fp['y_true'], fp['y_prob']))
            patient_aucs_base.append(roc_auc_score(bp['y_true'], bp['y_prob']))

    best_ranks_full = []
    best_ranks_base = []

    for fp, bp in zip(full_pr, base_pr):
        y_true = np.array(fp['y_true'])
        if y_true.sum() == 0:
            continue

        ranks_full = np.argsort(-np.array(fp['y_prob']))
        soz_pos_full = np.where(y_true[ranks_full] == 1)[0]
        if len(soz_pos_full) > 0:
            best_ranks_full.append(soz_pos_full[0] + 1)

        ranks_base = np.argsort(-np.array(bp['y_prob']))
        soz_pos_base = np.where(y_true[ranks_base] == 1)[0]
        if len(soz_pos_base) > 0:
            best_ranks_base.append(soz_pos_base[0] + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(patient_aucs_base, patient_aucs_full, alpha=0.7, s=50, c='#2c7bb6', edgecolors='k', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Baseline Model AUC', fontsize=11)
    ax.set_ylabel(f'{model_name} AUC', fontsize=11)
    ax.set_title('Per-Subject AUC: Full vs Baseline', fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])

    n_improved = sum(1 for f, b in zip(patient_aucs_full, patient_aucs_base) if f > b)
    n_total = len(patient_aucs_full)
    ax.text(0.05, 0.95, f'{n_improved}/{n_total} subjects improved',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    auc_diff = np.array(patient_aucs_full) - np.array(patient_aucs_base)
    sorted_idx = np.argsort(auc_diff)
    colors = ['#d7191c' if d < 0 else '#2c7bb6' for d in auc_diff[sorted_idx]]

    ax.barh(range(len(auc_diff)), auc_diff[sorted_idx], color=colors, edgecolor='k', linewidth=0.3)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.set_xlabel('AUC Difference (Full - Baseline)', fontsize=11)
    ax.set_ylabel('Subject (sorted)', fontsize=11)
    ax.set_title('Per-Subject AUC Change', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    mean_diff = np.mean(auc_diff)
    ax.axvline(x=mean_diff, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_diff:.3f}')
    ax.legend(loc='lower right')

    ax = axes[1, 0]
    bins = np.arange(0.5, max(max(best_ranks_full), max(best_ranks_base)) + 1.5, 1)

    ax.hist(best_ranks_base, bins=bins, alpha=0.6, label='Baseline', color='#d7191c', edgecolor='k')
    ax.hist(best_ranks_full, bins=bins, alpha=0.6, label=f'{model_name}', color='#2c7bb6', edgecolor='k')
    ax.set_xlabel('Best SOZ Channel Rank', fontsize=11)
    ax.set_ylabel('Number of Subjects', fontsize=11)
    ax.set_title('Distribution of Best SOZ Rank', fontsize=12, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    median_full = np.median(best_ranks_full)
    median_base = np.median(best_ranks_base)
    ax.axvline(x=median_full, color='#2c7bb6', linestyle='--', linewidth=2)
    ax.axvline(x=median_base, color='#d7191c', linestyle='--', linewidth=2)
    ax.text(0.95, 0.95, f'Median: Full={median_full:.1f}, Base={median_base:.1f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax = axes[1, 1]
    soz_probs_full = []
    nonsoz_probs_full = []

    for pr in full_pr:
        y_true = np.array(pr['y_true'])
        y_prob = np.array(pr['y_prob'])
        soz_probs_full.extend(y_prob[y_true == 1])
        nonsoz_probs_full.extend(y_prob[y_true == 0])

    violin_data = [nonsoz_probs_full, soz_probs_full]
    parts = ax.violinplot(violin_data, positions=[0, 1], showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#2c7bb6')
        pc.set_alpha(0.7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-SOZ', 'SOZ'])
    ax.set_ylabel('Predicted Probability', fontsize=11)
    ax.set_title(f'{model_name}: Prediction Distribution by SOZ Status', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    from scipy.stats import mannwhitneyu
    stat, pval = mannwhitneyu(soz_probs_full, nonsoz_probs_full, alternative='greater')
    ax.text(0.5, 0.95, f'Mann-Whitney p = {pval:.2e}',
            transform=ax.transAxes, fontsize=10, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_dir:
        save_path = ospj(save_dir, 'patient_details.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    return fig


def plot_patient_ranking_comparison(full_model, baseline_model, model_name="Model", save_dir=None):
    """
    Detailed patient-level ranking visualization showing how each subject's
    SOZ channels are ranked by both models.
    """
    full_pr = full_model['patient_results']
    base_pr = baseline_model['patient_results']

    fig, ax = plt.subplots(figsize=(12, 8))

    patient_data = []
    for fp, bp in zip(full_pr, base_pr):
        y_true = np.array(fp['y_true'])
        n_soz = y_true.sum()
        if n_soz == 0:
            continue

        n_channels = len(y_true)

        ranks_full = np.argsort(-np.array(fp['y_prob']))
        soz_pos_full = np.where(y_true[ranks_full] == 1)[0]
        best_rank_full = soz_pos_full[0] + 1 if len(soz_pos_full) > 0 else n_channels

        ranks_base = np.argsort(-np.array(bp['y_prob']))
        soz_pos_base = np.where(y_true[ranks_base] == 1)[0]
        best_rank_base = soz_pos_base[0] + 1 if len(soz_pos_base) > 0 else n_channels

        patient_data.append({
            'patient_id': fp['patient_id'],
            'n_channels': n_channels,
            'n_soz': n_soz,
            'rank_full': best_rank_full,
            'rank_base': best_rank_base,
            'percentile_full': best_rank_full / n_channels * 100,
            'percentile_base': best_rank_base / n_channels * 100
        })

    patient_data.sort(key=lambda x: x['rank_base'] - x['rank_full'], reverse=True)

    y_pos = np.arange(len(patient_data))

    for i, pd_row in enumerate(patient_data):
        color = '#2c7bb6' if pd_row['rank_full'] < pd_row['rank_base'] else '#d7191c' if pd_row['rank_full'] > pd_row['rank_base'] else 'gray'
        ax.plot([pd_row['rank_base'], pd_row['rank_full']], [i, i], color=color, linewidth=1.5, alpha=0.7)
        ax.scatter(pd_row['rank_base'], i, color='#d7191c', s=60, zorder=5, edgecolors='k', linewidth=0.5)
        ax.scatter(pd_row['rank_full'], i, color='#2c7bb6', s=60, zorder=5, edgecolors='k', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{pd_row['patient_id']}" for pd_row in patient_data], fontsize=8)
    ax.set_xlabel('Best SOZ Channel Rank (lower = better)', fontsize=11)
    ax.set_ylabel('Subject', fontsize=11)
    ax.set_title(f'Subject-Level SOZ Ranking: {model_name} vs Baseline', fontsize=12, fontweight='bold')

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d7191c', markersize=10, label='Baseline'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c7bb6', markersize=10, label=model_name),
        Line2D([0], [0], color='#2c7bb6', linewidth=2, label='Improved'),
        Line2D([0], [0], color='#d7191c', linewidth=2, label='Worsened'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(x=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Rank 1')
    ax.axvline(x=3, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Rank 3')

    plt.tight_layout()

    if save_dir:
        save_path = ospj(save_dir, 'patient_ranking_comparison.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()
    return fig


def plot_feature_importance(model_results, model_name="Model", save_dir=None):
    """
    Create a feature importance plot showing mean importance across LOOCV folds
    with error bars representing standard deviation.
    """

    fi_stats = model_results.get('feature_importances')

    if fi_stats is None:
        print(f"  No feature importances available for {model_name}")
        return None

    feature_names = fi_stats['feature_names']
    mean_importance = fi_stats['mean']
    std_importance = fi_stats['std']

    name_mapping = {
        'baseline_spike_rate': '24hr Baseline\nSpike Rate',
        'change_w_nearby_stim': 'Change When\nNearby Stimulated',
        'change_in_nearby_w_stim_from_this_ch': 'Change in Nearby\nWhen This Ch Stims'
    }
    display_names = [name_mapping.get(n, n) for n in feature_names]

    sorted_idx = np.argsort(mean_importance)[::-1]
    sorted_names = [display_names[i] for i in sorted_idx]
    sorted_means = mean_importance[sorted_idx]
    sorted_stds = std_importance[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_names)))[::-1]

    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_means, xerr=sorted_stds,
                   color=colors, edgecolor='black', linewidth=0.5,
                   capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlabel('Feature Importance (Mean ± SD across LOOCV folds)', fontsize=12)
    ax.set_title(f'{model_name}: Feature Importance for SOZ Prediction', fontsize=14, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, (mean, std) in enumerate(zip(sorted_means, sorted_stds)):
        ax.text(mean + std + 0.01, i, f'{mean:.3f}', va='center', fontsize=10)

    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(left=0)

    plt.tight_layout()

    if save_dir:
        save_path = ospj(save_dir, 'feature_importance.pdf')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.show()

    print(f"\n  Feature Importance Summary ({model_name}):")
    print(f"  {'Feature':<45} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*65}")
    for name, mean, std in zip(sorted_names, sorted_means, sorted_stds):
        clean_name = name.replace('\n', ' ')
        print(f"  {clean_name:<45} {mean:>10.4f} {std:>10.4f}")

    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function for SOZ prediction ML analysis."""

    analysis_data = load_and_prepare_data()

    ml_df = create_ml_features(analysis_data)

    all_features = full_features + id_cols
    ml_df_clean = ml_df[all_features].dropna().copy()
    print(f"\nAfter filtering for complete cases: {len(ml_df_clean)} channels, {ml_df_clean['subject_id'].nunique()} subjects")

    print("\n" + "=" * 70)
    print("RUNNING MODELS WITH INDEPENDENT HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    print("\nRunning RF Full Model (3 features) with independent grid search...")
    rf_full_model = run_loocv_model_full(
        ml_df_clean, full_features,
        RandomForestClassifier(),
        rf_param_grid,
        model_name="RF Full",
        use_grid_search=True
    )

    print("\nRunning Baseline Model (spike rate only) with independent grid search...")
    baseline_model = run_loocv_model_full(
        ml_df_clean, baseline_features,
        RandomForestClassifier(),
        rf_param_grid,
        model_name="Baseline",
        use_grid_search=True
    )

    print("\nModel training complete!\n")

    print("=" * 70)
    print("DELONG'S TEST: Comparing AUC Curves")
    print("=" * 70)

    auc1, auc2, z_rf, p_rf = delong_test(
        rf_full_model['y_true'],
        rf_full_model['y_prob'],
        baseline_model['y_prob']
    )
    print(f"\nRandom Forest Full vs Baseline:")
    print(f"  AUC Full: {auc1:.4f}, AUC Baseline: {auc2:.4f}")
    print(f"  Z-statistic: {z_rf:.4f}")
    print(f"  P-value: {p_rf:.4f} {'***' if p_rf < 0.001 else '**' if p_rf < 0.01 else '*' if p_rf < 0.05 else ''}")

    print("\n" + "=" * 70)
    print("RANDOM FOREST COMPARISON - ROC & PRC")
    print("=" * 70)
    plot_comparison_with_ci(rf_full_model, baseline_model, p_rf, " - Random Forest", save_dir=RESULTS_DIR)

    print("\n" + "=" * 70)
    print("RANDOM FOREST - PATIENT-SPECIFIC DETAILS")
    print("=" * 70)
    plot_patient_details(rf_full_model, baseline_model, "RF Full", save_dir=RESULTS_DIR)

    print("\n" + "=" * 70)
    print("RANDOM FOREST - PATIENT RANKING COMPARISON")
    print("=" * 70)
    plot_patient_ranking_comparison(rf_full_model, baseline_model, "RF Full", save_dir=RESULTS_DIR)

    print("\n" + "=" * 70)
    print("RANDOM FOREST - FEATURE IMPORTANCE")
    print("=" * 70)
    plot_feature_importance(rf_full_model, "RF Full Model", save_dir=RESULTS_DIR)

    def format_ci(val, ci):
        return f"{val:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]"

    print("\n" + "=" * 70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Baseline (Spike Rate)':<28} {'RF Full (3 Features)':<28}")
    print("-" * 70)

    print(f"{'AUC-ROC':<20} "
          f"{format_ci(baseline_model['auc_roc'], baseline_model['bootstrap_cis']['auc_ci']):<28} "
          f"{format_ci(rf_full_model['auc_roc'], rf_full_model['bootstrap_cis']['auc_ci']):<28}")

    print(f"{'AUC-PRC':<20} "
          f"{format_ci(baseline_model['auc_prc'], baseline_model['bootstrap_cis']['ap_ci']):<28} "
          f"{format_ci(rf_full_model['auc_prc'], rf_full_model['bootstrap_cis']['ap_ci']):<28}")

    print("-" * 70)
    print("PATIENT-LEVEL RANKING METRICS")
    print("-" * 70)

    print(f"{'Top-1 Accuracy':<20} "
          f"{format_ci(baseline_model['ranking']['top1'], baseline_model['bootstrap_cis']['top1_ci']):<28} "
          f"{format_ci(rf_full_model['ranking']['top1'], rf_full_model['bootstrap_cis']['top1_ci']):<28}")

    print(f"{'Top-3 Accuracy':<20} "
          f"{format_ci(baseline_model['ranking']['top3'], baseline_model['bootstrap_cis']['top3_ci']):<28} "
          f"{format_ci(rf_full_model['ranking']['top3'], rf_full_model['bootstrap_cis']['top3_ci']):<28}")

    print(f"{'MRR':<20} "
          f"{format_ci(baseline_model['ranking']['mrr'], baseline_model['bootstrap_cis']['mrr_ci']):<28} "
          f"{format_ci(rf_full_model['ranking']['mrr'], rf_full_model['bootstrap_cis']['mrr_ci']):<28}")

    print(f"{'Macro-AP':<20} "
          f"{format_ci(baseline_model['ranking']['macro_ap'], baseline_model['bootstrap_cis']['macro_ap_ci']):<28} "
          f"{format_ci(rf_full_model['ranking']['macro_ap'], rf_full_model['bootstrap_cis']['macro_ap_ci']):<28}")

    print("-" * 70)
    print(f"{'N Subjects':<20} {baseline_model['ranking']['n_patients']:<28} "
          f"{rf_full_model['ranking']['n_patients']:<28}")

    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (DeLong's Test)")
    print("=" * 70)
    print(f"\nRF Full vs Baseline:  z = {z_rf:+.3f}, p = {p_rf:.4f} {'(significant)' if p_rf < 0.05 else '(not significant)'}")
    print("\nNote: * p<0.05, ** p<0.01, *** p<0.001")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = ospj(RESULTS_DIR, 'model_comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON - SOZ PREDICTION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric':<20} {'Baseline (Spike Rate)':<28} {'RF Full (3 Features)':<28}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'AUC-ROC':<20} {format_ci(baseline_model['auc_roc'], baseline_model['bootstrap_cis']['auc_ci']):<28} {format_ci(rf_full_model['auc_roc'], rf_full_model['bootstrap_cis']['auc_ci']):<28}\n")
        f.write(f"{'AUC-PRC':<20} {format_ci(baseline_model['auc_prc'], baseline_model['bootstrap_cis']['ap_ci']):<28} {format_ci(rf_full_model['auc_prc'], rf_full_model['bootstrap_cis']['ap_ci']):<28}\n")
        f.write("-" * 70 + "\n")
        f.write("PATIENT-LEVEL RANKING METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Top-1 Accuracy':<20} {format_ci(baseline_model['ranking']['top1'], baseline_model['bootstrap_cis']['top1_ci']):<28} {format_ci(rf_full_model['ranking']['top1'], rf_full_model['bootstrap_cis']['top1_ci']):<28}\n")
        f.write(f"{'Top-3 Accuracy':<20} {format_ci(baseline_model['ranking']['top3'], baseline_model['bootstrap_cis']['top3_ci']):<28} {format_ci(rf_full_model['ranking']['top3'], rf_full_model['bootstrap_cis']['top3_ci']):<28}\n")
        f.write(f"{'MRR':<20} {format_ci(baseline_model['ranking']['mrr'], baseline_model['bootstrap_cis']['mrr_ci']):<28} {format_ci(rf_full_model['ranking']['mrr'], rf_full_model['bootstrap_cis']['mrr_ci']):<28}\n")
        f.write(f"{'Macro-AP':<20} {format_ci(baseline_model['ranking']['macro_ap'], baseline_model['bootstrap_cis']['macro_ap_ci']):<28} {format_ci(rf_full_model['ranking']['macro_ap'], rf_full_model['bootstrap_cis']['macro_ap_ci']):<28}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'N Subjects':<20} {baseline_model['ranking']['n_patients']:<28} {rf_full_model['ranking']['n_patients']:<28}\n")
        f.write("\n" + "=" * 70 + "\n")
        f.write("STATISTICAL SIGNIFICANCE (DeLong's Test)\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nRF Full vs Baseline:  z = {z_rf:+.3f}, p = {p_rf:.4f} {'(significant)' if p_rf < 0.05 else '(not significant)'}\n")
        f.write("\nNote: * p<0.05, ** p<0.01, *** p<0.001\n")
    print(f"Saved summary: {summary_path}")

    ml_df_path = ospj(RESULTS_DIR, 'ml_features.csv')
    ml_df_clean.to_csv(ml_df_path, index=False)
    print(f"Saved ML features: {ml_df_path}")

    print(f"\nAll results saved to: {RESULTS_DIR}")

    return {
        'baseline_model': baseline_model,
        'rf_full_model': rf_full_model,
        'delong_rf': {'z': z_rf, 'p': p_rf},
        'ml_df': ml_df_clean
    }


if __name__ == "__main__":
    results = main()
