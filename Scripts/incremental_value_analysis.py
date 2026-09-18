# -*- coding: utf-8 -*-

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
os.environ['SCIPY_ARRAY_API'] = '1' 

# Set plotting style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# NRI and IDI Functions
def category_free_nri(y_true, prob_old, prob_new):
    '''
    
    Category‑free NRI
    '''
    y_true = np.array(y_true)
    prob_old = np.array(prob_old)
    prob_new = np.array(prob_new)

    event_idx = y_true == 1
    nonevent_idx = y_true == 0

    event_up = np.mean(prob_new[event_idx] > prob_old[event_idx])
    event_down = np.mean(prob_new[event_idx] < prob_old[event_idx])
    nri_event = event_up - event_down

    nonevent_up = np.mean(prob_new[nonevent_idx] > prob_old[nonevent_idx])
    nonevent_down = np.mean(prob_new[nonevent_idx] < prob_old[nonevent_idx])
    nri_nonevent = nonevent_down - nonevent_up

    cf_nri = nri_event + nri_nonevent
    return {
        'cfNRI': cf_nri,
        'NRI_event': nri_event,
        'NRI_nonevent': nri_nonevent,
        'event_up': event_up,
        'event_down': event_down,
        'nonevent_up': nonevent_up,
        'nonevent_down': nonevent_down
    }

def categorical_nri(y_true, prob_old, prob_new, cutoffs):
    '''
    
    Category-based Net Reclassification Improvement, NRI
    '''
    def categorize(prob, cuts):
        cats = np.zeros(len(prob), dtype=int)
        for c in cuts:
            cats += (prob > c).astype(int)
        return cats

    cats_old = categorize(prob_old, cutoffs)
    cats_new = categorize(prob_new, cutoffs)

    event = y_true == 1
    nonevent = ~event

    correct_up = np.sum((cats_new > cats_old) & event)
    wrong_down = np.sum((cats_new < cats_old) & event)
    nri_event = (correct_up - wrong_down) / np.sum(event) if np.sum(event) > 0 else 0

    correct_down = np.sum((cats_new < cats_old) & nonevent)
    wrong_up = np.sum((cats_new > cats_old) & nonevent)
    nri_nonevent = (correct_down - wrong_up) / np.sum(nonevent) if np.sum(nonevent) > 0 else 0

    reclass_table = np.zeros((len(cutoffs)+1, len(cutoffs)+1), dtype=int)
    for old_cat in range(len(cutoffs)+1):
        for new_cat in range(len(cutoffs)+1):
            reclass_table[old_cat, new_cat] = np.sum((cats_old == old_cat) & (cats_new == new_cat))

    return {
        'categorical NRI': nri_event + nri_nonevent,
        'NRI_event': nri_event,
        'NRI_nonevent': nri_nonevent,
        'reclassification_table': reclass_table.tolist()
    }

def print_reclass_by_event(y_true, prob_old, prob_new, cutoffs):
    '''
    
    Print
    '''
    def categorize(prob, cuts):
        cats = np.zeros(len(prob), dtype=int)
        for c in cuts:
            cats += (prob > c).astype(int)
        return cats
    
    cats_old = categorize(prob_old, cutoffs)
    cats_new = categorize(prob_new, cutoffs)
    for label, name in [(1, "PCa (event)"), (0, "BPH (nonevent)")]:
        mask = y_true == label
        table = np.zeros((len(cutoffs)+1, len(cutoffs)+1), dtype=int)
        for i in range(len(cutoffs)+1):
            for j in range(len(cutoffs)+1):
                table[i,j] = np.sum((cats_old[mask]==i) & (cats_new[mask]==j))
                
        print(f"\nReclassification Table ({name} patients):")
        print(pd.DataFrame(table, 
              index=[f'Old_cat{i}' for i in range(len(cutoffs)+1)],
              columns=[f'New_cat{i}' for i in range(len(cutoffs)+1)]))

def idi(y_true, prob_old, prob_new):
    '''
    
    Calculate Integrated Discrimination Improvement (IDI)
    '''
    event_idx = y_true == 1
    nonevent_idx = y_true == 0

    idi_event = np.mean(prob_new[event_idx]) - np.mean(prob_old[event_idx])
    idi_nonevent = np.mean(prob_new[nonevent_idx]) - np.mean(prob_old[nonevent_idx])
    return {
        'IDI': idi_event - idi_nonevent,
        'IDI_event': idi_event,
        'IDI_nonevent': idi_nonevent
    }

def bootstrap_nri_idi(y_true, prob_old, prob_new, n_boot=1000, cutoffs=None, alpha=0.05):
    '''
    
    # Bootstrap method to calculate 95% CI and P-value for NRI/IDI
    '''
    def ci_pval(orig, boot_arr):
        boot_arr = np.array(boot_arr)
        ci_low = np.percentile(boot_arr, 100 * alpha / 2)
        ci_high = np.percentile(boot_arr, 100 * (1 - alpha / 2))
        p = 2 * min(np.mean(boot_arr >= 0), np.mean(boot_arr <= 0))
        p = max(p, 1.0 / n_boot)
        return ci_low, ci_high, p

    rng = np.random.RandomState(42)
    n = len(y_true)
    boot_cf_nri, boot_idi = [], []
    boot_cat_nri = [] if cutoffs else None

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        boot_cf_nri.append(category_free_nri(y_true[idx], prob_old[idx], prob_new[idx])['cfNRI'])
        boot_idi.append(idi(y_true[idx], prob_old[idx], prob_new[idx])['IDI'])
        if cutoffs is not None:
            boot_cat_nri.append(categorical_nri(y_true[idx], prob_old[idx], prob_new[idx], cutoffs)['categorical NRI'])

    results = {}
    orig_cf = category_free_nri(y_true, prob_old, prob_new)['cfNRI']
    ci_cf = ci_pval(orig_cf, boot_cf_nri)
    results['cfNRI'] = (ci_cf[0], ci_cf[1], orig_cf)
    results['cfNRI_P'] = ci_cf[2]

    orig_idi = idi(y_true, prob_old, prob_new)['IDI']
    ci_idi = ci_pval(orig_idi, boot_idi)
    results['IDI'] = (ci_idi[0], ci_idi[1], orig_idi)
    results['IDI_P'] = ci_idi[2]

    if cutoffs is not None:
        orig_cat = categorical_nri(y_true, prob_old, prob_new, cutoffs)['categorical NRI']
        ci_cat = ci_pval(orig_cat, boot_cat_nri)
        results['categorical NRI'] = (ci_cat[0], ci_cat[1], orig_cat)
        results['categorical NRI_P'] = ci_cat[2]

    return results

# Unnecessary biopsy analysis function
def find_threshold_for_sensitivity(y_true, prob, target_sens):
    '''
    
    Find the highest probability threshold that satisfies sensitivity >= target_sens (i.e., the most conservative positive standard),
    thereby obtaining the minimum number of recommended biopsies.
    If the target sensitivity cannot be achieved, a warning is issued and the result under the lowest threshold is returned.
    '''
    thresholds = np.unique(np.sort(prob))[::-1]
    best_t = None
    for t in thresholds:
        y_pred = (prob >= t).astype(int)
        sens = np.sum((y_pred == 1) & (y_true == 1)) / max(1, np.sum(y_true == 1))
        if sens >= target_sens:
            best_t = t
            break
    if best_t is None:
        best_t = thresholds[-1]  # fallback
        actual_sens_fallback = np.sum((prob >= best_t) & y_true) / max(1, np.sum(y_true))
        if actual_sens_fallback < target_sens:
            print(f"Warning: Unable to reach target sensitivity {target_sens:.3f}, the lowest threshold has been used, actual sensitivity={actual_sens_fallback:.3f}")
    
    y_pred_final = (prob >= best_t).astype(int)
    n_biopsy = np.sum(y_pred_final)
    n_unnecessary = np.sum((y_pred_final == 1) & (y_true == 0))
    actual_sens = np.sum((y_pred_final == 1) & (y_true == 1)) / max(1, np.sum(y_true == 1))
    return best_t, n_biopsy, n_unnecessary, actual_sens

# Sensitivity-unnecessary biopsy curve
def plot_sensitivity_biopsy_curve(y_true, prob_old, prob_new, sens_range = (0.50, 0.95), step = 0.02, save_path = None):
    '''
    
    Plot the curves of recommended biopsies and unnecessary biopsies for the traditional model and the comprehensive model within the specified sensitivity range.
    '''
    sens_values = np.arange(sens_range[0], sens_range[1] + step/2, step)
    bio_old_list, unnec_old_list = [], []
    bio_new_list, unnec_new_list = [], []
    for sens_target in sens_values:
        _, bio_old, unnec_old, _ = find_threshold_for_sensitivity(y_true, prob_old, sens_target)
        _, bio_new, unnec_new, _ = find_threshold_for_sensitivity(y_true, prob_new, sens_target)
        bio_old_list.append(bio_old)
        unnec_old_list.append(unnec_old)
        bio_new_list.append(bio_new)
        unnec_new_list.append(unnec_new)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4), dpi = 300)
    
    ax1.plot(sens_values, bio_old_list, 'o-', label = 'Traditional model', color = '#1f77b4')
    ax1.plot(sens_values, bio_new_list, 's-', label = 'Comprehensive model', color = '#ff7f0e')
    ax1.set_xlabel('Target sensitivity', fontsize = 11)
    ax1.set_ylabel('Recommended biopsies', fontsize = 11)
    ax1.set_title('Total biopsies vs sensitivity', fontsize = 12)
    ax1.legend(fontsize = 11)
    ax1.grid(True, alpha = 0.3)
    
    ax2.plot(sens_values, unnec_old_list, 'o-', label = 'Traditional model', color = '#1f77b4')
    ax2.plot(sens_values, unnec_new_list, 's-', label = 'Comprehensive model', color = '#ff7f0e')
    ax2.set_xlabel('Target sensitivity', fontsize = 11)
    ax2.set_ylabel('Unnecessary biopsies', fontsize = 11)
    ax2.set_title('Unnecessary biopsies vs sensitivity', fontsize = 12)
    ax2.legend(fontsize = 11)
    ax2.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}.pdf', bbox_inches = 'tight')
        plt.savefig(f'{save_path}.png', bbox_inches = 'tight')
        print(f'Sensitivity-biopsy curve saved to: {save_path}.pdf')
    else:
        plt.show()
    plt.close()
    

if __name__ == '__main__':
    
    # Path configuration
    base_dir = 'E:/BaiduSyncdisk/005.Bioinformatics/SCI/005/PCa-screening'
    model_dir = f'{base_dir}/Outputs/models'
    data_path = f'{base_dir}/data/validation-data.xlsx'
    data_path2 = f'{base_dir}/data/test-data.xlsx'
    fig_dir = f'{base_dir}/Outputs/Fig_NRI_IDI'
    os.makedirs(fig_dir, exist_ok = True)

    model21_path = os.path.join(model_dir, 'voting_model_model21.joblib')  # Traditional model
    model22_path = os.path.join(model_dir, 'voting_model_model22.joblib')  # Comprehensive model

    # Load models
    bundle21 = joblib.load(model21_path)
    bundle22 = joblib.load(model22_path)

    # Load external validation data and filter BPH(1) and PCa(2)
    df_raw = pd.concat([pd.read_excel(data_path), pd.read_excel(data_path2)])
    mask = df_raw['diagnose'].isin([1, 2])
    df = df_raw[mask].copy()
    y_true = df['diagnose'].map({1: 0, 2: 1}).values
    print("========= BPH vs PCa Incremental Value Analysis =========")
    print(f"Sample size: {len(y_true)} (PCa={np.sum(y_true==1)}, BPH={np.sum(y_true==0)})")

    # Extract features and preprocess
    X21 = df[bundle21['features']]
    X22 = df[bundle22['features']]
    X21_proc = bundle21['preprocessor'].transform(X21)
    X22_proc = bundle22['preprocessor'].transform(X22)

    # Get PCa probability (confirm positive class index)
    pos_old = list(bundle21['model'].classes_).index(1)
    pos_new = list(bundle22['model'].classes_).index(1)
    prob_old = bundle21['model'].predict_proba(X21_proc)[:, pos_old]
    prob_new = bundle22['model'].predict_proba(X22_proc)[:, pos_new]

    #  -------- NRI and IDI --------
    print("\n========== Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI) ==========")

    # Category-free NRI
    cf_nri = category_free_nri(y_true, prob_old, prob_new)
    print(f"Category‑Free NRI: {cf_nri['cfNRI']:.4f} "
          f"(Event NRI: {cf_nri['NRI_event']:.4f}, Nonevent NRI: {cf_nri['NRI_nonevent']:.4f})")

    # Categorical NRI (using 0.1, 0.3 cutoffs, adjustable based on clinical risk stratification)
    cutoffs = [0.1, 0.3]
    cat_nri = categorical_nri(y_true, prob_old, prob_new, cutoffs)
    print(f"Categorical NRI (cutoffs {cutoffs}): {cat_nri['categorical NRI']:.4f}")
    print("Reclassification Table (row=old model category, col=new model category):")
    print(pd.DataFrame(cat_nri['reclassification_table']))
    
    # Print stratified tables for events/non-events
    print_reclass_by_event(y_true, prob_old, prob_new, cutoffs)

    # IDI
    idi_res = idi(y_true, prob_old, prob_new)
    print(f"IDI: {idi_res['IDI']:.4f} "
          f"(Event: {idi_res['IDI_event']:.4f}, Nonevent: {idi_res['IDI_nonevent']:.4f})")

    # Bootstrap inference
    boot_res = bootstrap_nri_idi(y_true, prob_old, prob_new, n_boot = 1000, cutoffs = cutoffs)
    print("\nBootstrap 95% CI and P-value:")
    print(f"cfNRI: {boot_res['cfNRI'][2]:.4f} "
          f"(95% CI: {boot_res['cfNRI'][0]:.4f}–{boot_res['cfNRI'][1]:.4f}), P={boot_res['cfNRI_P']:.4f}")
    print(f"Categorical NRI: {boot_res['categorical NRI'][2]:.4f} "
          f"(95% CI: {boot_res['categorical NRI'][0]:.4f}–{boot_res['categorical NRI'][1]:.4f}), P={boot_res['categorical NRI_P']:.4f}")
    print(f"IDI: {boot_res['IDI'][2]:.4f} "
          f"(95% CI: {boot_res['IDI'][0]:.4f}–{boot_res['IDI'][1]:.4f}), P={boot_res['IDI_P']:.4f}")

    # -------- Reducing Unnecessary Biopsies --------
    print("\n========== Reducing Unnecessary Biopsies ==========")
    if 'TPSA' in df.columns:
        sens_psa = np.sum((df['TPSA'] >= 4) & (y_true == 1)) / max(1, np.sum(y_true == 1))
        bio_psa = np.sum(df['TPSA'] >= 4)
        unnec_psa = np.sum((df['TPSA'] >= 4) & (y_true == 0))
        print(f"\nControl: PSA>=4 ng/mL strategy -> Recommended biopsies={bio_psa}, "
              f"Unnecessary biopsies={unnec_psa}, Sensitivity={sens_psa:.3f}")

        # Threshold analysis targeting PSA>=4 sensitivity
        t_old_psa, bio_old_psa, unnec_old_psa, sen_old_psa = find_threshold_for_sensitivity(
            y_true, prob_old, sens_psa)
        t_new_psa, bio_new_psa, unnec_new_psa, sen_new_psa = find_threshold_for_sensitivity(
            y_true, prob_new, sens_psa)
        print(f"Targeting PSA>=4 sensitivity ({sens_psa:.3f}):")
        print(f"Traditional model: Threshold={t_old_psa:.3f}, Biopsies={bio_old_psa}, Unnecessary={unnec_old_psa}")
        print(f"Comprehensive model: Threshold={t_new_psa:.3f}, Biopsies={bio_new_psa}, Unnecessary={unnec_new_psa}")
        if unnec_old_psa > 0:
            red = unnec_old_psa - unnec_new_psa
            print(f"Comprehensive model reduced unnecessary biopsies by {red} cases (relative reduction of {red/unnec_old_psa*100:.1f}%)")

    # Plot sensitivity-unnecessary biopsy curve
    plot_sensitivity_biopsy_curve(
        y_true, 
        prob_old, 
        prob_new, 
        sens_range = (0.50, 0.95), 
        step = 0.02,
        save_path = os.path.join(fig_dir, 'sensitivity_biopsy_curve')
    )

