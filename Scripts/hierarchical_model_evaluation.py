# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, classification_report, confusion_matrix, accuracy_score,
    cohen_kappa_score, precision_score, recall_score, f1_score,
    brier_score_loss
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore', category = UserWarning)

# Define directory creation function
def mkdirs(dir_path: str, force: bool = False):
    if os.path.exists(dir_path):
        if os.path.isdir(dir_path):
            if force:
                shutil.rmtree(dir_path)
            else:
                return
        else:
            raise NotADirectoryError(f"Path '{dir_path}' exists but is not a directory.")
    os.makedirs(dir_path)

# Global settings
CLASS_LABELS = {
    0: 'Healthy',
    1: 'BPH',
    2: 'PCa'
}
N_CLASSES = len(CLASS_LABELS)

# Confusion Matrix Plotting
def plot_confusion_matrix_plus(y_true, y_pred, title = 'Confusion Matrix', class_names = None, save_path = None):
    
    '''
    Draws a detailed and aesthetically pleasing confusion matrix for three or more classes.
    - y_true: True labels
    - y_pred: Predicted labels
    - title: Chart title
    - class_names: A dictionary mapping numeric labels to readable class names, e.g., {0: 'Healthy', 1: 'BPH', 2: 'PCa'}
    - save_path: Path to save the image
    '''
    if class_names is None: class_names = {i: f'Class {i}' for i in range(N_CLASSES)}
    
    # Get all unique classes that appear in either the true or predicted labels
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Generate label names for plotting based on the unique classes present
    labels_for_plot = [class_names.get(label, f'Unknown-{label}') for label in unique_labels]
    
    cm = confusion_matrix(y_true, y_pred, labels = unique_labels)
    
    # Calculate row normalization (Recall)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        cm_normalized = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
    
    # Create a larger matrix to accommodate the totals (sum row and sum column)
    cm_with_totals = np.vstack([cm, cm.sum(axis = 0)])
    cm_with_totals = np.hstack([cm_with_totals, cm_with_totals.sum(axis = 1)[:, np.newaxis]])
    
    # Heatmap data (total row and column are left uncolored/blank)
    heatmap_data = np.vstack([cm_normalized, np.zeros((1, cm_normalized.shape[1]))])
    heatmap_data = np.hstack([heatmap_data, np.zeros((heatmap_data.shape[0], 1))])
    
    fig, ax = plt.subplots(figsize = (5, 5))
    
    # Use a green color palette
    cmap = sns.color_palette('Greens', as_cmap = True)
    
    sns.heatmap(
        heatmap_data,
        annot = False,
        cmap = cmap, 
        cbar = False, 
        square = True, 
        linewidths = 1.5,
        linecolor = 'white',
        ax = ax, 
        vmin = 0.0, 
        vmax = 1.0
    )
    
    # Populate the cells with text
    for i in range(cm_with_totals.shape[0]):
        for j in range(cm_with_totals.shape[1]):
            cell_value = cm_with_totals[i, j]
            # Main diagonal and internal cells (excluding totals)
            if i < len(unique_labels) and j < len(unique_labels):
                percentage = cm_normalized[i, j]
                ax.text(j + 0.5, i + 0.35, f'{percentage:.1%}', ha = 'center', va = 'center', fontsize = 14, color = 'black')
                ax.text(j + 0.5, i + 0.65, f'{int(cell_value)}', ha = 'center', va = 'center', fontsize = 13, color = 'black')
            # Total cells (row/column sums)
            else:
                ax.text(j + 0.5, i + 0.5, f'{int(cell_value)}', ha = 'center', va = 'center', fontsize = 14, color = 'black', weight = 'bold')
    
    # Set labels and title
    ax.set_xticklabels(labels_for_plot + ['Total'], rotation = 0, ha = 'center')
    ax.set_yticklabels(labels_for_plot + ['Total'], rotation = 0, ha = 'center')
    ax.set_xlabel('Predicted Label', fontsize = 12)
    ax.set_ylabel('True Label', fontsize = 12)
    ax.set_title(title, fontsize = 12, pad = 5)
    
    # Set tick parameters
    ax.tick_params(axis = 'x', labelsize = 12)
    ax.tick_params(axis = 'y', labelsize = 12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}.png', dpi = 300, bbox_inches = 'tight')
        plt.savefig(f'{save_path}.pdf', dpi = 300, bbox_inches = 'tight')
        print(f'Confusion matrix saved to: {save_path}')
        
    plt.close()


# Core Logic: Hierarchical Classifier
class HierarchicalClassifier:
    
    '''
    A hierarchical classifier that encapsulates two binary classification models.
    - Model 1: Healthy (0) vs. Disease (1) [BPH + PCa]
    - Model 2: BPH (0) vs. PCa (1)
    This class can now handle cases where the two models use different feature sets
    and allows for adjusting the decision threshold for the second-stage model.
    '''
    def __init__(self, model_step1, model_step2, preprocessor1, preprocessor2, pca_threshold = 0.5):
        
        '''
        Initializes the hierarchical classifier.
        :param model_step1: The first model (Healthy vs. Disease).
        :param model_step2: The second model (BPH vs. PCa).
        :param pca_threshold: float, the probability threshold to classify a case as PCa in the second step.
                              Lowering this value increases sensitivity for PCa.
        '''
        
        # Use the reusable preprocessors to transform new data
        self.preprocessor1 = preprocessor1
        self.preprocessor2 = preprocessor2
        
        self.model_step1 = model_step1
        self.model_step2 = model_step2
        
        # Store the tuned decision threshold for PCa
        self.pca_threshold = pca_threshold
        print(f'HierarchicalClassifier initialized with a PCa threshold of: {self.pca_threshold}')
        
        if not hasattr(model_step1, 'feature_names_in_') or not hasattr(model_step2, 'feature_names_in_'):
            raise AttributeError("Model object is missing 'feature_names_in_' attribute.")
            
        self.features1 = model_step1.feature_names_in_
        self.features2 = model_step2.feature_names_in_
        print(f'Model 1 requires {len(self.features1)} features.')
        print(f'Model 2 requires {len(self.features2)} features.')
        
        self.STEP1_HEALTHY = 0
        self.STEP1_DISEASE = 1
        self.STEP2_BPH = 0
        self.STEP2_PCA = 1
        
        self.FINAL_HEALTHY = 0
        self.FINAL_BPH = 1
        self.FINAL_PCA = 2
        
    def predict(self, X):
        
        '''
        Performs three-class prediction on the input data X using the custom threshold.
        :param X: pandas DataFrame, containing all features required by both models.
        :return: numpy array, containing the final predicted classes (0, 1, or 2).
        '''
        if not isinstance(X, pd.DataFrame): raise TypeError('Input data X must be a pandas DataFrame.')
        
        X1 = X[self.features1]
        X1 = self.preprocessor1.transform(X1)

        # Use Model 1 for prediction (Healthy vs. Disease)
        pred_step1 = self.model_step1.predict(X1)
        
        final_predictions = np.full(X.shape[0], self.FINAL_HEALTHY, dtype = int)
        
        disease_indices = np.where(pred_step1 == self.STEP1_DISEASE)[0]
        
        if len(disease_indices) == 0: return final_predictions
        
        X_disease = X.iloc[disease_indices]
        X2_disease = X_disease[self.features2]
        X2_disease = self.preprocessor2.transform(X2_disease)
        
        # Use probability and custom threshold for decision making
        # Get Model 2's prediction probabilities for the "disease" samples
        proba_step2 = self.model_step2.predict_proba(X2_disease)
        
        # Extract the probability of predicting PCa (assuming PCa is class 1 in Model 2)
        proba_pca_in_disease = proba_step2[:, self.STEP2_PCA]
        
        # Determine the final prediction based on our custom threshold
        # If P(PCa) > self.pca_threshold, predict PCa (class 1), otherwise BPH (class 0)
        pred_step2 = np.where(proba_pca_in_disease > self.pca_threshold, self.STEP2_PCA, self.STEP2_BPH)
        
        final_disease_predictions = np.where(pred_step2 == self.STEP2_BPH, self.FINAL_BPH, self.FINAL_PCA)
        
        final_predictions[disease_indices] = final_disease_predictions
        
        return final_predictions
    
    def predict_proba(self, X):
        
        '''
        Predicts the probability of each class for the input data X.
        (This method remains unchanged as it calculates the fundamental probabilities)
        '''
        if not isinstance(X, pd.DataFrame): raise TypeError('Input data X must be a pandas DataFrame.')
        
        X1 = X[self.features1]
        X2 = X[self.features2]
        X1 = self.preprocessor1.transform(X1)
        X2 = self.preprocessor2.transform(X2)
        
        proba_step1 = self.model_step1.predict_proba(X1)
        proba_step2 = self.model_step2.predict_proba(X2)
        
        final_probas = np.zeros((X.shape[0], N_CLASSES))
        
        p_healthy = proba_step1[:, self.STEP1_HEALTHY]
        p_disease = proba_step1[:, self.STEP1_DISEASE]
        p_bph = proba_step2[:, self.STEP2_BPH] * p_disease
        p_pca = proba_step2[:, self.STEP2_PCA] * p_disease
        
        final_probas[:, self.FINAL_HEALTHY] = p_healthy
        final_probas[:, self.FINAL_BPH] = p_bph
        final_probas[:, self.FINAL_PCA] = p_pca
        
        return final_probas

# ROC Curve
def plot_multiclass_roc(y_true, y_proba, title = 'Multi-class ROC Curve', save_path = None):
    
    '''
    Plots the ROC curves for a multi-class problem. (No changes needed here)
    '''
    y_true_binarized = label_binarize(y_true, classes = list(CLASS_LABELS.keys()))
    
    fpr, tpr, roc_auc = dict(), dict(), dict()
    
    for i in range(N_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true_binarized.ravel(), y_proba.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_CLASSES):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= N_CLASSES
    
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize = (5, 5))
    
    plt.plot(fpr['micro'], tpr['micro'], label = f'Micro-average ROC (AUC = {roc_auc["micro"]:.3f})', color = 'deeppink', linestyle = ':', linewidth = 4)
    plt.plot(fpr['macro'], tpr['macro'], label = f'Macro-average ROC (AUC = {roc_auc["macro"]:.3f})', color = 'navy', linestyle = ':', linewidth = 4)
    
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    for i, color in zip(range(N_CLASSES), colors): plt.plot(fpr[i], tpr[i], color = color, lw = 2, label = f'Class \'{CLASS_LABELS[i]}\' (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw = 2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize = 12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize = 12)
    plt.title(title, fontsize = 12, pad = 5)
    plt.legend(loc = "lower right", fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}.png', dpi = 300, bbox_inches = 'tight')
        plt.savefig(f'{save_path}.pdf', dpi = 300, bbox_inches = 'tight')
        print(f'ROC curve plot saved to: {save_path}')
        
    plt.close()

# Evaluate
def evaluate_hierarchical_brier_scores(hierarchical_model, X_test, y_true):
    
    print('\n--- Overall System Evaluation (3-Class: Healthy, BPH, PCa) ---')
    y_proba_overall = hierarchical_model.predict_proba(X_test)
    y_true_binarized = label_binarize(y_true, classes = list(CLASS_LABELS.keys()))
    
    # Multi-class Brier Score (BS)
    bs_model_overall = np.mean(np.sum((y_proba_overall - y_true_binarized)**2, axis = 1))
    
    # Multi-class Reference Brier Score (BS_ref)
    class_proportions = np.mean(y_true_binarized, axis = 0)
    bs_ref_overall = 1 - np.sum(class_proportions**2)
    
    # Multi-class Brier Skill Score (BSS)
    bss_overall = 1 - (bs_model_overall / bs_ref_overall) if bs_ref_overall > 0 else np.nan
    
    print(f'Model Brier Score (BS): {bs_model_overall:.4f}')
    print(f'Reference Baseline Brier Score (BS_ref): {bs_ref_overall:.4f}')
    print(f'Brier Skill Score (BSS): {bss_overall:.4f}')

    # 2. Stage 1 Evaluation (Binary: Healthy vs. Disease)
    print('\n--- Stage 1 Evaluation (Binary: Healthy vs. Disease) ---')
    model1 = hierarchical_model.model_step1
    preprocessor1 = hierarchical_model.preprocessor1
    features1 = hierarchical_model.features1
    
    y_true_stage1 = y_true.map({0: 0, 1: 1, 2: 1})
    X1_test = preprocessor1.transform(X_test[features1])
    y_proba_stage1 = model1.predict_proba(X1_test)
    
    # Binary Brier Score (BS) - using standard sklearn method
    pos_label_stage1 = 1 # Disease is the positive class
    bs_model_stage1 = brier_score_loss(y_true_stage1, y_proba_stage1[:, pos_label_stage1], pos_label = pos_label_stage1)
    
    # Binary Reference Brier Score (BS_ref)
    proportion_pos_stage1 = np.mean(y_true_stage1 == pos_label_stage1)
    bs_ref_stage1 = proportion_pos_stage1 * (1 - proportion_pos_stage1)
    
    # Binary Brier Skill Score (BSS)
    bss_stage1 = 1 - (bs_model_stage1 / bs_ref_stage1) if bs_ref_stage1 > 0 else np.nan
    
    print(f'Model Brier Score (BS): {bs_model_stage1:.4f}')
    print(f'Reference Baseline Brier Score (BS_ref): {bs_ref_stage1:.4f}')
    print(f'Brier Skill Score (BSS): {bss_stage1:.4f}')

    # 3. Stage 2 Evaluation (Binary: BPH vs. PCa)
    print('\n--- Stage 2 Evaluation (Binary: BPH vs. PCa) ---')
    model2 = hierarchical_model.model_step2
    preprocessor2 = hierarchical_model.preprocessor2
    features2 = hierarchical_model.features2

    disease_mask = y_true > 0
    if np.sum(disease_mask) > 0:
        X_test_disease = X_test[disease_mask]
        y_true_disease = y_true[disease_mask]

        y_true_stage2 = y_true_disease.map({1: 0, 2: 1}) # BPH = 0, PCa = 1

        X2_test_disease = preprocessor2.transform(X_test_disease[features2])
        y_proba_stage2 = model2.predict_proba(X2_test_disease)
        
        # Binary Brier Score (BS) - using standard sklearn method
        pos_label_stage2 = 1 # PCa is the positive class
        bs_model_stage2 = brier_score_loss(y_true_stage2, y_proba_stage2[:, pos_label_stage2], pos_label = pos_label_stage2)
        
        # Binary Reference Brier Score (BS_ref)
        proportion_pos_stage2 = np.mean(y_true_stage2 == pos_label_stage2)
        bs_ref_stage2 = proportion_pos_stage2 * (1 - proportion_pos_stage2)
        
        # Binary Brier Skill Score (BSS)
        bss_stage2 = 1 - (bs_model_stage2 / bs_ref_stage2) if bs_ref_stage2 > 0 else np.nan

        print(f'Model Brier Score (BS): {bs_model_stage2:.4f}')
        print(f'Reference Baseline Brier Score (BS_ref): {bs_ref_stage2:.4f}')
        print(f'Brier Skill Score (BSS): {bss_stage2:.4f}')
        print()
    else:
        print('Stage 2 Evaluation: Not applicable (no disease samples in the dataset).')

# Detailed Metrics Calculation
def calculate_multiclass_metrics(y_true, y_pred, class_labels):
    
    '''
    Calculates a comprehensive set of metrics for multi-class classification. (No changes needed here)
    '''
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    
    metrics = {
        'Sensitivity (Recall)': [], 'Specificity': [], 'Precision (PPV)': [],
        'NPV': [], 'F1-Score': [], 'Support': []
    }
    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        support = tp + fn
        metrics['Support'].append(support)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['Sensitivity (Recall)'].append(sensitivity)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['Specificity'].append(specificity)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['Precision (PPV)'].append(precision)
        
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics['NPV'].append(npv)
        
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        metrics['F1-Score'].append(f1)
        
    class_names = [class_labels[i] for i in range(n_classes)]
    df_metrics = pd.DataFrame(metrics, index = class_names)
    
    macro_precision = precision_score(y_true, y_pred, average = 'macro', zero_division = 0)
    macro_recall = recall_score(y_true, y_pred, average = 'macro', zero_division = 0)
    macro_f1 = f1_score(y_true, y_pred, average = 'macro', zero_division = 0)
    macro_specificity = np.mean(df_metrics['Specificity'])
    macro_npv = np.mean(df_metrics['NPV'])
    
    weighted_precision = precision_score(y_true, y_pred, average = 'weighted', zero_division = 0)
    weighted_recall = recall_score(y_true, y_pred, average = 'weighted', zero_division = 0)
    weighted_f1 = f1_score(y_true, y_pred, average = 'weighted', zero_division = 0)
    weighted_specificity = np.average(df_metrics['Specificity'], weights = df_metrics['Support'])
    weighted_npv = np.average(df_metrics['NPV'], weights = df_metrics['Support'])
    
    df_metrics.loc['Macro Average'] = [macro_recall, macro_specificity, macro_precision, macro_npv, macro_f1, np.nan]
    df_metrics.loc['Weighted Average'] = [weighted_recall, weighted_specificity, weighted_precision, weighted_npv, weighted_f1, len(y_true)]
    
    overall_accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    df_overall = pd.DataFrame({
        'Metric': ['Overall Accuracy', "Cohen's Kappa"],
        'Value': [overall_accuracy, kappa]
    }).set_index('Metric')

    print('\nDetailed Performance Metrics (Per-class and Averages)')
    print(df_metrics.to_string(float_format = '{:.4f}'.format))

    print('\nOverall Performance Metrics')
    print(df_overall.to_string(float_format = '{:.4f}'.format))
    
    return df_metrics, df_overall

    
if __name__ == '__main__':
    
    # -------------------------------------------------------------------------
    import configparser
    CF = configparser.ConfigParser()
    CF.read('E:/BaiduSyncdisk/005.Bioinformatics/SCI/005/PCa-screening/Scripts/.config.txt', encoding = 'utf-8')
    rtdir = CF.get('ENV', 'rtdir')
    otdir = CF.get('ENV', 'otdir')
    mddir = f'{otdir}/Outputs/models'
    # -------------------------------------------------------------------------
    fgdir = f'{otdir}/Outputs/Fig_Hierarchical_Model'; mkdirs(fgdir)
    
    # Configuration and Loading
    this = 'model2'
    if this == 'model1':
        lthis = 'Baseline Model'
        models = ['voting_model_model11.joblib', 'voting_model_model21.joblib']
    else:
        lthis = 'Comprehensive Hierarchical Model'
        models = ['voting_model_model12.joblib', 'voting_model_model22.joblib']

    MODEL_STEP1_PATH = os.path.join(mddir, models[0])
    MODEL_STEP2_PATH = os.path.join(mddir, models[1])

    ttfile = os.path.join(rtdir, 'data/test-data.xlsx')
    vnfile = os.path.join(rtdir, 'data/validation-data.xlsx')
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print('Starting hierarchical model evaluation...')

    try:
        model1_bundle = joblib.load(MODEL_STEP1_PATH)
        model2_bundle = joblib.load(MODEL_STEP2_PATH)
    except FileNotFoundError as e:
        print(f'Error: Model file not found. Please check if the path "{e.filename}" is correct.'); exit()
    
    try:
        pd.set_option('display.max_columns', 10) 
        dattt = pd.read_excel(ttfile)
        datvn = pd.read_excel(vnfile)
    except FileNotFoundError as e:
        print(f'Error: Data file not found. Please check if the path "{e.filename}" is correct.'); exit()

    full_data = pd.concat([dattt, datvn], ignore_index = True)
    print(f'Data loaded and merged successfully. Total samples: {len(full_data)}')

    if 'diagnose' not in full_data.columns:
        raise ValueError('The "diagnose" column was not found in the data files. Please check your data.')
        
    y_true = full_data['diagnose']
    X_test = full_data.drop(columns = ['diagnose'])

    # Model Prediction
    print('\nInstantiating hierarchical classifier and making predictions...')
    
    # Pass custom threshold during instantiation.
    # This value can be adjusted to find the optimal balance.
    # Lower values increase PCa sensitivity but may decrease precision.
    PCA_SENSITIVITY_THRESHOLD = 0.5
    
    hierarchical_model = HierarchicalClassifier(
        model1_bundle['model'], 
        model2_bundle['model'], 
        preprocessor1 = model1_bundle['preprocessor'],
        preprocessor2 = model2_bundle['preprocessor'],
        pca_threshold = PCA_SENSITIVITY_THRESHOLD
    )
    
    print('\nGenerating predictions...')
    y_pred = hierarchical_model.predict(X_test)
    
    print('Calculating final prediction probabilities...')
    y_proba = hierarchical_model.predict_proba(X_test)
    
    print('\nOverall Model Performance Evaluation (Report & Visualization)')
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Overall Hierarchical Model Accuracy (PCa threshold = {PCA_SENSITIVITY_THRESHOLD}): {accuracy:.4f} ({accuracy:.2%})')
    
    print('\n--- Detailed Classification Report ---')
    target_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    report = classification_report(y_true, y_pred, target_names = target_names, digits = 4)
    print(report)
    
    X_test['y_pred'] = y_pred
    X_test['y_true'] = y_true
    X_test = X_test[['y_pred', 'y_true', 'APOE', 'AFP', 'AR+TREM2+'] + list(set(['TPSA', 'LY%', 'HCT', 'RDW-CV', 'FPSA/TPSA', 'FPSA', 'Urea', 'HGB', 'LY#', 'TPSA*AR+TREM2+', 'NEUT#', 'MONO#', 'PLT', 'TPSA', 'FPSA/TPSA', 'NEUT#', 'MCHC', 'APOE*AR+TREM2+', 'MCH', 'AFP*AR+TREM2+', 'age']))]
    
    # Call the corrected evaluation function
    evaluate_hierarchical_brier_scores(hierarchical_model, X_test, y_true)
    
    plot_confusion_matrix_plus(
        y_true, y_pred, 
        title = f'Confusion Matrix (PCa Threshold = {PCA_SENSITIVITY_THRESHOLD})\n({lthis})',
        class_names = CLASS_LABELS,
        save_path = f'{this}_hierarchical_confusion_matrix_thresh_{PCA_SENSITIVITY_THRESHOLD}'
    )
    
    # The ROC curve is probability-based and unaffected by changes in decision threshold, so its AUC remains unchanged.
    plot_multiclass_roc(
        y_true,
        y_proba,
        title = f'Multi-class ROC Curve\n({lthis})',
        save_path = f'{this}_hierarchical_roc_auc_curve'
    )

    # Detailed Metrics Calculation
    df_metrics, df_overall = calculate_multiclass_metrics(y_true, y_pred, CLASS_LABELS)
    pd.DataFrame(df_metrics).to_excel(f'{otdir}/Outputs/Tbl_all/hierarchical_{this}.xlsx')

    print('\n--- Evaluation Completed ---')
    
 