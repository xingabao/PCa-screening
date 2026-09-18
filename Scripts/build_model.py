# -*- coding: utf-8 -*-

import os
os.environ["SCIPY_ARRAY_API"] = "1"
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
import scipy.stats as stats
from sklearn.metrics import roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
    
import joblib
import re
import os
import sys
import shutil
import logging

import shap
import optuna
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tabpfn import TabPFNClassifier


# Define function to create directory
def mkdirs(dir_path: str, force: bool):
    if os.path.exists(dir_path):
        if os.path.isdir(dir_path):
            if force:
                shutil.rmtree(dir_path)
            else:
                return
        else:
            raise NotADirectoryError(f"Path '{dir_path}' exists but is not a directory.")
    os.makedirs(dir_path)

# Calculate and display metrics for each model on the test set
def calculate_metrics(y_true, y_pred_proba, threshold = 0.5):
    y_pred = (y_pred_proba > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels = [0, 1])
    if cm.size == 1: # Handle cases where only a single class is predicted
        unique_class = np.unique(y_true)
        if len(unique_class) == 1 and unique_class[0] == 0:
            tn, fp, fn, tp = cm[0][0], 0, 0, 0
        elif len(unique_class) == 1 and unique_class[0] == 1:
            tn, fp, fn, tp = 0, 0, 0, cm[0][0]
        else: # All predicted 0 or 1, but true values contain both classes
            if np.unique(y_pred)[0] == 0: # All predicted negative
                tn, fp = cm[0,0], cm[0,1] if cm.shape[1]>1 else 0
                fn, tp = cm[1,0] if cm.shape[0]>1 else 0, 0
            else: # All predicted positive
                tn, fp = 0, cm[0,0]
                fn, tp = 0, cm[1,0] if cm.shape[0]>1 else 0
    else:
        tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, zero_division = 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = precision_score(y_true, y_pred, zero_division = 0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division = 0)
    kappa = cohen_kappa_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred_proba)
    return [accuracy, sensitivity, specificity, ppv, npv, f1, kappa, brier]

# Calculate AUC confidence intervals using bootstrap
def calculate_auc_ci(y_true, y_pred_proba, n_bootstraps = 1000, seed = 42):
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    rng = np.random.RandomState(seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_proba), len(y_pred_proba))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_pred_proba[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    original_auc = roc_auc_score(y_true, y_pred_proba)
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    return original_auc, ci_lower, ci_upper

#  DeLong's Test for AUC Comparison
def delong_test(y_true, y_scores1, y_scores2):
    """
    Computes the DeLong test for two correlated ROC curves.
    Args:
        y_true (np.array): Ground truth labels (0 or 1).
        y_scores1 (np.array): Predicted probabilities for model 1.
        y_scores2 (np.array): Predicted probabilities for model 2.
    Returns:
        z_score (float): The z-score for the test.
        p_value (float): The p-value for the test.
    """
    y_true = np.asarray(y_true)
    y_scores1 = np.asarray(y_scores1)
    y_scores2 = np.asarray(y_scores2)

    # Get the number of positive and negative cases
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    # Helper function to compute the V matrix components
    def get_v_components(y_true, y_scores):
        y_true_pos = y_true == 1
        y_true_neg = y_true == 0
        scores_pos = y_scores[y_true_pos]
        scores_neg = y_scores[y_true_neg]
        
        v_10 = np.zeros(len(scores_pos))
        for i in range(len(scores_pos)):
            v_10[i] = np.mean(scores_pos[i] > scores_neg)
            
        v_01 = np.zeros(len(scores_neg))
        for i in range(len(scores_neg)):
            v_01[i] = np.mean(scores_neg[i] < scores_pos)
            
        return v_10, v_01

    # Compute V components for both models
    v1_10, v1_01 = get_v_components(y_true, y_scores1)
    v2_10, v2_01 = get_v_components(y_true, y_scores2)

    # Compute AUCs
    auc1 = np.mean(v1_10)
    auc2 = np.mean(v2_10)

    # Compute the covariance
    # Use ddof = 1 for sample covariance
    s_10 = np.cov(v1_10, v2_10, ddof = 1)
    s_01 = np.cov(v1_01, v2_01, ddof = 1)
    
    # Compute the variance of the AUC difference
    var_diff = (s_10[0, 0] / n1) + (s_01[0, 0] / n0) - 2 * (s_10[0, 1] / n1 + s_01[0, 1] / n0)
    
    # Handle cases where variance is zero or negative
    if var_diff <= 0:
        return 0.0, 1.0

    # Compute z-score and p-value
    z_score = (auc1 - auc2) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

    return z_score, p_value

# Vectorized function to calculate model net revenue
def compute_net_benefit_model_vectorized(thresholds, y_pred_scores, y_labels):
    
    # Convert y_labels to numpy array first to avoid multi-dimensional indexing issues
    y_labels = np.array(y_labels)
    y_pred_scores = np.array(y_pred_scores)
    
    # Calculate the total number of samples
    n = len(y_labels)
    
    # Pre-allocate array
    net_benefit_model = np.zeros_like(thresholds)
    
    # Broadcast prediction scores and thresholds for comparison
    y_pred_matrix = (y_pred_scores[:, None] > thresholds).astype(int)
    
    # Vectorized calculation of confusion matrix elements: TP and FP
    tp = (y_pred_matrix & y_labels[:, None]).sum(axis=0)
    fp = ((y_pred_matrix == 1) & (y_labels[:, None] == 0)).sum(axis = 0)
    
    # Calculate net benefit
    try:
        net_benefit_model = (tp / n) - (fp / n) * (thresholds / (1 - thresholds))
    except:
        # Add a small epsilon to prevent division by zero
        epsilon = 1e-9
        net_benefit_model = (tp / n) - (fp / n) * (thresholds / (1 - thresholds + epsilon))
        
    return net_benefit_model

# Vectorized function to calculate the net benefit of the "Treat All" strategy
def compute_net_benefit_all_vectorized(thresholds, y_labels):
    
    # Convert y_labels to numpy array first to avoid multi-dimensional indexing issues
    y_labels = np.array(y_labels)
    
    # Calculate confusion matrix elements (under "Treat All", all samples are predicted positive)
    tn, fp, fn, tp = confusion_matrix(y_labels, y_labels).ravel()  
    total = tp + tn
    
    # Pre-allocate array
    net_benefit_all = np.zeros_like(thresholds)
    
    # Vectorized calculation of net benefit
    net_benefit_all = (tp / total) - (tn / total) * (thresholds / (1 - thresholds))
    
    return net_benefit_all

# Function to plot Decision Curve Analysis (DCA)
def plot_dca_custom(thresholds, net_benefit_model, net_benefit_all, prefix):
    
    fig, ax = plt.subplots(figsize = (4, 5), dpi = 300)
    
    # Plot net benefit curves
    ax.plot(thresholds, net_benefit_model, color = 'deepskyblue', label = 'Model') 
    ax.plot(thresholds, net_benefit_all, color = 'black', label = 'Treat all')  
    ax.plot((0, 1), (0, 0), color = '#808080', label = 'Treat none')  

    # Fill the area where the model outperforms both "Treat all" and "Treat none"
    y2 = np.maximum(net_benefit_all, 0)
    y1 = np.maximum(net_benefit_model, y2)
    ax.fill_between(thresholds, y1, y2, color = 'deepskyblue', alpha = 0.3)
    
    # Beautify chart
    ax.set_xlim(0, 1)
    ax.set_ylim(net_benefit_model.min() - 0.15, net_benefit_model.max() + 0.15)
    ax.set_xlabel('Threshold Probability', fontdict = {'family': 'Times New Roman', 'fontsize': 10})
    ax.set_ylabel('Net Benefit', fontdict = {'family': 'Times New Roman', 'fontsize': 10})
    ax.grid(True)
    ax.legend(loc = 'upper right')
    
    plt.savefig(f'{fgdir4}/{prefix}.png', bbox_inches = 'tight')
    plt.savefig(f'{fgdir4}/{prefix}.pdf', bbox_inches = 'tight')
    plt.close()   
    
# Function to run Decision Curve Analysis (DCA)
def run_dca_analysis(model, X_test, y_test, prefix):
    
    # Predict probabilities using the model
    y_pred_scores = model.predict_proba(X_test)[:, 1]
    y_labels = y_test
    
    # Define the range of threshold probabilities
    thresholds = np.arange(0, 1, 0.01)
    
    # Calculate net benefits at different thresholds
    net_benefit_model = compute_net_benefit_model_vectorized(thresholds, y_pred_scores, y_labels)
    net_benefit_all = compute_net_benefit_all_vectorized(thresholds, y_labels)
    
    # Call the plotting function
    plot_dca_custom(thresholds, net_benefit_model, net_benefit_all, prefix)


import warnings
warnings.filterwarnings(
    action = 'ignore', 
    category = UserWarning,
    message = '.*Parameters.*use_label_encoder.*are not used.*'
)


if __name__ == '__main__':

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Basic Parameter Settings for Multiple Models
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ------------------------------------
    # 1. Model 1-1: Normal vs. Abnormal (BPH + Prostate Cancer), traditional markers: FPSA, TPSA, FPSA/TPSA
    # 2. Model 1-2: Normal vs. Abnormal (BPH + Prostate Cancer), traditional markers + new markers
    # 3. Model 2-1: BPH vs. Prostate Cancer, traditional markers: FPSA, TPSA, FPSA/TPSA
    # 4. Model 2-2: BPH vs. Prostate Cancer, traditional markers + new markers
    modeldict = {
        'model11': {'normal': [0], 'abnormal': [1, 2], 'delcols': ['age'], 'features': ['FPSA', 'TPSA', 'FPSA/TPSA'], 'best_num': 3},
        'model12': {'normal': [0], 'abnormal': [1, 2], 'delcols': ['age'], 'features': None, 'best_num': 13},
        'model21': {'normal': [1], 'abnormal': [2], 'delcols': [], 'features': ['FPSA', 'TPSA', 'FPSA/TPSA'], 'best_num': 3},
        'model22': {'normal': [1], 'abnormal': [2], 'delcols': [], 'features': None, 'best_num': 8},
    }
    this = 'model12'
    
    # ------------------------------------
    mkstatus = False
    if True:
        savefig = True
        savetbl = True
        verbose = False
        shapsta = True
    else:
        savefig = True
        savetbl = True
        verbose = True
        shapsta = False
    finalmodel = 'Ensemble Model'

    # -------------------------------------------------------------------------
    import configparser
    CF = configparser.ConfigParser()
    CF.read('E:/BaiduSyncdisk/005.Bioinformatics/SCI/005/PCa-screening/.config.txt', encoding = 'utf-8')
    rtdir = CF.get('ENV', 'rtdir')
    otdir = CF.get('ENV', 'otdir')
    dtdir = CF.get('ENV', 'dtdir')
    fgdir = f'{otdir}/Outputs/Fig_all'; mkdirs(fgdir, mkstatus)
    fgdir1 = f'{otdir}/Outputs/Fig_Feature_Ranking'; mkdirs(fgdir1, mkstatus)
    fgdir2 = f'{otdir}/Outputs/Fig_Feature_Selection'; mkdirs(fgdir2, mkstatus)
    fgdir3 = f'{otdir}/Outputs/Fig_ROC_AUC'; mkdirs(fgdir3, mkstatus)
    fgdir4 = f'{otdir}/Outputs/Fig_DCA'; mkdirs(fgdir4, mkstatus)
    fgdir5 = f'{otdir}/Outputs/Fig_SHAP'; mkdirs(fgdir5, mkstatus)
    tldir = f'{otdir}/Outputs/Tbl_all'; mkdirs(tldir, mkstatus)
    tpdir = f'{otdir}/Outputs/Tbl_temp'; mkdirs(tpdir, mkstatus)
    mddir = f'{otdir}/Outputs/models'; mkdirs(mddir, mkstatus)
    # -------------------------------------------------------------------------
    ckpt = CF.get('ENV', 'ckpt')
    device = CF.get('ENV', 'device')
    eindex = CF.get('ENV', 'eindex')
    ereverse = False if eindex in ['Brier'] else True
    kfold = int(CF.get('ENV', 'kfold'))
    seed = int(CF.get('ENV', 'seed'))
    topim = int(CF.get('ENV', 'topim'))
    ntrials = int(CF.get('ENV', 'ntrials'))
    njobs = int(CF.get('ENV', 'njobs'))
    nbootstraps = int(CF.get('ENV', 'nbootstraps'))
    nmodels = int(CF.get('ENV', 'nmodels'))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Logging configuration. Global logging switch; set to False to disable all log output.
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Determine how to handle logging based on the verbose flag
    if verbose:
        logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', stream = sys.stdout, force = True)
    else:
        logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s', force = True, filename = f'{tpdir}/{this}.log', encoding = 'utf-8')
    
    logging.info('Loading global parameters from configuration file config.txt')
    logging.info(f'Root working directory rtdir: {rtdir}')
    logging.info(f'Data loading directory dtdir: {dtdir}')
    logging.info(f'Number of folds kfold = {kfold}; Number of trials per optimization ntrials = {ntrials}')
    logging.info(f'Number of top potential features topim = {topim}; Random seed seed = {seed}')
    logging.info(f'Number of threads njobs = {njobs}; Number of bootstrap resamples nbootstraps = {nbootstraps}')
    logging.info(f'Final evaluation metric eindex = {eindex}; Sort descending ereverse = {ereverse}')
    logging.info(f'TabPFN checkpoint ckpt = {ckpt}')
    logging.info(f'Number of models included in the ensemble nmodels = {nmodels}')
    
    # ------------------------------------
    normal = modeldict[this]['normal']
    abnormal = modeldict[this]['abnormal']
    delcols = modeldict[this]['delcols']
    selffeatures = modeldict[this]['features']
    best_num = modeldict[this]['best_num']
    
    logging.info(f'Currently processing model: {this}')
    logging.info(f'normal: {normal}')
    logging.info(f'abnormal: {abnormal}')
    logging.info(f'delcols: {delcols}')
     
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Load Data
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    logging.info('Loading data ...')
    
    ttfile = f'{rtdir}/data/train-test-data.xlsx'
    vnfile = f'{rtdir}/data/validation-data.xlsx'
    tefile = f'{rtdir}/data/test-data.xlsx'
    
    dattt = pd.read_excel(ttfile); dattt2 = dattt.copy()
    datvn = pd.read_excel(vnfile)
    
    logging.info(f'Data loading complete. Loaded train-test dataset: \n\n{dattt} \n\n Loaded external independent validation dataset:\n\n{datvn}')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Data Preprocessing and Splitting
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # 1. Pre-exclude variables not included in the model
    logging.info(f'Pre-excluding variables not included in the model: {delcols}')
    for delcol in delcols: del dattt[delcol]; del datvn[delcol]
    
    # 2. Label processing
    logging.info(f'Label processing: {normal} vs {abnormal}')
    dattt = dattt[dattt['diagnose'].isin(normal + abnormal)]
    datvn = datvn[datvn['diagnose'].isin(normal + abnormal)]
    
    datvn['diagnose'] = (datvn['diagnose'].isin(abnormal)).astype(int)
    dattt['diagnose'] = (dattt['diagnose'].isin(abnormal)).astype(int)
    
    # 3. Split features and target variable
    X = dattt.drop(['diagnose'], axis = 1)
    y = dattt['diagnose']
    
    all_features = X.columns.tolist()
    
    # 4. Globally unique train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.3,
        random_state = seed,
        stratify = y
    )
    
    if this == 'model12':
        dattest = pd.concat([y_test, X_test_raw], axis = 1)
        dattest = dattt2.loc[dattest.index, :]
        dattest.to_excel(tefile, index = False)
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 1: Initial Feature Importance Ranking (!! Enhanced via Cross-Validation)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if selffeatures == None:
        
        logging.info('Step 1: Enhancing initial feature importance ranking using cross-validation ...')
        
        # Globally define a preprocessing pipeline for feature transformation and standardization
        # `log1p`: Applies log transformation to feature data, typically to compress the dynamic range, especially when feature values span multiple orders of magnitude
        # `scaler`: Standardizes each feature to have a mean of 0 and a standard deviation of 1 (Z-score scaling)
        logging.info('Globally defining a preprocessing pipeline for feature transformation and standardization')
        preprocessor_for_ranking = Pipeline(steps = [
            ('log1p', FunctionTransformer(np.log1p, validate = False)),
            ('scaler', StandardScaler())
        ])
        logging.info(preprocessor_for_ranking)
        
        # Use N-fold cross-validation to obtain more robust feature importances
        logging.info(f'Using {kfold}-fold cross-validation to obtain robust feature importances')
        cv_ranking = KFold(n_splits = kfold, shuffle = True, random_state = seed)
        
        importance_list = []
        for fold, (train_idx, val_idx) in enumerate(cv_ranking.split(X_train_raw, y_train)):
            X_train_fold, y_train_fold = X_train_raw.iloc[train_idx], y_train.iloc[train_idx]
            
            # Fit and transform the preprocessor within each fold to prevent data leakage
            X_train_fold_processed = preprocessor_for_ranking.fit_transform(X_train_fold)
            X_train_fold_processed = X_train_fold_processed.values if isinstance(X_train_fold_processed, pd.DataFrame) else X_train_fold_processed
            
            lgbm_clf_ranking = lgb.LGBMClassifier(random_state = seed, verbose = -1)
            lgbm_clf_ranking.fit(X_train_fold_processed, y_train_fold)
            
            importance_list.append(lgbm_clf_ranking.feature_importances_)
            logging.info(f'Completed fold {fold + 1}/{kfold} cross-validation ...')
    
        # Calculate mean feature importance
        mean_importances = np.mean(importance_list, axis = 0)
        
        feature_importance_df = pd.DataFrame({
            'Feature': all_features,
            'Importance': mean_importances
        }).sort_values(by = 'Importance', ascending = False)
        
        tmp = ', '.join([f"{row['Feature']} ({row['Importance']})" for _, row in feature_importance_df.head(6).iterrows()])
        logging.info(f'Importance of top 6 features: {tmp}')
    
        # Plot the ranking of the top N feature importances
        if savefig:
            
            logging.info(f'Plotting the top {topim} feature importance ranking chart')
            
            top_features = feature_importance_df.nlargest(topim, 'Importance')
        
            # Resolve the issue where minus signs are displayed as blocks after changing fonts
            plt.rcParams['axes.unicode_minus'] = False
        
            # Dynamically calculate chart dimensions to prevent label overlap
            fig_height = topim * 0.2 + 2 
            fig_width = 6
            
            # Create chart using object-oriented approach (fig, ax)
            fig, ax = plt.subplots(figsize = (fig_width, fig_height), dpi = 300)
            
            # Plot horizontal bar chart
            bars = ax.barh(top_features['Feature'], top_features['Importance'], color = 'steelblue')
            
            # Add specific values to the end of each bar
            for bar in bars:
                width = bar.get_width()
                # Leave a small gap between the value label and the end of the bar
                ax.text(width + ax.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va = 'center', ha = 'left', fontsize = 12)
            
            # Set labels and title
            ax.set_xlabel(f'Average Importance over {kfold} Folds', fontsize = 14)
            ax.set_ylabel('Feature', fontsize = 14)
            ax.set_title(f'Top {topim} Feature Importance (from robust ranking)', fontsize = 16, fontweight = 'bold')
            
            # Optimize axes and grid
            ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
            ax.invert_yaxis()
            
            # Hide top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Set thickness of left and bottom spines
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            
            # Add horizontal grid lines behind the bars
            ax.grid(axis = 'x', linestyle = '--', color = 'gray', alpha = 0.6)
            ax.set_axisbelow(True)
            
            # Remove top and bottom margins of Y-axis to make the chart more compact
            ax.margins(y = 0)
            
            # Automatically adjust layout to eliminate redundant margins
            fig.tight_layout()
            
            # Save as high-resolution images
            plt.savefig(f'{fgdir1}/Feature_Ranking_{this}.png', bbox_inches = 'tight')
            plt.savefig(f'{fgdir1}/Feature_Ranking_{this}.pdf', bbox_inches = 'tight')
            plt.close()
            
            logging.info(f'{fgdir1}/Feature_Ranking_{this}.png')
            logging.info(f'{fgdir1}/Feature_Ranking_{this}.pdf')   
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 2: Rigorous K-Fold Recursive Feature Selection
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if selffeatures == None:
        
        logging.info(f'Step 2: Using rigorous {kfold}-fold cross-validation to perform further feature selection ...')
        kf = KFold(n_splits = kfold, shuffle = True, random_state = seed)
        
        selection_results_list = []
        fold_columns = [f'Fold_{i+1}_ROC' for i in range(kf.get_n_splits())]
        for i in range(1, len(top_features) + 1):
            current_features = top_features['Feature'].iloc[:i].tolist()
            last_added_feature = top_features.iloc[i-1]['Feature']
            feature_importance = top_features.iloc[i-1]['Importance']
            logging.info(f'Evaluating the top {i} features, newly added: {last_added_feature}')
            fold_roc_scores = []
            for train_idx, val_idx in kf.split(X_train_raw, y_train):
                X_train_fold, y_train_fold = X_train_raw.iloc[train_idx], y_train.iloc[train_idx]
                X_val_fold, y_val_fold = X_train_raw.iloc[val_idx], y_train.iloc[val_idx]
                fold_preprocessor = Pipeline(steps = [
                    ('log1p', FunctionTransformer(np.log1p, validate = False)),
                    ('scaler', StandardScaler())
                ])
                fold_preprocessor.set_output(transform = 'pandas')
                X_train_fold_processed = fold_preprocessor.fit_transform(X_train_fold[current_features])
                X_val_fold_processed = fold_preprocessor.transform(X_val_fold[current_features])
                lgbm_clf_fold = lgb.LGBMClassifier(random_state = seed, verbose = -1)
                lgbm_clf_fold.fit(X_train_fold_processed, y_train_fold)
                y_val_proba = lgbm_clf_fold.predict_proba(X_val_fold_processed)[:, 1]
                fold_roc_score = roc_auc_score(y_val_fold, y_val_proba)
                fold_roc_scores.append(fold_roc_score)
            mean_roc_score = np.mean(fold_roc_scores)
            n_folds = len(fold_roc_scores)
            if n_folds > 1:
                std_err = stats.sem(fold_roc_scores)
                t_value = stats.t.ppf(0.975, df = n_folds - 1)
                ci_lower = mean_roc_score - t_value * std_err
                ci_upper = mean_roc_score + t_value * std_err
                ci_upper = min(ci_upper, 1.0)
            else:
                ci_lower, ci_upper = None, None
            row_data = {'Feature': last_added_feature, 'Num_Features': i, 'Importance': feature_importance, 'Mean_ROC': mean_roc_score, 'CI_Lower': ci_lower, 'CI_Upper': ci_upper}
            for j, score in enumerate(fold_roc_scores):
                row_data[fold_columns[j]] = score
            selection_results_list.append(row_data)
            
        logging.info('Feature selection complete!')
        
        selection_results = pd.DataFrame(selection_results_list)
        total_importance = selection_results['Importance'].sum()
        selection_results['Importance'] = selection_results['Importance'] / total_importance if total_importance > 0 else 0
        final_columns_order = ['Feature', 'Num_Features', 'Importance', 'Mean_ROC', 'CI_Lower', 'CI_Upper'] + fold_columns
        selection_results = selection_results[final_columns_order]
        if savetbl: selection_results.to_excel(f'{tldir}/Feature_Curve_{this}.xlsx', index = False)
        if savetbl: logging.info(f'{tldir}/Feature_Curve_{this}.xlsx')
        
        best_num_features = int(selection_results.loc[selection_results['Mean_ROC'].idxmax()]['Num_Features'])
        best_num_features = best_num_features if best_num == None else best_num
        
        logging.info(f'After recursive feature selection, the top {best_num_features} features are highlighted')
        
        if savefig:
            
            logging.info('Plotting the Feature AUC Curve')
            
            fig, ax1 = plt.subplots(figsize = (9, fig_height), dpi = 300)
            norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
            colors = plt.cm.Blues(norm(selection_results['Importance']))
            ax1.bar(selection_results['Feature'], selection_results['Importance'], color = colors, label = 'Feature Importance')
            ax1.set_xlabel('Features', fontsize = 16, fontweight = 'bold')
            ax1.set_ylabel('Feature Importance', fontsize = 16, fontweight = 'bold')
            ax1.tick_params(axis = 'y', labelsize = 12, width = 1.5)
            x_labels = selection_results['Feature']
            x_colors = ['red' if i < best_num_features else 'black' for i in range(len(x_labels))]
            ax1.tick_params(axis = 'x', rotation = 90, labelsize = 12, width = 1.5)
            ax1.margins(x = 0)
            for tick_label, color in zip(ax1.get_xticklabels(), x_colors):
                tick_label.set_color(color)
            ax2 = ax1.twinx()
            ax2.plot(selection_results['Feature'][best_num_features - 1:], selection_results['Mean_ROC'][best_num_features - 1:], color = 'black', marker = 'o', linestyle = '-', label = 'Mean AUC (Other Features)')
            ax2.plot(selection_results['Feature'][:best_num_features], selection_results['Mean_ROC'][:best_num_features], color = 'red', marker = 'o', linestyle = '-', label = 'Mean AUC (Top Features)')
            ax2.fill_between(selection_results['Feature'], selection_results['CI_Lower'], selection_results['CI_Upper'], color = 'red', alpha = 0.2)
            ax2.set_ylabel('Mean AUC', fontsize = 16, fontweight = 'bold')
            ax2.tick_params(axis = 'y', labelsize = 12, width = 1.5)
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
            plt.title(f'Feature Contribution and AUC Performance (Top {best_num_features} Highlighted)', fontsize = 14, fontweight = 'bold')
            fig.tight_layout()
            plt.savefig(f'{fgdir2}/Feature_Curve_{this}.png', bbox_inches = 'tight')
            plt.savefig(f'{fgdir2}/Feature_Curve_{this}.pdf', bbox_inches = 'tight')
            plt.close()
            logging.info(f'{fgdir2}/Feature_Curve_{this}.png')
            logging.info(f'{fgdir2}/Feature_Curve_{this}.pdf')
    
    else:
        best_num_features = 3
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Final Model Training and Evaluation Using Selected Features
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Step 3: Training the model using the final feature set ...')
    
    if selffeatures == None:
        logging.info(f"Based on K-Fold evaluation, the optimal number of features is: {best_num_features}, with a corresponding mean AUC of: {selection_results['Mean_ROC'].max():.4f}")
        final_features = list(selection_results['Feature'])[:best_num_features]
    final_features = final_features if selffeatures == None else selffeatures
    
    logging.info(f'Final selected features: {final_features}')
    
    X_train_final_raw = X_train_raw[final_features]
    X_test_final_raw = X_test_raw[final_features]
    final_preprocessor = Pipeline(steps = [
        ('log1p', FunctionTransformer(np.log1p, validate = False)),
        ('scaler', StandardScaler())
    ])
    X_train_final_processed = final_preprocessor.fit_transform(X_train_final_raw)
    X_test_final_processed = final_preprocessor.transform(X_test_final_raw)
    X_train_final = pd.DataFrame(X_train_final_processed, index = X_train_final_raw.index, columns = final_features)
    X_test_final = pd.DataFrame(X_test_final_processed, index = X_test_final_raw.index, columns = final_features)
    
    # ---------------------------------
    # Nested Cross-Validation for Robust Performance Evaluation
    # ---------------------------------
    logging.info('Performing nested cross-validation to evaluate the LightGBM modeling pipeline ...')
    model_lgbm_nested = LGBMClassifier(random_state = seed, verbose = -1)
    param_grid_lgbm_nested = {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.05, 0.01, 0.1], 'num_leaves': [11, 21, 31, 41]}
    outer_cv = KFold(n_splits = kfold, shuffle = True, random_state = seed)
    inner_cv = KFold(n_splits = kfold, shuffle = True, random_state = seed)
    outer_loop_scores = []
    
    logging.info(f'Executing {outer_cv.get_n_splits()}-fold nested cross-validation on the final {best_num_features} selected features ...')
    fold_num = 1
    for train_idx, test_idx in outer_cv.split(X_train_final, y_train):
        X_train_outer, X_test_outer = X_train_final.iloc[train_idx], X_train_final.iloc[test_idx]
        y_train_outer, y_test_outer = y_train.iloc[train_idx], y_train.iloc[test_idx]
        grid_search_inner = GridSearchCV(estimator = model_lgbm_nested, param_grid = param_grid_lgbm_nested, scoring = 'roc_auc', cv = inner_cv, n_jobs = -1, verbose = 0)
        grid_search_inner.fit(X_train_outer, y_train_outer)
        best_model_inner = grid_search_inner.best_estimator_
        y_pred_proba_outer = best_model_inner.predict_proba(X_test_outer)[:, 1]
        auc_score = roc_auc_score(y_test_outer, y_pred_proba_outer)
        outer_loop_scores.append(auc_score)
        logging.info(f'Outer fold {fold_num}/{outer_cv.get_n_splits()} completed | AUC: {auc_score:.4f} | Best parameters found: {grid_search_inner.best_params_}')
        fold_num += 1
        
    logging.info('Nested cross-validation performance evaluation results (LightGBM):')
    
    mean_auc = np.mean(outer_loop_scores)
    std_auc = np.std(outer_loop_scores)
    n_outer_folds = len(outer_loop_scores)
    if n_outer_folds > 1:
        t_value = stats.t.ppf(0.975, df = n_outer_folds - 1)
        ci_margin = t_value * (std_auc / np.sqrt(n_outer_folds))
        ci_lower = mean_auc - ci_margin
        ci_upper = mean_auc + ci_margin
        logging.info(f'Model Mean AUC: {mean_auc:.4f}')
        logging.info(f'AUC Standard Deviation: {std_auc:.4f}')
        logging.info(f'95% Confidence Interval (CI): [{ci_lower:.4f}, {ci_upper:.4f}]')
    else:
        logging.info(f'Model Mean AUC: {mean_auc:.4f} (Only 1 fold available; confidence interval cannot be calculated)')
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 4: Bayesian Hyperparameter Optimization via Optuna & Final Model Training
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Step 4: Performing Bayesian optimization with Optuna and training the final model for deployment ...')
    
    # Reduce Optuna logging verbosity
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Define the parameter search space for each model
    def get_param_space(trial, model_name):
        if model_name == "Random Forest":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            }
        if model_name == "XGBoost":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log = True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
            }
        if model_name == "LightGBM":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log = True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
        if model_name == "SVM":
            params = {
                'C': trial.suggest_float('C', 0.1, 100, log = True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            }
            if params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)
            if params['kernel'] in ['rbf', 'poly']:
                params['gamma'] = trial.suggest_float('gamma', 1e-4, 1.0, log = True)
            return params
        if model_name == "KNN":
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 21, step = 2),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_int('p', 1, 2),
            }
        if model_name == "Decision Tree":
            return {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            }
        if model_name == "Extra Trees":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            }
        if model_name == "ANN":
            return {
                'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', ['50', '100', '50,25', '100,50']),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log = True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log = True),
            }
            
        if model_name == "CatBoost":
            return {
                'iterations': trial.suggest_int('iterations', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log = True),
                'depth': trial.suggest_int('depth', 4, 10),
            }
        if model_name == "Gradient Boosting":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log = True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
            }
        if model_name == "Logistic Regression":
            return {
                'C': trial.suggest_float('C', 0.01, 100, log = True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            }
        if model_name == "AdaBoost":
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log = True),
            }
        return {}
    
    # Define the models to be optimized
    models_to_tune = {
        # 1.
        'Random Forest': RandomForestClassifier,
        # 2.
        'XGBoost': XGBClassifier,
        # 3.
        'LightGBM': LGBMClassifier,
        # 4.
        'SVM': SVC,
        # 5.
        'KNN': KNeighborsClassifier,
        # 6.
        'Decision Tree': DecisionTreeClassifier,
        # 7.
        'Extra Trees': ExtraTreesClassifier,
        # 8.
        'ANN': MLPClassifier,
        # 9.
        'CatBoost': CatBoostClassifier,
        # 10.
        'Gradient Boosting': GradientBoostingClassifier,
        # 11.
        'Logistic Regression': LogisticRegression,
        # 12.
        'AdaBoost': AdaBoostClassifier,
    }
    
    # Dictionary to store the optimized best models
    best_models = {}
    
    # Loop through and optimize each model
    for model_name, model_class in models_to_tune.items():
        logging.info(f'Optimizing {model_name} with Optuna ...')
        
        def objective(trial):
            
            # Retrieve the hyperparameter search space
            params = get_param_space(trial, model_name)
            
            # Dynamically build the model parameter dictionary
            model_kwargs = params.copy()
        
            # Decode the string back to a tuple inside the objective function
            if model_name == 'ANN':
                hls_str = model_kwargs['hidden_layer_sizes']
                hls_tuple = tuple(map(int, hls_str.split(',')))
                model_kwargs['hidden_layer_sizes'] = hls_tuple
    
            # Only add random_state for models that accept it
            if model_name not in ['KNN', 'SVM']:
                 model_kwargs['random_state'] = seed

            # Add other fixed parameters for specific models
            if model_name == 'XGBoost':
                model_kwargs.update({'eval_metric': 'logloss'})
            if model_name == 'LightGBM':
                model_kwargs.update({'verbose': -1})
            if model_name == 'CatBoost':
                model_kwargs.update({'verbose': 0, 'allow_writing_files': False})
            if model_name == 'SVM':
                model_kwargs.update({'probability': True})
            if model_name == 'ANN':
                model_kwargs.update({'max_iter': 10000, 'solver': 'adam', 'early_stopping': True, 'validation_fraction': 0.1, 'n_iter_no_change': 10})
            if model_name == 'Logistic Regression':
                model_kwargs.update({'solver': 'liblinear'})

            model = model_class(**model_kwargs)
            
            # Evaluate performance using cross-validation, aiming to maximize the mean AUC
            score = cross_val_score(model, X_train_final, y_train, n_jobs = njobs, cv = kfold, scoring = 'roc_auc').mean()
            return score

        study = optuna.create_study(direction = 'maximize')
        study.optimize(objective, n_trials = ntrials)
        
        # Train the final model on the entire training dataset using the best parameters found
        best_params = study.best_params
        logging.info(f'{model_name} Best Params: {best_params}')
        
        # Dynamically build the final model parameter dictionary
        final_model_kwargs = best_params.copy()
        
        # Decoding is also required when creating the final model
        if model_name == 'ANN':
            hls_str = final_model_kwargs['hidden_layer_sizes']
            hls_tuple = tuple(map(int, hls_str.split(',')))
            final_model_kwargs['hidden_layer_sizes'] = hls_tuple
        
        # Ensure random_state is only added to appropriate models again
        if model_name not in ['KNN', 'SVM']:
             final_model_kwargs['random_state'] = seed
        
        # Add other fixed parameters for specific models
        if model_name == 'XGBoost':
            final_model_kwargs.update({'eval_metric': 'logloss'})
        if model_name == 'LightGBM':
            final_model_kwargs.update({'verbose': -1})
        if model_name == 'CatBoost':
            final_model_kwargs.update({'verbose': 0, 'allow_writing_files': False})
        if model_name == 'SVM':
            final_model_kwargs.update({'probability': True})
        if model_name == 'ANN':
            final_model_kwargs.update({'max_iter': 10000, 'solver': 'adam', 'early_stopping': True, 'validation_fraction': 0.1, 'n_iter_no_change': 10})
        if model_name == 'Logistic Regression':
            final_model_kwargs.update({'solver': 'liblinear'})
            
        final_model = model_class(**final_model_kwargs)
        final_model.fit(X_train_final, y_train)
        
        # Store the trained model in the dictionary
        best_models[model_name] = final_model

    # Training non-tunable models
    logging.info('Training non-tunable models ...')
    
    # 13. LDA
    best_models['LDA'] = LinearDiscriminantAnalysis().fit(X_train_final, y_train)
    logging.info('LDA model trained.')

    # 14. Naive Bayes
    best_models['Naive Bayes'] = GaussianNB().fit(X_train_final, y_train)
    logging.info('Naive Bayes model trained.')

    # 15. TabPFN 
    logging.info('Training TabPFN model... (This may take a moment)')
    best_models['TabPFN'] = TabPFNClassifier(device = device, model_path = ckpt).fit(X_train_final, y_train)
    logging.info('TabPFN model trained.')
    
    # Retrieve the optimized model instances from the dictionary
    best_model_lgbm = best_models['LightGBM']
    best_model_rf = best_models['Random Forest']
    best_model_xgb = best_models['XGBoost']
    best_model_gbm = best_models['Gradient Boosting']
    best_model_svm = best_models['SVM']
    best_model_knn = best_models['KNN']
    best_model_dt = best_models['Decision Tree']
    best_model_et = best_models['Extra Trees']
    best_model_ann = best_models['ANN']
    best_model_cat = best_models['CatBoost']
    best_model_lr = best_models['Logistic Regression']
    best_model_ada = best_models['AdaBoost']
    best_model_lda = best_models['LDA']
    best_model_nb = best_models['Naive Bayes']
    best_model_tabpfn = best_models['TabPFN']
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Evaluate Models on the Independent Test Set and Plot ROC Curves
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Step 5: Evaluating final models ...')
    
    # Add models to the evaluation dictionary
    models = {
        'LightGBM': {'model': best_model_lgbm, 'color': '#F7C2CD'},
        'Random Forest': {'model': best_model_rf, 'color': '#FDD379'},
        'XGBoost': {'model': best_model_xgb, 'color': '#B0D9A5'},
        'Gradient Boosting': {'model': best_model_gbm, 'color': '#A6DAEF'},
        'SVM': {'model': best_model_svm, 'color': '#F06292'},
        'KNN': {'model': best_model_knn, 'color': '#64B5F6'},
        'Decision Tree': {'model': best_model_dt, 'color': '#FFB74D'},
        'Extra Trees': {'model': best_model_et, 'color': '#9575CD'},
        'ANN': {'model': best_model_ann, 'color': '#4DB6AC'},
        'CatBoost': {'model': best_model_cat, 'color': '#2DE62C'},
        'Logistic Regression': {'model': best_model_lr, 'color': '#FFD180'},
        'AdaBoost': {'model': best_model_ada, 'color': '#CE93D8'},
        'LDA': {'model': best_model_lda, 'color': '#BCAAA4'},
        'Naive Bayes': {'model': best_model_nb, 'color': '#80CBC4'},
        'TabPFN': {'model': best_model_tabpfn, 'color': '#F33A3B'}
    }
    
    best_test_auc = -1
    best_model_name = None
    best_model_object = None
    
    # Common parameters for plotting
    figsize = (5, 4.8)
    dpi = 300
    linewidth = 0.8
    fontsize1 = 10
    fontsize2 = 8
    
    # 1. Plot the ROC curves for the training set
    if savefig:
    
        logging.info('Plotting the training set ROC curves ...')
        plt.figure(figsize = figsize, dpi = dpi)
        for name, info in models.items():
            y_pred_proba_train = info['model'].predict_proba(X_train_final)[:, 1]
            fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
            roc_auc_train, ci_lower_train, ci_upper_train = calculate_auc_ci(y_train, y_pred_proba_train, n_bootstraps = nbootstraps, seed = seed)
            label_text = f'{name}: AUC = {roc_auc_train:.3f} (95% CI: {ci_lower_train:.3f}-{ci_upper_train:.3f})'
            linestyle = '--' if name == finalmodel else '-'
            linewidth_ = 2 if name == finalmodel else 1.5
            plt.plot(fpr_train, tpr_train, label = label_text, color = info['color'], linewidth = linewidth_, linestyle = linestyle)
        
        # plt.plot([0, 1], [0, 1], 'k--', linewidth = linewidth, alpha = 0.8)
        # if False: plt.title('ROC Curves - Training Set', fontsize = fontsize1, fontweight = 'bold')
        # plt.xlabel('False Positive Rate (1-Specificity)', fontsize = fontsize1)
        # plt.ylabel('True Positive Rate (Sensitivity)', fontsize = fontsize1)
        # plt.xticks(fontsize = fontsize1); plt.yticks(fontsize = fontsize1)
        # plt.legend(loc = 'lower right', fontsize = fontsize2)
        # plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
        # plt.gca().spines['left'].set_linewidth(linewidth); plt.gca().spines['bottom'].set_linewidth(linewidth)
        # plt.grid(False); plt.tight_layout()
        # plt.savefig(f'{fgdir3}/Training_{this}.png', bbox_inches = 'tight')
        # plt.savefig(f'{fgdir3}/Training_{this}.pdf', bbox_inches = 'tight')
        # plt.close()
        
        # logging.info(f'{fgdir3}/Training_{this}.png')
        # logging.info(f'{fgdir3}/Training_{this}.pdf')

    # 2. Plot the ROC curves for the test set
    if savefig:
        
        # Store test set results for DeLong's test
        test_set_results = []
        
        logging.info('Plotting the test set ROC curves ...')
        plt.figure(figsize = figsize, dpi = dpi)
        for name, info in models.items():
            y_pred_proba_test = info['model'].predict_proba(X_test_final)[:, 1]
            fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
            roc_auc_test, ci_lower_test, ci_upper_test = calculate_auc_ci(y_test, y_pred_proba_test, n_bootstraps = nbootstraps, seed = seed)
            
            # Store results for DeLong's test
            metric_names = ['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'F1 score', 'Kappa', 'Brier']
            metric_values = calculate_metrics(y_test, y_pred_proba_test)
            metrics_dict = dict(zip(metric_names, metric_values))
            tmp = {'name': name, 'auc': roc_auc_test, 'scores': y_pred_proba_test}
            tmp.update(metrics_dict)
            test_set_results.append(tmp)

            label_text = f'{name}: AUC = {roc_auc_test:.3f} (95% CI: {ci_lower_test:.3f}-{ci_upper_test:.3f})'
            linestyle = '--' if name == finalmodel else '-'
            linewidth_ = 2 if name == finalmodel else 1.5
            plt.plot(fpr_test, tpr_test, label = label_text, color = info['color'], linewidth = linewidth_, linestyle = linestyle)
            
            if roc_auc_test > best_test_auc:
                best_test_auc = roc_auc_test
                best_model_name = name
                best_model_object = info['model']
        
        # plt.plot([0, 1], [0, 1], 'k--', linewidth = linewidth, alpha = 0.8)
        # if False: plt.title('ROC Curves - Test Set', fontsize = 20, fontweight = 'bold')
        # plt.xlabel('False Positive Rate (1-Specificity)', fontsize = fontsize1)
        # plt.ylabel('True Positive Rate (Sensitivity)', fontsize = fontsize1)
        # plt.xticks(fontsize = fontsize1); plt.yticks(fontsize = fontsize1)
        # plt.legend(loc = 'lower right', fontsize = fontsize2)
        # plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
        # plt.gca().spines['left'].set_linewidth(linewidth); plt.gca().spines['bottom'].set_linewidth(linewidth)
        # plt.grid(False); plt.tight_layout()
        # plt.savefig(f'{fgdir3}/Test_{this}.png', bbox_inches = 'tight')
        # plt.savefig(f'{fgdir3}/Test_{this}.pdf', bbox_inches = 'tight')
        # plt.close()

        # logging.info(f'{fgdir3}/Test_{this}.png')
        # logging.info(f'{fgdir3}/Test_{this}.pdf')
        
    # Perform DeLong's test to compare the best and second-best models
    if len(test_set_results) >= 2:
       
        # Sort by AUC in descending order
        test_set_results.sort(key = lambda x: x[eindex], reverse = ereverse)
        model1 = test_set_results[0]
        model2 = test_set_results[1]
        
        logging.info("DeLong's Test for AUC Comparison (Test Set)")
        logging.info(f"Comparing '{model1['name']}' (AUC={model1['auc']:.4f}, {eindex}={model1[eindex]:.4f}) vs. '{model2['name']}' (AUC={model2['auc']:.4f}, {eindex}={model2[eindex]:.4f})")
        
        z_score, p_value = delong_test(y_test, model1['scores'], model2['scores'])
        
        logging.info(f'Z-score: {z_score:.4f}, P-value: {p_value:.4f}')
        if p_value < 0.05:
            logging.info(f"Conclusion: The difference in AUC is statistically significant. '{model1['name']}' is superior.")
        else:
            logging.info('Conclusion: The difference in AUC is not statistically significant.')
        
    # Calculate and display metrics for each model on the test set
    logging.info('Calculating and displaying metrics for each model on the test set ...')
    metrics_dict = {'Metrics': ['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'F1 score', 'Kappa', 'Brier']}
    for model_name, info in models.items():
        y_pred_proba = info["model"].predict_proba(X_test_final)[:, 1]
        metrics_dict[model_name] = calculate_metrics(y_test, y_pred_proba)

    metrics_df_test = pd.DataFrame(metrics_dict)
    logging.info('Test Set Performance Metrics\n\n' + metrics_df_test.round(4).to_string() + '\n')
    
    metrics_df_test.to_excel(f'{tldir}/Test_Set_Metrics_{this}.xlsx', index = False)
    logging.info(f'{tldir}/Test_Metrics_{this}.xlsx')
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 6: Final Validation on the External Validation Set
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Step 6: Performing final validation on the external validation set ...')
    
    if 'datvn' in locals() and not datvn.empty:
        X_val_raw = datvn[final_features]
        y_val = datvn['diagnose']
        
        X_val_processed = final_preprocessor.transform(X_val_raw)
        X_val_final = pd.DataFrame(X_val_processed, index=X_val_raw.index, columns=final_features)
        
        # Store validation set results for DeLong's test
        validation_set_results = []

        # Plot the ROC curves for the external validation set
        if savefig:
            
            logging.info('Plotting the external validation set ROC curves ...')
            plt.figure(figsize = figsize, dpi = dpi)
            for name, info in models.items():
                y_pred_proba_val = info['model'].predict_proba(X_val_final)[:, 1]
                fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_proba_val)
                roc_auc_val, ci_lower_val, ci_upper_val = calculate_auc_ci(y_val, y_pred_proba_val, n_bootstraps = nbootstraps, seed = seed)
                
                # Store results for DeLong's test
                metric_names = ['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'F1 score', 'Kappa', 'Brier']
                metric_values = calculate_metrics(y_val, y_pred_proba_val)
                metrics_dict = dict(zip(metric_names, metric_values))
                tmp = {'name': name, 'auc': roc_auc_val, 'scores': y_pred_proba_val}
                tmp.update(metrics_dict)
                validation_set_results.append(tmp)
                
                label_text = f"{name}: AUC={roc_auc_val:.3f} (95% CI: {ci_lower_val:.3f}-{ci_upper_val:.3f})"
                linestyle = '--' if name == finalmodel else '-'
                linewidth_ = 2 if name == finalmodel else 1.5
                plt.plot(fpr_val, tpr_val, label = label_text, color = info["color"], linewidth = linewidth_, linestyle = linestyle)
            
            # plt.plot([0, 1], [0, 1], 'k--', linewidth = linewidth, alpha = 0.8)
            # if False: plt.title('ROC Curves - External Validation Set', fontsize = fontsize1, fontweight='bold')
            # plt.xlabel('False Positive Rate (1-Specificity)', fontsize = fontsize1)
            # plt.ylabel('True Positive Rate (Sensitivity)', fontsize = fontsize1)
            # plt.xticks(fontsize = fontsize1); plt.yticks(fontsize = fontsize1)
            # plt.legend(loc='lower right', fontsize = fontsize2)
            # plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
            # plt.gca().spines['left'].set_linewidth(linewidth); plt.gca().spines['bottom'].set_linewidth(linewidth)
            # plt.grid(False); plt.tight_layout()
            # plt.savefig(f'{fgdir3}/External_Validation_{this}.png', bbox_inches = 'tight')
            # plt.savefig(f'{fgdir3}/External_Validation_{this}.pdf', bbox_inches = 'tight')
            # plt.close()
    
            # logging.info(f'{fgdir3}/External_Validation_{this}.png')
            # logging.info(f'{fgdir3}/External_Validation_{this}.pdf')
        
        # Perform DeLong's test to compare the best and second-best models (Validation Set)
        if len(validation_set_results) >= 2:
            
            validation_set_results.sort(key = lambda x: x[eindex], reverse = ereverse)
            model1_val = validation_set_results[0]
            model2_val = validation_set_results[1]
            
            logging.info("DeLong's Test for AUC Comparison (External Validation Set)")
            logging.info(f"Comparing '{model1_val['name']}' (AUC={model1_val['auc']:.4f}, {eindex}={model1_val[eindex]:.4f}) vs. '{model2_val['name']}' (AUC={model2_val['auc']:.4f}, {eindex}={model2_val[eindex]:.4f})")
            
            z_score_val, p_value_val = delong_test(y_val, model1_val['scores'], model2_val['scores'])
            
            logging.info(f'Z-score: {z_score_val:.4f}, P-value: {p_value_val:.4f}')
            if p_value_val < 0.05:
                logging.info(f"Conclusion: The difference in AUC is statistically significant. '{model1_val['name']}' is superior.")
            else:
                logging.info('Conclusion: The difference in AUC is not statistically significant.')

        # Calculate and display metrics for each model on the external validation set
        logging.info('Calculating external validation set performance metrics ...')
        metrics_dict_val = {'Metrics': ['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'F1 score', 'Kappa', 'Brier']}
        for model_name, info in models.items():
            y_pred_proba_val = info['model'].predict_proba(X_val_final)[:, 1]
            metrics_dict_val[model_name] = calculate_metrics(y_val, y_pred_proba_val)

        metrics_df_val = pd.DataFrame(metrics_dict_val)
        logging.info('External Validation Set Performance Metrics\n\n' + metrics_df_val.round(4).to_string() + '\n')
        
        metrics_df_val.to_excel(f'{tldir}/External_Validation_Metrics_{this}.xlsx', index = False)
        logging.info(f'{tldir}/External_Validation_Metrics_{this}.xlsx')
    else:
        logging.warning('External validation data (datvn) not found or empty, skipping this step!')
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 7: Sensitivity Analysis of Voting Weights
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Step 7: Sensitivity analysis of voting weights (base on top 5 best models) ...')
    
    # 1. Filter out non-ensemble single model results
    if len(test_set_results) >= nmodels:
        
        single_model_results = [res for res in test_set_results if 'Voting' not in res['name']]
        single_model_results.sort(key = lambda x: x[eindex], reverse = ereverse)
        
        refined_estimators = []
        topnames = []
        for nin in range(nmodels):
            top_model_name = single_model_results[nin]['name']
            top_model = best_models[top_model_name]
            topnames.append(top_model_name)
            refined_estimators.append((top_model_name, top_model))
            
        logging.info(f'Weighting the top {nmodels} best-performing single models: {topnames}')

        # Estimate weights
        logging.info('Estimating the weights of the Soft Voting Ensemble model ...')
        sensitivities = np.array([item['sensitivity'] for item in single_model_results[0:nmodels]])
        specificities = np.array([item['specificity'] for item in single_model_results[0:nmodels]])
        
        # Define comprehensive weights
        weights = [1, 1, 1]

    # Add models to the evaluation dictionary
    modelsfinal = models.copy()
        
    logging.info(f'Define comprehensive weights as: {weights}')

    # 2. Define the new elite ensemble model
    refined_voting_clf = VotingClassifier(
        estimators = refined_estimators,
        voting = 'soft',
        weights = weights
    )
    
    # 3. Train this optimized model on the complete training data
    refined_voting_clf.fit(X_train_final, y_train)
    logging.info('Optimized ensemble model (Soft Voting) training completed.')

    # 4. Add this new model to the models dictionary for subsequent unified evaluation
    #    Assign a special color and name to make it stand out in the legend
    modelsfinal[finalmodel] = {
        'model': refined_voting_clf, 
        'color': '#64B5F6'
    }
    logging.info('Soft Voting has been added to the final evaluation list.')
        
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Supplementary Analysis: Learning Curve
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Supplementary Analysis: Calculating learning curve data ...')
    
    # Train the ensemble model with different proportions of training data and evaluate AUC on a fixed test set
    learning_curve_proportions = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    learning_curve_results = []
    
    for prop in learning_curve_proportions:
        n_sub = max(int(len(X_train_final) * prop), 20)
        
        # Stratified subsampling from the training set
        if prop < 1.0:
            X_sub, _, y_sub, _ = train_test_split(
                X_train_final, y_train,
                train_size = n_sub,
                random_state = seed,
                stratify = y_train
            )
        else:
            X_sub, y_sub = X_train_final.copy(), y_train.copy()
    
        # Retrain the Soft Voting ensemble model at each proportion (reusing the refined_estimators architecture)
        lc_estimators = []
        for est_name, est_obj in refined_estimators:
            # Re-instantiate to avoid using already fitted models
            lc_estimators.append((est_name, type(est_obj)(**est_obj.get_params())))
        
        lc_voting = VotingClassifier(
            estimators = lc_estimators,
            voting = 'soft',
            weights = weights
        )
        lc_voting.fit(X_sub, y_sub)
        
        # Evaluate on the fixed test set
        y_pred_proba_lc = lc_voting.predict_proba(X_test_final)[:, 1]
        auc_lc, ci_low_lc, ci_up_lc = calculate_auc_ci(
            y_test, y_pred_proba_lc,
            n_bootstraps = nbootstraps,
            seed = seed
        )
        
        learning_curve_results.append({
            'model_type'  : this,
            'proportion'  : prop,
            'n_train'     : n_sub,
            'auc_mean'    : auc_lc,
            'auc_lower'   : ci_low_lc,
            'auc_upper'   : ci_up_lc
        })
        logging.info(f'  比例 {prop:.0%} (n={n_sub}): AUC = {auc_lc:.4f} '
                     f'[{ci_low_lc:.4f}, {ci_up_lc:.4f}]')
    
    lc_df = pd.DataFrame(learning_curve_results)
    lc_df.to_csv(f'{tldir}/learning_curve_{this}.csv', index = False)
    logging.info(f'Learning curve data saved to: {tldir}/learning_curve_{this}.csv')
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 8: Re-plotting ROC Curves for Training, Test, and External Validation Sets
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Step 8: Re-plotting ROC curves for training, test, and external validation sets ...')
    
    # 1. Plot ROC curves for the training set
    if savefig:
    
        logging.info('Plotting training set ROC curves ...')
        plt.figure(figsize = figsize, dpi = dpi)
        for name, info in modelsfinal.items():
            y_pred_proba_train = info['model'].predict_proba(X_train_final)[:, 1]
            fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
            roc_auc_train, ci_lower_train, ci_upper_train = calculate_auc_ci(y_train, y_pred_proba_train, n_bootstraps = nbootstraps, seed = seed)
            label_text = f'{name}: AUC = {roc_auc_train:.3f} (95% CI: {ci_lower_train:.3f}-{ci_upper_train:.3f})'
            linestyle = '--' if name == finalmodel else '-'
            linewidth_ = 2 if name == finalmodel else 1.5
            plt.plot(fpr_train, tpr_train, label = label_text, color = info['color'], linewidth = linewidth_, linestyle = linestyle)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth = linewidth, alpha = 0.8)
        if False: plt.title('ROC Curves - Training Set', fontsize = fontsize1, fontweight = 'bold')
        plt.xlabel('False Positive Rate (1-Specificity)', fontsize = fontsize1)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize = fontsize1)
        plt.xticks(fontsize = fontsize1); plt.yticks(fontsize = fontsize1)
        plt.legend(loc = 'lower right', fontsize = fontsize2)
        plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(linewidth); plt.gca().spines['bottom'].set_linewidth(linewidth)
        plt.grid(False); plt.tight_layout()
        plt.savefig(f'{fgdir3}/Ensemble_Model_Training_{this}.png', bbox_inches = 'tight')
        plt.savefig(f'{fgdir3}/Ensemble_Model_Training_{this}.pdf', bbox_inches = 'tight')
        plt.close()
        
        logging.info(f'{fgdir3}/Ensemble_Model_Training_{this}.png')
        logging.info(f'{fgdir3}/Ensemble_Model_Training_{this}.pdf')

    # 2. Plot ROC curves for the test set
    if savefig:
        
        logging.info('Plotting test set ROC curves ...')
        
        # Store test set results for DeLong's test
        test_set_results2 = []
        
        plt.figure(figsize = figsize, dpi = dpi)
        for name, info in modelsfinal.items():
            y_pred_proba_test = info['model'].predict_proba(X_test_final)[:, 1]
            fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
            roc_auc_test, ci_lower_test, ci_upper_test = calculate_auc_ci(y_test, y_pred_proba_test, n_bootstraps = nbootstraps, seed = seed)
            
            # Store results for DeLong's test
            metric_names = ['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'F1 score', 'Kappa', 'Brier']
            metric_values = calculate_metrics(y_test, y_pred_proba_test)
            metrics_dict = dict(zip(metric_names, metric_values))
            tmp = {'name': name, 'auc': roc_auc_test, 'scores': y_pred_proba_test}
            tmp.update(metrics_dict)
            test_set_results2.append(tmp)
            
            label_text = f'{name}: AUC = {roc_auc_test:.3f} (95% CI: {ci_lower_test:.3f}-{ci_upper_test:.3f})'
            linestyle = '--' if name == finalmodel else '-'
            linewidth_ = 2 if name == finalmodel else 1.5
            plt.plot(fpr_test, tpr_test, label=label_text, color = info['color'], linewidth = linewidth_, linestyle = linestyle)
            
            if roc_auc_test > best_test_auc:
                best_test_auc = roc_auc_test
                best_model_name = name
                best_model_object = info['model']
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth = linewidth, alpha = 0.8)
        if False: plt.title('ROC Curves - Test Set', fontsize = 20, fontweight = 'bold')
        plt.xlabel('False Positive Rate (1-Specificity)', fontsize = fontsize1)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize = fontsize1)
        plt.xticks(fontsize = fontsize1); plt.yticks(fontsize = fontsize1)
        plt.legend(loc = 'lower right', fontsize = fontsize2)
        plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(linewidth); plt.gca().spines['bottom'].set_linewidth(linewidth)
        plt.grid(False); plt.tight_layout()
        plt.savefig(f'{fgdir3}/Ensemble_Model_Test_{this}.png', bbox_inches = 'tight')
        plt.savefig(f'{fgdir3}/Ensemble_Model_Test_{this}.pdf', bbox_inches = 'tight')
        plt.close()

        logging.info(f'{fgdir3}/Ensemble_Model_Test_{this}.png')
        logging.info(f'{fgdir3}/Ensemble_Model_Test_{this}.pdf')
        
        # Perform DeLong's test to compare the best and second-best models (Test Set)
        if len(test_set_results2) >= 2:
           
            # Sort by AUC in descending order
            test_set_results2.sort(key = lambda x: x[eindex], reverse = ereverse)
            model1 = test_set_results2[0]
            model2 = test_set_results2[1]
            
            logging.info("DeLong's Test for AUC Comparison (Test Set)")
            logging.info(f"Comparing '{model1['name']}' (AUC={model1['auc']:.4f}, {eindex}={model1[eindex]:.4f}) vs. '{model2['name']}' (AUC={model2['auc']:.4f}, {eindex}={model2[eindex]:.4f})")
            
            z_score, p_value = delong_test(y_test, model1['scores'], model2['scores'])
            
            logging.info(f'Z-score: {z_score:.4f}, P-value: {p_value:.4f}')
            if p_value < 0.05:
                logging.info(f"Conclusion: The difference in AUC is statistically significant. '{model1['name']}' is superior.")
            else:
                logging.info('Conclusion: The difference in AUC is not statistically significant.')
            
        # Calculate and display performance metrics for each model on the test set
        logging.info('Calculating and displaying test set performance metrics for each model ...')
        metrics_dict = {'Metrics': ['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'F1 score', 'Kappa', 'Brier']}
        for model_name, info in modelsfinal.items():
            y_pred_proba = info["model"].predict_proba(X_test_final)[:, 1]
            metrics_dict[model_name] = calculate_metrics(y_test, y_pred_proba)

        metrics_df_test = pd.DataFrame(metrics_dict)
        logging.info('Test Set Performance Metrics\n\n' + metrics_df_test.round(4).to_string() + '\n')
        
        metrics_df_test.to_excel(f'{tldir}/Ensemble_Model_Test_Metrics_{this}.xlsx', index = False)
        logging.info(f'{tldir}/Ensemble_Model_Test_Metrics_{this}.xlsx')

    # 3. Plot ROC curves for the external validation set
    if savefig:
        
        logging.info('Plotting external validation set ROC curves ...')
        
        # Store validation set results for DeLong's test
        validation_set_results2 = []

        plt.figure(figsize = figsize, dpi = dpi)
        for name, info in modelsfinal.items():
            y_pred_proba_val = info['model'].predict_proba(X_val_final)[:, 1]
            fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_proba_val)
            roc_auc_val, ci_lower_val, ci_upper_val = calculate_auc_ci(y_val, y_pred_proba_val, n_bootstraps = nbootstraps, seed = seed)
            
            # Store results for DeLong's test
            metric_names = ['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'F1 score', 'Kappa', 'Brier']
            metric_values = calculate_metrics(y_val, y_pred_proba_val)
            metrics_dict = dict(zip(metric_names, metric_values))
            tmp = {'name': name, 'auc': roc_auc_val, 'scores': y_pred_proba_val}
            tmp.update(metrics_dict)
            validation_set_results2.append(tmp)
            
            label_text = f"{name}: AUC={roc_auc_val:.3f} (95% CI: {ci_lower_val:.3f}-{ci_upper_val:.3f})"
            linestyle = '--' if name == finalmodel else '-'
            linewidth_ = 2 if name == finalmodel else 1.5
            plt.plot(fpr_val, tpr_val, label = label_text, color = info["color"], linewidth = linewidth_, linestyle = linestyle)

        plt.plot([0, 1], [0, 1], 'k--', linewidth=linewidth, alpha = 0.8)
        if False: plt.title('ROC Curves - External Validation Set', fontsize = fontsize1, fontweight = 'bold')
        plt.xlabel('False Positive Rate (1-Specificity)', fontsize = fontsize1)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize = fontsize1)
        plt.xticks(fontsize = fontsize1); plt.yticks(fontsize = fontsize1)
        plt.legend(loc='lower right', fontsize = fontsize2)
        plt.gca().spines['top'].set_visible(False); plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(linewidth); plt.gca().spines['bottom'].set_linewidth(linewidth)
        plt.grid(False); plt.tight_layout()
        plt.savefig(f'{fgdir3}/Ensemble_Model_External_Validation_{this}.png', bbox_inches = 'tight')
        plt.savefig(f'{fgdir3}/Ensemble_Model_External_Validation_{this}.pdf', bbox_inches = 'tight')
        plt.close()

        logging.info(f'{fgdir3}/Ensemble_Model_External_Validation_{this}.png')
        logging.info(f'{fgdir3}/Ensemble_Model_External_Validation_{this}.pdf')
    
        # Perform DeLong's test to compare the best and second-best models (External Validation Set)
        if len(validation_set_results2) >= 2:
            validation_set_results2.sort(key = lambda x: x[eindex], reverse = ereverse)
            model1_val = validation_set_results2[0]
            model2_val = validation_set_results2[1]
            
            logging.info("DeLong's Test for AUC Comparison (External Validation Set)")
            logging.info(f"Comparing '{model1_val['name']}' (AUC={model1_val['auc']:.4f}, {eindex}={model1_val[eindex]:.4f}) vs. '{model2_val['name']}' (AUC={model2_val['auc']:.4f}, {eindex}={model2_val[eindex]:.4f})")
            
            z_score_val, p_value_val = delong_test(y_val, model1_val['scores'], model2_val['scores'])
            
            logging.info(f'Z-score: {z_score_val:.4f}, P-value: {p_value_val:.4f}')
            if p_value_val < 0.05:
                logging.info(f"Conclusion: The difference in AUC is statistically significant. '{model1_val['name']}' is superior.")
            else:
                logging.info('Conclusion: The difference in AUC is not statistically significant.')

        # Calculate and display performance metrics for each model on the external validation set
        logging.info('Calculating external validation set performance metrics ...')
        metrics_dict_val = {'Metrics': ['accuracy', 'sensitivity', 'specificity', 'PPV', 'NPV', 'F1 score', 'Kappa', 'Brier']}
        for model_name, info in modelsfinal.items():
            y_pred_proba_val = info['model'].predict_proba(X_val_final)[:, 1]
            metrics_dict_val[model_name] = calculate_metrics(y_val, y_pred_proba_val)

        metrics_df_val = pd.DataFrame(metrics_dict_val)
        logging.info('External Validation Set Performance Metrics\n\n' + metrics_df_val.round(4).to_string() + '\n')
        
        metrics_df_val.to_excel(f'{tldir}/Ensemble_Model_External_Validation_Metrics_{this}.xlsx', index = False)
        logging.info(f'{tldir}/Ensemble_Model_External_Validation_Metrics_{this}.xlsx')
    

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 9: SHAP Model Interpretation for the Final Ensemble Model
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if this in ['model12', 'model22'] and shapsta:
        
        logging.info('Step 9: Performing SHAP model interpretation for the final Soft Voting ensemble model ...')
        
        # Limit the number of samples used for SHAP calculation.
        # SHAP calculations can be extremely slow, especially for KernelExplainer.
        # For visualization, 100-300 samples are usually sufficient to show feature importance and distribution.
        # Running KernelExplainer on thousands of rows may cause the process to hang or crash.
        N_SHAP_SAMPLES = 250 
        
        if len(X_test_final) > N_SHAP_SAMPLES:
            logging.info(f'Test set is too large ({len(X_test_final)} rows). Randomly sampling {N_SHAP_SAMPLES} rows for SHAP analysis to prevent out-of-memory errors ...')
            # Use a fixed random_state to ensure consistent sampling across runs
            X_test_shap = shap.sample(X_test_final, N_SHAP_SAMPLES, random_state = seed)
        else:
            X_test_shap = X_test_final.copy()
        
        # Define a robust function to extract SHAP values for Class 1
        def get_shap_values_for_class1(explainer, data):
            """ Calculates SHAP values and robustly extracts the values for the positive class (class 1). """
            logging.getLogger('shap').setLevel(logging.WARNING)
            shap_values = explainer.shap_values(data)
            
            # SHAP can return a list of arrays (one per class) or a 3D array (samples, features, classes)
            if isinstance(shap_values, list):
                # It's a list, take the second element (for class 1)
                return shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # It's a 3D array, slice along the class dimension to get class 1
                return shap_values[:, :, 1]
            else:
                # It's already a 2D array (e.g., regression) or something else we don't expect for classification
                # For this context, we assume it's the correct 2D array for the positive class
                return shap_values
    
        # Define a helper function to create the appropriate SHAP explainer
        def create_explainer(model, data, model_name):
            """ Creates a SHAP explainer appropriate for the model type. """
            
            # Try to use TreeExplainer (fast)
            try:
                logging.info(f"Attempting to create shap.TreeExplainer for '{model_name}' ...")
                explainer = shap.TreeExplainer(model) 
                
                # Perform a quick test to check if it works
                _ = explainer.shap_values(data.iloc[:2, :], check_additivity = False)
                logging.info(f"Successfully initialized shap.TreeExplainer for '{model_name}'")
                return explainer
                
            except Exception as e:
                logging.warning(f"'{model_name}' cannot use TreeExplainer (Reason: {str(e)}). Falling back to shap.KernelExplainer.")
                logging.warning('KernelExplainer computation is extremely slow, please be patient ...')
                
                # Using 50 kmeans centroids as background is more representative and faster than random sampling.
                background_summary = shap.kmeans(data, 50) 
                
                # For KernelExplainer, we need the probability prediction function
                predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
                
                return shap.KernelExplainer(predict_fn, background_summary)

    
        # 1. Create explainers for the three base models respectively
        # Using X_train_final as background data, which is the standard practice in SHAP
        top1_model = refined_estimators[0][1]; top1_model_name = refined_estimators[0][0]
        top2_model = refined_estimators[1][1]; top2_model_name = refined_estimators[1][0]
        top3_model = refined_estimators[2][1]; top3_model_name = refined_estimators[2][0]
        
        explainer1 = create_explainer(top1_model, X_train_final, top1_model_name)
        explainer2 = create_explainer(top2_model, X_train_final, top2_model_name)
        explainer3 = create_explainer(top3_model, X_train_final, top3_model_name)
        
        # 2. Calculate SHAP values on the test set for each base model
        # For TreeExplainer, shap_values returns a list of arrays (one for each class)
        # We are interested in the SHAP values for the "positive" class (class 1)
        logging.info(f'Calculating SHAP values for {top1_model_name} on the sampled test set ({len(X_test_shap)} rows) ...')
        shap_values1 = get_shap_values_for_class1(explainer1, X_test_shap)
        
        logging.info(f'Calculating SHAP values for {top2_model_name} on the sampled test set ...')
        shap_values2 = get_shap_values_for_class1(explainer2, X_test_shap)
        
        logging.info(f'Calculating SHAP values for {top3_model_name} on the sampled test set ...')
        shap_values3 = get_shap_values_for_class1(explainer3, X_test_shap)
        
        # 3. Compute the weighted average of SHAP values using the optimal weights
        logging.info('Combining SHAP values using the optimal weights ...')
        w1, w2, w3 = weights
        
        # Ensure consistent dimensions
        s1 = shap_values1.values if hasattr(shap_values1, 'values') else shap_values1
        s2 = shap_values2.values if hasattr(shap_values2, 'values') else shap_values2
        s3 = shap_values3.values if hasattr(shap_values3, 'values') else shap_values3
        
        shap_values_ensemble = (w1 * s1) + (w2 * s2) + (w3 * s3)
    
        # Calculate the mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values_ensemble).mean(axis = 0)
        
        # Create a DataFrame to present the results clearly
        feature_names = X_test_final.columns
        shap_summary_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        })
        
        # Sort features by importance
        shap_summary_df = shap_summary_df.sort_values(by = 'mean_abs_shap', ascending = False)
        
        # Print results
        logging.info(f'Mean absolute SHAP values for each feature:\n\n{shap_summary_df.to_string()}\n')
    
        # 4. Visualize using the combined SHAP values
        logging.info('Generating SHAP summary plot for the final ensemble model ...')
        
        # SHAP Summary Plot (Beeswarm)
        plt.figure(dpi = dpi)
        shap.summary_plot(shap_values_ensemble, X_test_shap, plot_type = 'dot', show = False)
        fig = plt.gcf()
        fig.set_size_inches(4, 4) 
        if False: plt.title('SHAP Summary for Ensemble Model', fontsize = 12)
        plt.savefig(f'{fgdir5}/SHAP_Summary_Ensemble_{this}.png', bbox_inches = 'tight')
        plt.savefig(f'{fgdir5}/SHAP_Summary_Ensemble_{this}.pdf', bbox_inches = 'tight')
        plt.close()
        
        logging.info(f'{fgdir5}/SHAP_Summary_Ensemble_{this}.png')
        logging.info(f'{fgdir5}/SHAP_Summary_Ensemble_{this}.pdf')
        
        # SHAP Feature Importance Plot (Bar)
        plt.figure(dpi = dpi)
        shap.summary_plot(shap_values_ensemble, X_test_final, plot_type = 'bar', show = False)
        fig = plt.gcf()
        fig.set_size_inches(6, 4) 
        if False: plt.title('Feature Importance for Ensemble Model', fontsize = 12)
        plt.savefig(f'{fgdir5}/SHAP_Bar_Ensemble_{this}.png', bbox_inches = 'tight')
        plt.savefig(f'{fgdir5}/SHAP_Bar_Ensemble_{this}.pdf', bbox_inches = 'tight')
        plt.close()
        
        logging.info(f'{fgdir5}/SHAP_Bar_Ensemble_{this}.png')
        logging.info(f'{fgdir5}/SHAP_Bar_Ensemble_{this}.pdf')
        
        # SHAP Dependence Plots for top features
        # We can find the top features by the mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values_ensemble).mean(axis = 0)
        top_feature_indices = np.argsort(mean_abs_shap)[::-1]
        
        num_dep_plots = min(10, len(final_features))
        logging.info(f'Generating SHAP dependence plots for the top {num_dep_plots} features ...')
        
        for i in range(num_dep_plots):
            
            feature_index = top_feature_indices[i]
            feature_name = final_features[feature_index]
    
            feature_name_ = re.sub('/', '_', feature_name)
            feature_name_ = re.sub('\*', '_', feature_name_)
            
            plt.figure(dpi = dpi)
            shap.dependence_plot(feature_name, shap_values_ensemble, X_test_final, interaction_index = 'auto', show = False)
            plt.title(f'SHAP Dependence Plot for {feature_name}', fontsize = 14)
            plt.savefig(f'{fgdir5}/SHAP_Dependence_{feature_name_}_{this}.png', bbox_inches = 'tight')
            plt.savefig(f'{fgdir5}/SHAP_Dependence_{feature_name_}_{this}.pdf', bbox_inches = 'tight')
            plt.close()
            
            logging.info(f'{fgdir5}/SHAP_Dependence_{feature_name_}_{this}.png')
            logging.info(f'{fgdir5}/SHAP_Dependence_{feature_name_}_{this}.pdf')


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 10: Decision Curve Analysis (DCA)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Step 10: Performing Decision Curve Analysis (DCA) ...')

    # DCA for Test Set
    logging.info('Calculating and plotting DCA curves for the test set ...')
    run_dca_analysis(refined_voting_clf, X_test_final, y_test, prefix = f'Ensemble_Model_Test_{this}')
    
    # DCA for External Validation Set
    run_dca_analysis(refined_voting_clf, X_val_final, y_val, prefix = f'Ensemble_Model_External_Validation_{this}')
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 11: Model Calibration Assessment and Brier Score
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Select the model for calibration assessment (the best ensemble model)
    models_to_calibrate = {}

    # Include the Soft Voting ensemble model
    models_to_calibrate[finalmodel] = modelsfinal[finalmodel]

    if models_to_calibrate:
        
        plt.figure(figsize = (4, 5), dpi = 300)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        
        ax1.plot([0, 1], [0, 1], 'k:', label = 'Perfectly calibrated')
        
        for name, info in models_to_calibrate.items():
            
            # Check if the model supports predict_proba
            if not hasattr(info['model'], 'predict_proba'):
                logging.warning(f"Model '{name}' does not support predict_proba; skipping calibration curve plotting.")
                continue

            y_pred_proba = info['model'].predict_proba(X_test_final)[:, 1]
            brier = brier_score_loss(y_test, y_pred_proba)
            fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins = 10, strategy = 'uniform')
            
            # Retrieve the color stored in info (safely default to None if not present)
            color = info.get('color', None)
            ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label = f'{name} (Brier: {brier:.4f})', color = color)
            ax2.hist(y_pred_proba, range = (0, 1), bins = 10, label = name, histtype = 'step', lw = 2, color = color)
            logging.info(f'{this}, {name}, Test Cohorts, (Brier: {brier:.4f})')

        ax1.set_ylabel('Fraction of positives', fontsize = 10)
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc = 'lower right', fontsize = 10)
        ax1.set_title('Calibration plots (Reliability curve)', fontsize = 10, fontweight = 'bold')

        ax2.set_xlabel('Mean predicted value', fontsize = 10)
        ax2.set_ylabel('Count', fontsize = 10)
        ax2.legend(loc = 'upper center', ncol = len(models_to_calibrate), fontsize = 10)

        plt.tight_layout()
        plt.savefig(f'{fgdir}/Calibration_Curve_Test_{this}.pdf', bbox_inches = 'tight')
        plt.savefig(f'{fgdir}/Calibration_Curve_Test_{this}.png', bbox_inches = 'tight')
        plt.close()
        
        logging.info(f'{fgdir}/Calibration_Curve_Test_{this}.pdf')
        logging.info(f'{fgdir}/Calibration_Curve_Test_{this}.png')
    else:
        logging.info('No models available for calibration assessment.')
        
    if models_to_calibrate:
        
        plt.figure(figsize = (4, 5), dpi = 300)
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        
        ax1.plot([0, 1], [0, 1], 'k:', label = 'Perfectly calibrated')
        
        for name, info in models_to_calibrate.items():
            
            # Check if the model supports predict_proba
            if not hasattr(info['model'], 'predict_proba'):
                logging.warning(f"Model '{name}' does not support predict_proba; skipping calibration curve plotting.")
                continue

            y_pred_proba = info['model'].predict_proba(X_val_final)[:, 1]
            brier = brier_score_loss(y_val, y_pred_proba)
            fraction_of_positives, mean_predicted_value = calibration_curve(y_val, y_pred_proba, n_bins = 10, strategy = 'uniform')
            
            # Retrieve the color stored in info (safely default to None if not present)
            color = info.get('color', None)
            ax1.plot(mean_predicted_value, fraction_of_positives, 's-', label = f'{name} (Brier: {brier:.4f})', color = color)
            ax2.hist(y_pred_proba, range = (0, 1), bins = 10, label = name, histtype = 'step', lw = 2, color = color)
            logging.info(f'{this}, {name}, Validation Cohorts, (Brier: {brier:.4f})')

        ax1.set_ylabel('Fraction of positives', fontsize = 10)
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc = 'lower right', fontsize = 10)
        ax1.set_title('Calibration plots (Reliability curve)', fontsize = 10, fontweight = 'bold')

        ax2.set_xlabel('Mean predicted value', fontsize = 10)
        ax2.set_ylabel('Count', fontsize = 14)
        ax2.legend(loc = 'upper center', ncol = len(models_to_calibrate), fontsize = 10)

        plt.tight_layout()
        plt.savefig(f'{fgdir}/Calibration_Curve_Validation_{this}.pdf', bbox_inches = 'tight')
        plt.savefig(f'{fgdir}/Calibration_Curve_Validation_{this}.png', bbox_inches = 'tight')
        plt.close()
        
        logging.info(f'{fgdir}/Calibration_Curve_Validation_{this}.pdf')
        logging.info(f'{fgdir}/Calibration_Curve_Validation_{this}.png')
    else:
        logging.info('No models available for calibration assessment.')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Step 12: Save the Best Model, Preprocessor, and Feature List
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logging.info('Step 12: Saving the best model, its preprocessor, and the feature list ...')
    
    # Save the voting classifier package
    model_package = {
        'model': refined_voting_clf,
        'preprocessor': final_preprocessor,
        'features': final_features,
        'is_voting_classifier': True
    }

    model_filename = f'voting_model_{this}.joblib'
    model_save_path = os.path.join(mddir, model_filename)
    joblib.dump(model_package, model_save_path)
        
    logging.info("The model, along with its preprocessor and feature list, has been successfully packaged and saved to:")
    logging.info(f'--> {model_save_path}')
    logging.info('For future inference, please load this file first. Use the saved preprocessor to transform new data, and then use the model for prediction.')
    logging.info('Ensure that the column names of the incoming new data match the saved features list exactly.')

    logging.info('Script finished successfully.')
    
