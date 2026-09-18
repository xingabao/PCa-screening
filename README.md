# PCa-screening

This repository includes Python and R code to reproduce all of the analyses for the paper "**Blood-based integration of circulating AR⁺TREM2⁺ monocytes and routine biomarkers refines prostate cancer risk stratification**", to generate the main and supplementary figures of the paper, and to run the interactive web-based decision-support prototype described in the manuscript ([https://pca-screening.streamlit.app](https://pca-screening.streamlit.app)).

# Graphical Abstract

The graphical abstract is provided as a high-resolution PDF: [Figures/Graphical Abstract Image.pdf](./Figures/Graphical%20Abstract%20Image.pdf)

<img src='Figures\Graphical Abstract Image.png'>

# Overview

Accurate prebiopsy risk stratification remains challenging in prostate cancer (PCa), because PSA-based assessment has limited specificity for distinguishing benign prostatic hyperplasia (BPH) from malignancy and may lead to unnecessary biopsies. Building on a previously characterized APOE–TREM2–AR myeloid program, we evaluated whether circulating AR⁺TREM2⁺ monocytes could improve blood-based prostate cancer risk stratification when integrated with routine blood biomarkers in a two-stage machine-learning framework. This multicenter development and external validation study enrolled 669 participants from four tertiary centers in China, including a Guangdong development cohort (n = 518) and an independent Anhui validation cohort (n = 151). The comprehensive hierarchical model improved three-class accuracy from 0.694 to 0.818 and macro-average AUC from 0.869 to 0.943 compared with PSA-based models. At matched sensitivity, retrospective threshold simulation projected a 72.4% reduction in unnecessary biopsy recommendations compared with PSA-based assessment. These findings support a mechanism-informed immune profiling strategy for precision prebiopsy prostate cancer risk stratification beyond PSA-based evaluation.

The analysis pipeline comprises five phases: (i) study design and data collection; (ii) data preprocessing and feature engineering; (iii) two-stage hierarchical model development; (iv) performance evaluation, reclassification and interpretability analysis; and (v) clinical application deployment as a Streamlit web application.

**Keywords**: Prostate cancer; Mechanism-informed machine learning; Liquid biopsy; AR⁺TREM2⁺ monocytes; APOE–TREM2–AR axis; Pre-biopsy risk stratification

# Repo Contents

- [PCa-screening.py](./PCa-screening.py): source code of the Streamlit web application (`https://pca-screening.streamlit.app`). It implements the two-stage hierarchical ensemble pipeline, automatically calculates the derived features (fPSA/tPSA ratio, `TPSA*AR+TREM2+`, `APOE*AR+TREM2+`, `AFP*AR+TREM2+`), and returns stage-wise and global probabilities with an adjustable PCa decision threshold (default 0.50).
- [models](./models): the two serialized model bundles used by the web application, each containing the fitted soft-voting ensemble, its preprocessing pipeline (log(x+1) transformation + z-score standardization) and the exact input feature list.

  + `disease_screening.joblib`. Stage 1 model (healthy control vs. prostatic disease). Soft-voting ensemble of LDA, logistic regression and SVM; 13 features (`TPSA`, `LY%`, `HCT`, `RDW-CV`, `FPSA/TPSA`, `FPSA`, `Urea`, `HGB`, `LY#`, `TPSA*AR+TREM2+`, `NEUT#`, `MONO#`, `PLT`). Corresponds to `model12` in [Scripts/build_model.py](./Scripts/build_model.py).
  + `malignancy_differentiation.joblib`. Stage 2 model (BPH vs. PCa). Soft-voting ensemble of ANN, XGBoost and KNN; 8 features (`TPSA`, `FPSA/TPSA`, `NEUT#`, `MCHC`, `APOE*AR+TREM2+`, `MCH`, `AFP*AR+TREM2+`, `age`). Corresponds to `model22` in [Scripts/build_model.py](./Scripts/build_model.py).

- [Scripts](./Scripts): Python and R scripts for data quality control, statistical modeling, model evaluation and figure generation.

  + `build_model.py`. Core machine-learning pipeline. Performs log(x+1)/z-score preprocessing, LightGBM-based importance ranking, recursive feature elimination (RFE) with 10-fold cross-validation, Bayesian hyperparameter optimization of 15 candidate algorithms with Optuna (50 trials per algorithm), selection of the three best algorithms by F1 score in the internal test set, construction and evaluation of the equally weighted soft-voting ensemble, DeLong's test, decision curve analysis (DCA), learning curve analysis, SHAP interpretation, and export of the model bundle (`voting_model_<model>.joblib`). The manuscript reports four modelling streams, which are selected by the `this` variable at the top of the script: `model11` (Stage 1, PSA-derived baseline), `model12` (Stage 1, comprehensive), `model21` (Stage 2, PSA-derived baseline) and `model22` (Stage 2, comprehensive). The script must be run once per modelling stream. **The approximate runtime is several hours per run on a CPU** (15 algorithms × 50 Optuna trials × 10-fold cross-validation, plus SHAP and learning-curve retraining).
  + `hierarchical_model_evaluation.py`. Rebuilds the complete two-stage hierarchical classifier from the saved Stage 1 and Stage 2 bundles and evaluates it on the pooled validation cohort (internal test set + external validation cohort). Reports overall three-class accuracy, multiclass Brier score, confusion matrix and multiclass ROC/AUC curves. Set `this = 'model1'` for the PSA-derived baseline system or `this = 'model2'` for the comprehensive system.
  + `incremental_value_analysis.py`. Computes the incremental value of the comprehensive Stage 2 model over the PSA-derived baseline: category-free NRI, categorical NRI (cut-offs 0.10 and 0.30), IDI and the associated reclassification tables, with 1000 bootstrap replicates for 95% confidence intervals and two-sided P values.
  + `missing_data_date.py`. Computes and visualizes the per-feature missingness rate (frequency, percentage, cumulative percentage) of the raw dataset, with color coding according to the pre-specified missingness thresholds (exclusion at >50% missingness). Generates `Outputs/Fig_Missing_Data_Rate`.
  + `data_imputation_diagnostic.py`. Compares the density distributions of the original and imputed data for the 24 features with low-level missingness, separately for the development cohort (`tt`) and the external validation cohort (`vn`), documenting that missForest imputation preserved the original feature distributions. Generates `Outputs/Fig_Data_Imputation_Diagnostic`.
  + `Table1.R`. Generates Table 1 (baseline demographic and clinical characteristics of the development and external validation cohorts) using gtsummary and gt.
  + `feature_boxplot.R`. Generates the feature-level distribution figures: the main figure comparing the three diagnostic groups for the PSA-derived, Stage 1 (13-feature) and Stage 2 (8-feature) biomarkers, including the Venn diagram of the feature sets, and the supplementary figures for the remaining routine laboratory variables. Produces `MAIN-1-MODEL`, `SUP-1-MODEL`, `SUP-2-MODEL` and `SUP-3-MODEL` in `Outputs/Fig_boxplot`.
  + `Correlation.R`. Tests the relationship between plasma APOE and circulating AR⁺TREM2⁺ monocytes (Spearman's rank correlation, patients with BPH or PCa and tPSA >10 ng/mL in the development cohort, log-transformed APOE) and generates the scatter plot with fitted line and 95% confidence band. Generates `Outputs/Fig_correlation`.

- [data](./data): de-identified datasets required to reproduce the analysis (`.xlsx`).

  | File | n | Content |
  | --- | --- | --- |
  | `dat.final.xlsx` | 669 | Full study population before imputation (77 columns), including the cohort identifier (`Group`: `GZ` = Guangdong development cohort, `AH` = Anhui external validation cohort), the diagnosis label (`diagnose`: 0 = healthy control, 1 = BPH, 2 = PCa) and the raw clinical variables with their original missing values. |
  | `train-test-data.xlsx` | 518 | Development cohort after quality control and missForest imputation (66 columns), used for feature selection, model training, hyperparameter tuning and internal testing. |
  | `test-data.xlsx` | 156 | Internal hold-out test set of the development cohort (30% stratified split). |
  | `validation-data.xlsx` | 151 | Independent external validation cohort from Anhui Provincial Hospital, kept separate from all model development procedures. |

- [Figures](./Figures): high-resolution publication-ready figures in PDF/TIF format, including `Fig_1`–`Fig_7`, the graphical abstract, and `Supplementary Figure/Supplementary Fig-1`–`16`.
- [Outputs](./Outputs): analysis outputs generated by the scripts, organized by analysis (`Fig_ROC_AUC`, `Fig_DCA`, `Fig_SHAP`, `Fig_all` (calibration curves), `Fig_Feature_Ranking`, `Fig_Feature_Selection`, `Fig_boxplot`, `Fig_Hierarchical_Model`, `Fig_NRI_IDI`, `Fig_Missing_Data_Rate`, `Fig_Data_Imputation_Diagnostic`, `Fig_correlation`), with numerical results and summary tables in `Tbl_all` and run logs in `Tbl_temp`.
- [requirements.txt](./requirements.txt): pinned Python environment for the web application (and for the versions of the core machine-learning libraries used in this study).

# System Requirements

The analysis was performed on a Windows 11 personal computer (16 GB RAM; Intel® Core™ i7-7700K processor). The statistical and figure scripts were run in R (version 4.6.0) using the RStudio IDE with R packages `gtsummary` (v2.5.0) and `gt` (v1.3.0); the machine-learning pipeline was implemented in Python (version 3.13.5) using `scikit-learn` (v1.6.1), `lightgbm` (v4.6.0), `xgboost` (v3.0.5), `tabpfn` (v6.0.5), `optuna` (v4.5.0) and `shap` (v0.50.0). The full R session information recorded at runtime is provided in `Outputs/Fig_correlation/sessionInfo.txt`. No non-standard hardware is required. All scripts except `build_model.py` complete within a few minutes; `build_model.py` requires several hours per modelling stream, dominated by hyperparameter optimization, ensemble retraining and SHAP analysis.

# Installation

## Step 1: Clone the repository

```shell
git clone https://github.com/xingabao/PCa-screening.git
```

## Step 2: Install Python dependencies

The pinned Python environment used for this study is provided in `requirements.txt`:

```shell
cd PCa-screening
pip install -r requirements.txt
```

This environment is sufficient to run the web application. To re-run the full model-development pipeline (`Scripts/build_model.py`, `Scripts/hierarchical_model_evaluation.py`, `Scripts/incremental_value_analysis.py`), the following additional packages are required:

```shell
pip install openpyxl optuna shap
```

In addition, the Stage 1 and Stage 2 comprehensive models are trained within a pool of 15 candidate algorithms, which requires the full set of algorithm libraries already listed in `requirements.txt` (`lightgbm`, `xgboost`, `catboost`, `tabpfn`). `tabpfn` additionally requires a pretrained checkpoint file, whose path is specified by the `ckpt` entry of the configuration file. If TabPFN is not required, the corresponding entry can be removed from the candidate algorithm dictionary in `Scripts/build_model.py`.

The installation time should be < 10 minutes on a typical desktop computer.

## Step 3: Install R dependencies (for Table 1 and the figure scripts)

```R
install.packages("glue")            # 1.8.1
install.packages("dplyr")           # 1.2.1
install.packages("ggplot2")         # 4.0.3
install.packages("patchwork")       # 1.3.2
install.packages("ggbreak")         # 0.1.7
install.packages("ggsignif")        # 0.6.4
install.packages("ggVennDiagram")   # 1.5.7
install.packages("readxl")          # 1.4.5
install.packages("rstudioapi")      # 0.18.0
install.packages("gt")              # 1.3.0
install.packages("gtsummary")       # 2.5.0
```

The versions above correspond to those recorded in the `sessionInfo()` output shipped with the repository.

Low-level missing values (<5%) in the raw dataset were imputed with the missForest algorithm (`missForest` R package) separately in the development and external validation cohorts before modelling. The imputed datasets are distributed as `data/train-test-data.xlsx` and `data/validation-data.xlsx`, so the imputation step does not need to be re-run to reproduce the reported results.

# Configuration

All scripts read their runtime settings from `.config.txt` located in the `Scripts` directory:

```ini
[ENV]
rtdir = <project root>          # repository root directory
otdir = <project root>          # directory in which Outputs/ is written
ckpt  = <path to the TabPFN pretrained checkpoint>
dtdir = data                    # directory containing the input datasets
seed = 42                       # random seed used throughout the pipeline
kfold = 10                      # number of cross-validation folds
topim = 30                      # number of features shown in the importance ranking
ntrials = 50                    # number of Optuna trials per candidate algorithm
njobs = -1                      # number of parallel jobs (-1 = all available cores)
alpha = 0.3                     # legacy entry, not used by the current scripts
nbootstraps = 1000              # number of bootstrap replicates for the 95% CIs
device = cpu                    # device used by TabPFN
eindex = F1 score               # metric used to rank candidate algorithms
nmodels = 3                     # number of base models in the soft-voting ensemble
```

Before running the scripts, update `rtdir`, `otdir` and `ckpt` to paths that exist on your machine, and point the scripts to the configuration file of your local copy of the repository. `Scripts/build_model.py` reads the configuration file from the project root, whereas `Scripts/hierarchical_model_evaluation.py`, `Scripts/data_imputation_diagnostic.py` and `Scripts/missing_data_date.py` read it from the `Scripts` directory; `Scripts/incremental_value_analysis.py` resolves the project root through the `base_dir` variable defined at the top of the script. Only these path entries need to be adjusted, and no other change to the code is required.

The R scripts (`Table1.R`, `feature_boxplot.R`, `Correlation.R`) determine their working directory with `rstudioapi::getActiveDocumentContext()$path` and read the datasets through the relative path `../data/`; they should therefore be run from within RStudio (or the working directory should be set manually).

Note that `Scripts/missing_data_date.py` imports the custom helper module `clearn` (`from clearn.data_preprocessing import compute_missing`) for missingness computation. The same missingness statistics are also calculated in plain pandas within the script itself.

# Running the Analyses

Model development is grouped into four modelling streams, which are run by setting the `this` variable at the top of `Scripts/build_model.py`:

| Stream | Comparison | Candidate features |
| --- | --- | --- |
| `model11` | Healthy vs. disease (Stage 1) | fPSA, tPSA, fPSA/tPSA |
| `model12` | Healthy vs. disease (Stage 1) | Full 67-feature candidate set (age excluded) |
| `model21` | BPH vs. PCa (Stage 2) | fPSA, tPSA, fPSA/tPSA |
| `model22` | BPH vs. PCa (Stage 2) | Full 67-feature candidate set |

Each run writes its figures, tables, logs and model bundle (`Outputs/models/voting_model_<stream>.joblib`) into `Outputs`. The recommended order of execution is:

**1. Data quality control and descriptive analyses**

```shell
python Scripts/missing_data_date.py          # missingness profile of the raw dataset
python Scripts/data_imputation_diagnostic.py # original vs. imputed distributions
```

Run `Scripts/Table1.R` in RStudio to generate Table 1, and `Scripts/Correlation.R` and `Scripts/feature_boxplot.R` to generate the feature-level figures.

**2. Model development**

Run `Scripts/build_model.py` once for each of the four modelling streams (`this = 'model11'`, `'model12'`, `'model21'`, `'model22'`). Each run produces the feature ranking and RFE curves, the ROC/AUC curves of the individual algorithms and of the ensemble in the test and external validation cohorts, the DCA curves, the calibration curves, the SHAP summary/dependence plots, the performance tables, the learning-curve data and the serialized ensemble bundle.

**3. Hierarchical system and incremental value**

```shell
python Scripts/hierarchical_model_evaluation.py   # set this = 'model1' or 'model2'
python Scripts/incremental_value_analysis.py      # NRI and IDI
```

Both scripts load the bundles written to `Outputs/models/`, so the four modelling streams of step 2 must be completed first. The two bundles used by the web application (`models/disease_screening.joblib` and `models/malignancy_differentiation.joblib`) are the `model12` and `model22` bundles, respectively, as exported for deployment.

# Web Application and Streamlit Deployment

`PCa-screening.py` implements the decision-support prototype described in the manuscript. It reconstructs the hierarchical classifier from the two bundles in `models/`, automatically calculates the fPSA/tPSA ratio and the mechanism-informed interaction terms, and reports the Stage 1 disease probability, the Stage 2 malignancy probability, the global probability distribution over healthy/BPH/PCa, and a clinical interpretation. The PCa decision threshold is adjustable in the sidebar (default 0.50) to explore different sensitivity–specificity trade-offs.

## Run the application locally

```shell
pip install -r requirements.txt
streamlit run PCa-screening.py
```

The application is then available at `http://localhost:8501`. No dataset, external path or TabPFN checkpoint is needed: all predictions are produced from `models/disease_screening.joblib` and `models/malignancy_differentiation.joblib`, which are loaded relative to the repository root with `@st.cache_resource`.

## Deploy the application on Streamlit Community Cloud

The public prototype was deployed from this repository with Streamlit Community Cloud:

1. Push the repository to GitHub (`https://github.com/xingabao/PCa-screening`), ensuring that the `models/` directory (both `.joblib` files) and `requirements.txt` are committed.
2. Sign in at [https://share.streamlit.io](https://share.streamlit.io) with the GitHub account that owns the repository and select **Create app** → **Deploy a public app from GitHub**.
3. Set **Repository** to `xingabao/PCa-screening`, **Branch** to `main`, and **Main file path** to `PCa-screening.py`.
4. In **Advanced settings**, select the Python version corresponding to the environment used in this study (Python 3.13; Python 3.11 also works). Dependencies are installed automatically from `requirements.txt`; no secrets or environment variables are required.
5. Select **Deploy**. Startup takes a few minutes on the first build, after which the application is available at a public URL (for the published prototype, [https://pca-screening.streamlit.app](https://pca-screening.streamlit.app)); the subdomain can be customized in the app settings.

Because the deployed application only loads the two serialized bundles, its runtime only requires `streamlit`, `scikit-learn`, `xgboost`, `joblib`, `numpy`, `pandas` and `scipy`. The remaining pinned packages (`lightgbm`, `catboost`, `tabpfn`, `seaborn`, `matplotlib`) are needed to re-run the model-development pipeline; if a lighter cloud build is preferred, they can be moved to a separate requirements file used only for the analysis environment.

For reproducibility, the versions of the core machine-learning libraries used in this study are pinned in `requirements.txt` and reported in the manuscript (Statistical analysis). The complete package versions of the model-development environment can be regenerated with `pip freeze` after installation.

# Data Availability

Use of the shared data requires citation of this publication. The data of this study are accessible at the GitHub repository: [https://github.com/xingabao/PCa-screening](https://github.com/xingabao/PCa-screening). A web-based tool for clinical use is available at [https://pca-screening.streamlit.app](https://pca-screening.streamlit.app). Requests for further information, resources and reagents should be directed to and will be fulfilled by the lead contact, Guanmin Jiang (jianggm3@mail.sysu.edu.cn).

All datasets distributed in this repository are de-identified. Original clinical records are protected and are not publicly available owing to patient privacy regulations.

# License

This project is covered by the MIT license.
