# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 12:53:45 2025
Updated on Sat Jan 31 2026
 @author Abao Xing
 @email  albertxn7@gmail.com
 This scripts writen by Abao Xing

   ┏┓　　┏┓
  ┏┛┻━━━━┛┻┓
  ┃　　　　  ┃
  ┃　━　　━　 ┃
  ┃　┳┛　┗┳　 ┃
  ┃　　　　　 ┃
  ┃　　　┻　　┃
  ┃　　　　　 ┃
  ┗━━┓　　　┏━┛
  　　┃　　 ┃ 神兽保佑
  　　┃　　 ┃ 代码无BUG！！！
  　　┃　　 ┗━━━━━┓
  　　┃　　　　　　  ┣┓
 　　┃　　　　　　  ┏┛┃
 　　┗┓┓┏━━━━━┳┓┏━━┛
  　　┃┫┫　   ┃┫┫
  　　┗┻┛　   ┗┻┛

"""

import os
import joblib
import warnings
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 0. Suppress Warnings
# --------------------------------------------------
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# --------------------------------------------------
# 1. Page Configuration
# --------------------------------------------------
st.set_page_config(page_title = 'Prostate Cancer Advanced Diagnostic Application', page_icon = '⚕️', layout = 'wide')

# Inject Global CSS
st.markdown("""
<style>
    /* Adjust top margin */
    .stApp { background-color: #FFFFFF; margin-top: -80px; }
    
    /* Section Headers */
    .section-header {
        color: #2c3e50;
        font-size: 1.1rem;
        font-weight: bold;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    /* Input labels */
    .stNumberInput label { font-size: 0.85rem; }
    
    /* Result card style */
    .result-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #eee;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .card-header { font-weight: bold; font-size: 1.1rem; margin-bottom: 5px; }
    .card-sub { font-size: 0.9rem; color: #666; }
    
    /* Interpretation Box */
    .interpret-box {
        background-color: #f1f3f5;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #495057;
        margin-top: 15px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Sidebar buttons */
    div[data-testid="stSidebar"] .stButton button {
        width: 100%; border-radius: 8px; height: 3em; font-weight: bold;
    }

    /* FORCE DIALOG WIDTH TO ~1000px */
    div[data-testid="stDialog"] div[role="dialog"] {
        width: 1000px !important;
        max-width: 90vw !important; 
    }
</style>
""", unsafe_allow_html = True)

# --------------------------------------------------
# 2. Core Logic Class
# --------------------------------------------------
class HierarchicalClassifier:
    
    def __init__(self, model_step1, model_step2, preprocessor1, preprocessor2, pca_threshold = 0.5):
        self.preprocessor1 = preprocessor1
        self.preprocessor2 = preprocessor2
        self.model_step1 = model_step1
        self.model_step2 = model_step2
        self.pca_threshold = pca_threshold
        
        self.features1 = getattr(model_step1, 'feature_names_in_', None)
        self.features2 = getattr(model_step2, 'feature_names_in_', None)
        
    def predict_full_detail(self, X):
        # Step 1: Healthy vs Disease
        cols1 = self.features1 if self.features1 is not None else X.columns
        X_step1 = X.copy()
        # Ensure all columns exist
        for col in cols1:
            if col not in X_step1.columns: X_step1[col] = 0
                
        X1 = self.preprocessor1.transform(X_step1[cols1])
        prob1 = self.model_step1.predict_proba(X1)[0]
        
        p_healthy = prob1[0]
        p_disease_total = prob1[1]
        
        is_disease = p_disease_total > 0.5 
        
        # Step 2: BPH vs PCa
        cols2 = self.features2 if self.features2 is not None else X.columns
        X_step2 = X.copy()
        for col in cols2:
            if col not in X_step2.columns: X_step2[col] = 0
        
        X2 = self.preprocessor2.transform(X_step2[cols2])
        prob2 = self.model_step2.predict_proba(X2)[0]
        
        p_bph_cond = prob2[0] # P(BPH | Disease)
        p_pca_cond = prob2[1] # P(PCa | Disease)
        
        # Confidence level for Step 2 (between 0 and 1)
        step2_confidence = abs(p_bph_cond - p_pca_cond)
        
        # Bayesian-like adjustment: 
        # If Step 1 says Disease, and Step 2 is very confident, we penalize the Healthy probability further.
        if is_disease:
            penalty_factor = 1.0 - step2_confidence
            p_healthy = p_healthy * penalty_factor
            p_disease_total = 1.0 - p_healthy

        # Global probability calculation
        global_healthy = p_healthy
        global_bph = p_disease_total * p_bph_cond
        global_pca = p_disease_total * p_pca_cond
        
        # Final normalization
        total = global_healthy + global_bph + global_pca
        
        probs_dict = {
            'Healthy': global_healthy / total,
            'BPH': global_bph / total,
            'PCa': global_pca / total
        }
        
        # Determine the final result.
        is_pca_final = p_pca_cond > self.pca_threshold
        final_code = 0
        if is_disease:
            final_code = 2 if is_pca_final else 1

        return {
            'step1': {'is_disease': is_disease, 'probs': prob1},
            'step2': {'probs': prob2, 'is_pca': is_pca_final, 'threshold': self.pca_threshold},
            'final_label_code': final_code,
            'global_probs': probs_dict
        }
    
# --------------------------------------------------
# 3. Model Loading
# --------------------------------------------------
DEFAULT_PATH = 'models'
MODEL1_FILE = 'disease_screening.joblib'
MODEL2_FILE = 'malignancy_differentiation.joblib'

@st.cache_resource
def models():
    p1 = os.path.join(DEFAULT_PATH, MODEL1_FILE)
    p2 = os.path.join(DEFAULT_PATH, MODEL2_FILE)
    
    if not os.path.exists(p1):
        p1 = MODEL1_FILE
        p2 = MODEL2_FILE
        
    try:
        b1 = joblib.load(p1)
        b2 = joblib.load(p2)
        return (b1, b2), None
    except Exception as e:
        return None, str(e)
    
# --------------------------------------------------
# 4. Help & Documentation Dialog
# --------------------------------------------------
@st.dialog('📘 Model Documentation & User Guide', width = 'large')
def show_help():
    st.markdown("""
    ### 1. Overview
    This tool is based on the study **"A novel biomarker-based model, AR⁺TREM2⁺, outperforms conventional markers in prostate cancer detection"**. It addresses the critical clinical challenge of differentiating Prostate Cancer (PCa) from Benign Prostatic Hyperplasia (BPH) and healthy controls.

    ### 2. Hierarchical Diagnostic Architecture
    The model employs a two-stage hierarchical classification strategy:
    
    *   **Stage 1: Disease Screening Model (Healthy vs. Disease)**
        *   **Top Features:** TPSA, LY%, HCT, RDW-CV, FPSA/TPSA, FPSA, Urea, HGB, LY#, NEUT#, MONO#, PLT.
        *   **Interaction:** `TPSA*AR+TREM2+` is also used in this stage to enhance sensitivity.
    
    *   **Stage 2: Malignancy Differentiation Model (BPH vs. PCa)**
        *   **Top Features:** TPSA, FPSA/TPSA, NEUT#, MCHC, MCH, Age.
        *   **Novel Interactions:** `APOE*AR+TREM2+` and `AFP*AR+TREM2+`.
        *   **Innovation:** This stage integrates novel immunophenotypic markers to capture tumor-specific immune microenvironmental alterations.

    ### 3. The Novel Biological Mechanism (AR⁺TREM2⁺)
    The core innovation of this model lies in the **APOE-TREM2-AR axis**:
    *   **Mechanism:** APOE in the tumor microenvironment binds to **TREM2** on macrophages, upregulating **Androgen Receptor (AR)** expression.
    *   **Effect:** This induces pathogenic immunosuppression. The interaction terms mathematically capture this synergistic crosstalk.

    ### 4. How to Use
    1.  **Input Data:** Enter the patient's clinical data. Ensure units match the labels.
    2.  **Threshold:** Adjust the "PCa Decision Threshold" in the sidebar if needed (Default 0.5).
    3.  **Run:** Click "🚀 Launch Prediction !".
    """)
    
# --------------------------------------------------
# 5. Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header('⚙️ PCa screening')
    
    if st.button('📘 User Guide & Documentation'):
        show_help()
        
    st.markdown('---')
    
    pca_threshold = st.slider('PCa Decision Threshold', 0.0, 1.0, 0.5, 0.01, help = 'Adjust sensitivity/specificity trade-off for PCa detection.')
    
    st.markdown('---')
    run_btn = st.button('🚀 Launch Prediction !', type = 'primary')

# --------------------------------------------------
# 6. Main Interface
# --------------------------------------------------
st.title('Prostate Cancer Hierarchical Diagnostic Application')

bundles, err = models()
if err:
    st.error(f'⚠️ Error: {err}')
    st.stop()

classifier = HierarchicalClassifier(
    bundles[0]['model'], bundles[1]['model'],
    bundles[0]['preprocessor'], bundles[1]['preprocessor'],
    pca_threshold
)

# ==========================================
#        INPUT FORM AREA
# ==========================================
# Reorganized by Clinical Category to avoid duplicates

# --- SECTION 1: Patient Info & Tumor Markers ---
st.markdown('<div class="section-header">1. Patient Info & Tumor Markers</div>', unsafe_allow_html = True)
c1, c2, c3, c4 = st.columns(4)

# Age (Used in Diagnosis)
age = c1.number_input('Age (year)', 0, 120, 75, step = 1, help = "Patient's age.")

# TPSA (Used in Screening & Diagnosis)
tpsa = c2.number_input('tPSA (µg/L)', 0.0, 1000.0, 1.44, format = '%.2f', step = 0.01, help = 'Total Prostate Specific Antigen.')

# FPSA (Used in Screening, and for Ratio calculation)
fpsa = c3.number_input('fPSA (µg/L)', 0.0, 1000.0, 0.416, format = '%.3f', step = 0.001, help = 'Free Prostate Specific Antigen.')

# AFP (Used in Diagnosis Interaction)
afp = c4.number_input('AFP (µg/L)', 0.0, 1000.0, 2.44, format = '%.2f', step = 0.01, help = 'Alpha-fetoprotein.')

# Ratio Calculation Display
ratio_val = fpsa / tpsa if tpsa > 0 else 0.0
# Modified display logic: Larger font and Red color for the number
st.markdown(f"ℹ️ Calculated fPSA/tPSA Ratio: <span style='color: #FF0000; font-size: 1.3rem; font-weight: bold;'>{ratio_val:.4f}</span> (Used in both models)", unsafe_allow_html = True)

# --- SECTION 2: Complete Blood Count (CBC) ---
st.markdown('<div class="section-header">2. Complete Blood Count (CBC)</div>', unsafe_allow_html = True)

c1, c2, c3, c4 = st.columns(4)
# NEUT# (Used in Screening & Diagnosis)
neut_abs = c1.number_input('NEUT# (×10⁹/L)', 0.0, 1000.0, 4.16, format = '%.2f', step = 0.01, help = 'Neutrophil Absolute Count.')

# LY% (Used in Screening)
ly_pct = c2.number_input('LY (%)', 0.0, 100.0, 27.7, format = '%.1f', step = 0.1, help = 'Lymphocyte percentage.')

# LY# (Used in Screening)
ly_abs = c3.number_input('LY# (×10⁹/L)', 0.0, 1000.0, 1.88, format = '%.2f', step = 0.01, help = 'Lymphocyte Absolute Count.')

# MONO# (Used in Screening)
mono_abs = c4.number_input('MONO# (×10⁹/L)', 0.0, 1000.0, 0.51, format = '%.2f', step = 0.01, help = 'Monocyte Absolute Count.')

c1, c2, c3, c4 = st.columns(4)
# HGB (Used in Screening)
hgb = c1.number_input('HGB (g/L)', 0.0, 1000.0, 96.00, format = '%.2f', step = 0.01, help = 'Hemoglobin.')

# HCT (Used in Screening)
hct = c2.number_input('HCT (%)', 0.0, 100.0, 29.2, format = '%.1f', step = 0.1, help = 'Hematocrit.')

# PLT (Used in Screening)
plt_cnt = c3.number_input('PLT (×10⁹/L)', 0.0, 1000.0, 205.00, format = '%.2f', step = 0.01, help = 'Platelet Count.')

# RDW-CV (Used in Screening)
rdw_cv = c4.number_input('RDW-CV (%)', 0.0, 100.0, 16.1, format = '%.1f', step = 0.1, help = 'Red Cell Distribution Width-CV.')

c1, c2, c3, c4 = st.columns(4)
# MCH (Used in Diagnosis)
mch = c1.number_input('MCH (pg)', 0.0, 1000.0, 21.20, format = '%.2f', step = 0.01, help = 'Mean Corpuscular Hemoglobin.')

# MCHC (Used in Diagnosis)
mchc = c2.number_input('MCHC (g/L)', 0.0, 1000.0, 329.00, format = '%.2f', step = 0.01, help = 'Mean Corpuscular Hemoglobin Concentration.')
c3.write('')
c4.write('')

# --- SECTION 3: Biochemistry & Advanced Markers ---
st.markdown('<div class="section-header">3. Biochemistry & Advanced Immunophenotyping</div>', unsafe_allow_html = True)

c1, c2, c3 = st.columns(3)
# Urea (Used in Screening)
urea = c1.number_input('Urea (mmol/L)', 0.0, 1000.0, 7.69, format = '%.2f', step = 0.01, help = 'Blood Urea Nitrogen.')

# APOE (Used in Diagnosis Interaction)
apoe = c2.number_input('APOE', 0.0, 10000.0, 46.55, format = '%.2f', step = 0.01, help = 'Apolipoprotein E expression.')

# AR+TREM2+ (Used in Interactions for both models)
ar_trem2_pos = c3.number_input('AR+TREM2+ ratio', 0.0, 1.0, 0.218, format = '%.3f', step = 0.001, help = 'Key Ratio: AR+TREM2+ cells. Used to calculate interaction terms.')

# --------------------------------------------------
# Result Display Function
# --------------------------------------------------
def get_progress_bar_html(label, prob, color, threshold = None):
    pct = prob * 100
    marker = ""
    if threshold is not None:
        t_pct = threshold * 100
        marker = f'<div style="position:absolute; left:{t_pct}%; top:-2px; bottom:-2px; width:2px; background:#333; z-index:5; border:1px solid #fff;" title="Threshold"></div>'
    
    html = f"""<div style="margin-bottom:12px;">
<div style="display:flex; justify-content:space-between; font-size:0.85rem; font-weight:bold; color:#555; margin-bottom:4px;">
<span>{label}</span>
<span style="color:{color}">{pct:.1f}%</span>
</div>
<div style="position:relative; width:100%; background-color:#e9ecef; border-radius:4px; height:10px;">
{marker}
<div style="width:{pct}%; background-color:{color}; height:100%; border-radius:4px;"></div>
</div>
</div>"""
    return html

@st.dialog("📊 Diagnostic Report", width = 'large')
def show_report(res):
    gp = res['global_probs']
    ph, pb, pp = gp['Healthy']*100, gp['BPH']*100, gp['PCa']*100
    
    st.markdown(f"""
    <div style="margin-bottom:20px;">
        <div style="font-weight:bold; margin-bottom:5px;">Global Probability Distribution</div>
        <div style="display:flex; height:20px; border-radius:10px; overflow:hidden; width:100%;">
            <div style="width:{ph}%; background:#28a745;" title="Healthy"></div>
            <div style="width:{pb}%; background:#fd7e14;" title="BPH"></div>
            <div style="width:{pp}%; background:#dc3545;" title="PCa"></div>
        </div>
        <div style="display:flex; gap:15px; font-size:0.8rem; margin-top:5px; color:#555;">
            <span style="color:#28a745">■ Healthy ({ph:.1f}%)</span>
            <span style="color:#fd7e14">■ BPH ({pb:.1f}%)</span>
            <span style="color:#dc3545">■ PCa ({pp:.1f}%)</span>
        </div>
    </div>
    <hr style="margin: 10px 0;">
    """, unsafe_allow_html = True)

    col1, col2 = st.columns(2, gap = 'medium')
    
    with col1:
        s1 = res['step1']
        is_dis = s1['is_disease']
        st.markdown(f"""
        <div class="result-card" style="border-left: 5px solid {'#ffc107' if is_dis else '#28a745'}">
            <div class="card-header">1️⃣ Screening <span class="card-sub">(Healthy vs. Disease)</span></div>
            <div style="font-size:1.2rem; font-weight:bold;">{'Risk Detected' if is_dis else 'Healthy'}</div>
        </div>
        """, unsafe_allow_html = True)
        bar_html = get_progress_bar_html('Disease Prob.', s1['probs'][1], '#ffc107' if is_dis else '#28a745')
        st.markdown(bar_html, unsafe_allow_html = True)

    with col2:
        if is_dis:
            s2 = res['step2']
            is_pca = s2['is_pca']
            color = '#dc3545' if is_pca else '#fd7e14'
            st.markdown(f"""
            <div class="result-card" style="border-left: 5px solid {color}">
                <div class="card-header">2️⃣ Diagnosis <span class="card-sub">(BPH vs. PCa)</span></div>
                <div style="font-size:1.2rem; font-weight:bold;">{'PCa (Malignant)' if is_pca else 'BPH (Benign)'}</div>
               
            </div>
            """, unsafe_allow_html = True)
            bar_html = get_progress_bar_html('Malignancy Prob.', s2['probs'][1], color, threshold = s2['threshold'])
            st.markdown(bar_html, unsafe_allow_html = True)
        else:
            st.markdown("""
            <div class="result-card" style="border-left: 5px solid #ccc; opacity: 0.6;">
                <div class="card-header">2️⃣ Diagnosis <span class="card-sub">(Not required)</span></div>
                <div style="font-size:1.2rem; font-weight:bold;">Skipped</div>
            </div>
            """, unsafe_allow_html = True)

    # Detailed Interpretation Section
    st.write('') 
    final_code = res['final_label_code']
    
    if final_code == 2: # PCa
        st.error("⚠️ **High Risk**: Immunological markers suggest malignancy (PCa).")
        st.markdown("""
        <div class="interpret-box">
            <b>🔍 Result Interpretation:</b><br>
            The model has identified a high probability of <b>Prostate Cancer (PCa)</b>. 
            <br><br>
            <b>🩺 Clinical Recommendation:</b><br>
            Immediate consultation with a urologist is recommended. Consider multiparametric MRI or biopsy for confirmation.
        </div>
        """, unsafe_allow_html = True)
        
    elif final_code == 1: # BPH
        st.warning("🔸 **Observation**: Markers suggest benign enlargement (BPH).")
        st.markdown("""
        <div class="interpret-box">
            <b>🔍 Result Interpretation:</b><br>
            The model predicts <b>Benign Prostatic Hyperplasia (BPH)</b>.
            <br><br>
            <b>🩺 Clinical Recommendation:</b><br>
            Regular monitoring of PSA levels and symptoms is advised. Invasive procedures (like biopsy) might be avoidable based on this risk assessment, subject to clinical judgment.
        </div>
        """, unsafe_allow_html = True)
        
    else: # Healthy
        st.success("✅ **Low Risk**: No further differential diagnosis needed.")
        st.markdown("""
        <div class="interpret-box">
            <b>🔍 Result Interpretation:</b><br>
            The model predicts the subject is <b>Healthy</b>.
            <br><br>
            No significant abnormalities were found during the screening phase, indicating a low likelihood of prostate disease.
            <br><br>
            <b>🩺 Clinical Recommendation:</b><br> Maintain a healthy lifestyle. Routine annual health check-ups are recommended.
        </div>
        """, unsafe_allow_html = True)

# --------------------------------------------------
# 8. Execution Logic
# --------------------------------------------------
if run_btn:
    # Calculate Ratio
    ratio_calc = fpsa / tpsa if tpsa > 0 else 0
    
    # Construct Data with Interactions calculated internally
    input_data = pd.DataFrame([{
        # --- Shared / Raw Features ---
        'TPSA': tpsa, 
        'FPSA': fpsa, 
        'FPSA/TPSA': ratio_calc,
        'age': age,
        
        # --- Blood Routine ---
        'LY%': ly_pct, 
        'HCT': hct, 
        'RDW-CV': rdw_cv,
        'Urea': urea, 
        'HGB': hgb, 
        'PLT': plt_cnt, 
        'LY#': ly_abs,
        'NEUT#': neut_abs,
        'MONO#': mono_abs,
        
        # --- Advanced ---
        'MCH': mch,
        'MCHC': mchc,
        'APOE': apoe,
        'AFP': afp,
        
        # --- Interaction Terms (Calculated) ---
        'TPSA*AR+TREM2+': tpsa * ar_trem2_pos,
        'APOE*AR+TREM2+': apoe * ar_trem2_pos,
        'AFP*AR+TREM2+': afp * ar_trem2_pos,
        
        # Keeping raw AR ratio just in case
        'AR+TREM2+': ar_trem2_pos
    }])
    
    try:
        result = classifier.predict_full_detail(input_data)
        show_report(result)
    except Exception as e:
        st.error(f'Prediction Error: {e}')
        st.write('Debug Info - Input Data:', input_data)