# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 12:53:45 2025
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
st.set_page_config(page_title = 'Prostate Cancer Advanced Diagnostic Application', page_icon = '', layout = 'wide')

# Inject Global CSS
st.markdown("""
<style>
    /* Hide Streamlit Header */
    /* header[data-testid="stHeader"] { display: none; } */
    
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
    
    def __init__(self, model_step1, model_step2, preprocessor1, preprocessor2, pca_threshold=0.5):
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
        # If probabilities are 0.5/0.5, confidence is 0; if 0.9/0.1, confidence is 0.8.
        step2_confidence = abs(p_bph_cond - p_pca_cond)
        
        # If Step 1 determines the presence of disease, and Step 2 is highly confident.
        # (whether confident in BPH or PCa)
        # We use this confidence level to penalize the probability of being Healthy.
        if is_disease:
            # Penalty factor: the higher the confidence in Step 2, the more severely the Healthy probability is reduced.
            # Example: Step 2 confidence of 0.85 means only 15% of the original Healthy probability is retained.
            penalty_factor = 1.0 - step2_confidence
            
            # An intensity coefficient can also be added, e.g., penalty_factor = 1.0 - (step2_confidence * 0.8)
            # Here we use the strongest correction directly
            p_healthy = p_healthy * penalty_factor
            
            # Recalculate p_disease_total to keep the sum equal to 1 (after local adjustment)
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
# 4. Help & Documentation Dialog ---
# --------------------------------------------------
@st.dialog('📘 Model Documentation & User Guide', width = 'large')
def show_help():
    st.markdown("""
    ### 1. Overview
    This tool is based on the study **"A novel biomarker-based model, AR⁺TREM2⁺, outperforms conventional markers in prostate cancer detection"**. It addresses the critical clinical challenge of differentiating Prostate Cancer (PCa) from Benign Prostatic Hyperplasia (BPH) and healthy controls, particularly within the PSA gray zone.

    ### 2. Hierarchical Diagnostic Architecture
    The model employs a two-stage hierarchical classification strategy designed to mimic clinical workflow:
    
    *   **Stage 1: Disease Screening Model**
        *   **Goal:** Distinguish **Healthy** vs. **Prostatic Pathology (BPH or PCa)**.
        *   **Key Features:** tPSA, fPSA/tPSA ratio, Lymphocyte % (LY%), Hematocrit (HCT), Urea, etc.
        *   **Note:** Age is intentionally excluded in this stage to discover pathological signatures independent of natural aging.
    
    *   **Stage 2: Malignancy Differentiation Model**
        *   **Goal:** Distinguish **BPH** vs. **PCa** (Malignant).
        *   **Key Features:** tPSA, **APOE*AR⁺TREM2⁺**, **AFP*AR⁺TREM2⁺**, Neutrophil count (NEUT#), Monocyte % (MONO%), etc.
        *   **Innovation:** This stage integrates novel immunophenotypic markers to capture tumor-specific immune microenvironmental alterations.

    ### 3. The Novel Biological Mechanism (AR⁺TREM2⁺)
    The core innovation of this model lies in the **APOE-TREM2-AR axis**:
    *   **Mechanism:** APOE in the tumor microenvironment binds to **TREM2** on macrophages.
    *   **Effect:** This activates downstream signaling that upregulates **Androgen Receptor (AR)** expression.
    *   **Result:** This induces pathogenic immunosuppression, promoting tumor progression. The interaction term `APOE*AR⁺TREM2⁺` mathematically captures this non-linear synergistic crosstalk.

    ### 4. How to Use
    1.  **Input Data:** Enter the patient's clinical data in the main form. Ensure units match the labels.
    2.  **Threshold:** Adjust the "PCa Decision Threshold" in the sidebar if needed (Default 0.5). Lowering it increases sensitivity (catches more cancer but maybe more false alarms); raising it increases specificity.
    3.  **Run:** Click "🚀 Launch Prediction !".
    4.  **Interpret:** The report provides a global probability and a step-by-step breakdown.
    """)
    
# --------------------------------------------------
# 5. Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header('⚙️ PCa screening')
    
    # Help Button (Trigger Dialog)
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
# SECTION 1: Traditional Markers
st.markdown('<div class="section-header">1. Traditional Markers (Base)</div>', unsafe_allow_html = True)
c1, c2, c3, c4 = st.columns(4)

# TPSA and FPSA are the inputs, Ratio is calculated
tpsa = c1.number_input('tPSA (µg/L)', 0.0, 1000.0, 8.81, format = '%.2f', step = 0.01, help = 'Total Prostate Specific Antigen (tPSA): Reference range 0-4 ng/mL. This is a classic marker for prostate cancer; levels generally increase with the severity of malignant prostate disease.')
fpsa = c2.number_input('fPSA (µg/L)', 0.0, 1000.0, 1.92, format = '%.2f', step = 0.01, help = 'Free Prostate Specific Antigen (fPSA): Reference range 0-0.93 ng/mL. A classic marker for prostate cancer; levels generally increase with malignancy.')

# Calculate Ratio for display
ratio_val = fpsa / tpsa if tpsa > 0 else 0.0
c3.metric('Calculated fPSA/tPSA', f'{ratio_val:.4f}', help = 'Free/Total PSA Ratio: <0.25 indicates high risk of Prostate Cancer (CA); >0.25 indicates high risk of Benign Prostatic Hyperplasia (BPH).')
c4.write('')

# SECTION 2: Disease Screening Model
st.markdown('<div class="section-header">2. Disease Screening Model Parameters</div>', unsafe_allow_html = True)
st.caption('Parameters required for Healthy vs. Disease classification (excluding those already entered above).')

c1, c2, c3, c4 = st.columns(4)
# col.model1 includes: TPSA, FPSA/TPSA (Done), plus:
ly_pct = c1.number_input('LY (%)', 0.0, 100.0, 36.6, format = '%.1f', step = 0.1, help = 'Lymphocyte percentage (unit: %), with a reference range of 20.0–50.0, is often observed to be lower in prostatic disease patients compared to healthy controls.')
hct = c2.number_input('HCT (%)', 0.0, 100.0, 44.0, format = '%.1f', step = 0.1, help = 'Hematocrit. (unit: %), with a reference range of 40.0–50.0. Prostatic disease patients tend to have slightly lower HCT values.')
urea = c3.number_input('Urea (mmol/L)', 0.0, 100.0, 4.03, format = '%.2f', step = 0.01, help = 'Urea serves as a key indicator of kidney function. The typical reference range is 3.6-9.5 mmol/L. Patients with prostate diseases often exhibit elevated levels.')
hgb = c4.number_input('HGB (g/L)', 0.0, 1000.0, 143.32, format = '%.2f', step = 0.01, help = 'Hemoglobin (HGB) has a reference range of 130-175 g/L. Patients with prostate disease tend to have lower HGB levels compared to the healthy population.')

c1, c2, c3, c4 = st.columns(4)
ly_abs = c1.number_input('LY# (×10⁹/L)', 0.0, 100.0, 2.63, format = '%.2f', step = 0.01, help = 'Lymphocyte Absolute Count (unit: ×10⁹/L; reference range: 1.10–3.20), is often observed to be lower in prostatic disease patients compared to healthy controls.')
eo_abs = c2.number_input('EO# (×10⁹/L)', 0.0, 10.0, 0.11, format = '%.2f', step = 0.01, help = 'The Eosinophil Count (EO#) has a reference range of 0.02-0.52 ×10⁹/L. In patients with prostate disease, this indicator may be slightly lower than in the normal healthy population.')
plt_cnt = c3.number_input('PLT (×10⁹/L)', 0.0, 1000.0, 217.12, format = '%.2f', step = 0.01, help = 'The Platelet Count (PLT) has a reference range of 125-350 ×10⁹/L. This indicator may be slightly lower in patients compared to the normal healthy population.')
c4.write('')

# SECTION 3: Malignancy Differentiation Model
st.markdown('<div class="section-header">3. Malignancy Differentiation Model Parameters</div>', unsafe_allow_html = True)
st.caption('Parameters required for BPH vs. PCa classification. Enter these values to automatically calculate interaction terms like `APOE*AR+TREM2+` and `AFP*AR+TREM2+`.')

# Basic Clinical for Model 2
c1, c2, c3, c4 = st.columns(4)
age = c1.number_input('Age (year)', 0, 100, 77, step = 1, help = "Patient's age in years. Generally, the risk of prostate-related diseases increases with age.")
mch = c2.number_input('MCH (pg)', 0.0, 100.0, 28.70, format = '%.2f', step = 0.01, help = 'Mean Corpuscular Hemoglobin (MCH). Normal Reference Range: 27.0 - 34.0 pg.')
mchc = c3.number_input('MCHC (g/L)', 0.0, 1000.0, 336.10, format = '%.2f', step = 0.01, help = 'Mean Corpuscular Hemoglobin Concentration (MCHC). Normal Reference Range: 316 - 354 g/L.')
neut_abs = c4.number_input('NEUT# (×10⁹/L)', 0.0, 100.0, 3.83, format = '%.2f', step = 0.01, help = 'Neutrophil Count (NEUT#). Normal Reference Range: 1.80 - 6.30 ×10⁹/L.')

c1, c2, c3, c4 = st.columns(4)
mono_pct = c1.number_input('MONO (%)', 0.0, 100.0, 8.2, format = '%.1f', step = 0.1, help = 'Monocyte Percentage (MONO%). Normal Reference Range: 3.0 - 10.0 %.')
ar_trem2_neg = c2.number_input('AR+TREM2- ratio', 0.0, 1.0, 0.133, format = '%.3f', step = 0.001, help = 'AR+TREM2- ratio. Reference Range: 0 - 1.')
ar_trem2_pos = c3.number_input('AR+TREM2+ ratio', 0.0, 1.0, 0.358, format = '%.3f', step = 0.001, help = 'AR+TREM2+ ratio. Reference Range: 0 - 1.')
c4.write('')

# Interaction Components
c1, c2, c3, c4 = st.columns(4)
afp = c1.number_input('AFP (µg/L)', 0.0, 1000.0, 1.63, format = '%.2f', step = 0.01, help = 'Alpha-fetoprotein (AFP) Quantification, with a reference range of 0-7 ng/mL.')
apoe = c2.number_input('APOE', 0.0, 10000.0, 175.48, format = '%.2f', step = 0.01, help = 'Quantification of APOE expression.')
c3.write('')
c4.write('')

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
            <div class="result-card" style = "border-left: 5px solid #ccc; opacity: 0.6;">
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
            No significant abnormalities were found during the screening phase, indicating a low likehood of prostate disease.
            <br><br>
            <b>🩺 Clinical Recommendation:</b><br> Maintain a healthy lifestyle. Routine annual health check-ups are recommended.
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# 8. Execution Logic
# --------------------------------------------------
if run_btn:
    # Calculate Ratio
    ratio_calc = fpsa / tpsa if tpsa > 0 else 0
    
    # Construct Data with Interactions calculated internally
    input_data = pd.DataFrame([{
        # Base & Model 1
        'TPSA': tpsa, 
        'FPSA': fpsa, 
        'FPSA/TPSA': ratio_calc,
        'LY%': ly_pct, 
        'HCT': hct, 
        'Urea': urea, 
        'HGB': hgb, 
        'EO#': eo_abs, 
        'PLT': plt_cnt, 
        'LY#': ly_abs,
        
        # Model 2 Specifics
        'age': age,
        'MCH': mch,
        'MCHC': mchc,
        'NEUT#': neut_abs,
        'MONO%': mono_pct,
        'AR+TREM2-': ar_trem2_neg,
        
        # Interaction Terms (Calculated)
        'APOE*AR+TREM2+': apoe * ar_trem2_pos,
        'AFP*AR+TREM2+': afp * ar_trem2_pos,
        
        # Raw values kept just in case model needs them directly
        'APOE': apoe,
        'AFP': afp,
        'AR+TREM2+': ar_trem2_pos
    }])
    
    try:
        result = classifier.predict_full_detail(input_data)
        show_report(result)
    except Exception as e:
        st.error(f'Prediction Error: {e}')
        st.write('Debug Info - Input Columns:', input_data.columns.tolist())