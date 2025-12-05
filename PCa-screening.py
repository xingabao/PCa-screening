import streamlit as st
import pandas as pd
import joblib
import os
import warnings

# --- 0. Suppress Warnings ---
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Prostate Cancer Advanced Diagnostic System",
    page_icon="🧬",
    layout="wide"
)

# Inject Global CSS
st.markdown("""
<style>
    /* Hide Streamlit Header */
    header[data-testid="stHeader"] { display: none; }
    
    /* Adjust top margin */
    .stApp { background-color: #f8f9fa; margin-top: -80px; }
    
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

    /* --- FORCE DIALOG WIDTH TO ~800px --- */
    div[data-testid="stDialog"] div[role="dialog"] {
        width: 800px !important;
        max-width: 90vw !important; 
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Core Logic Class ---

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
        # --- Step 1: Healthy vs Disease ---
        cols1 = self.features1 if self.features1 is not None else X.columns
        X_step1 = X.copy()
        for col in cols1:
            if col not in X_step1.columns: X_step1[col] = 0
                
        X1 = self.preprocessor1.transform(X_step1[cols1])
        prob1 = self.model_step1.predict_proba(X1)[0]
        
        p_healthy = prob1[0]
        p_disease_total = prob1[1]
        is_disease = p_disease_total > 0.5 
        
        step2_result = None
        final_code = 0 # 0:Healthy, 1:BPH, 2:PCa
        
        # --- Step 2: BPH vs PCa ---
        p_bph_cond = 0.0
        p_pca_cond = 0.0
        
        if is_disease:
            cols2 = self.features2 if self.features2 is not None else X.columns
            X_step2 = X.copy()
            for col in cols2:
                if col not in X_step2.columns: X_step2[col] = 0
            
            X2 = self.preprocessor2.transform(X_step2[cols2])
            prob2 = self.model_step2.predict_proba(X2)[0]
            
            p_bph_cond = prob2[0]
            p_pca_cond = prob2[1]
            
            is_pca = p_pca_cond > self.pca_threshold
            final_code = 2 if is_pca else 1
            
            step2_result = {
                "probs": prob2,
                "is_pca": is_pca,
                "threshold": self.pca_threshold
            }
        
        # Calculate Global Probabilities
        global_bph = p_disease_total * p_bph_cond if is_disease else 0
        global_pca = p_disease_total * p_pca_cond if is_disease else 0
        total = p_healthy + global_bph + global_pca
        if total == 0: total = 1
            
        return {
            "step1": {"is_disease": is_disease, "probs": prob1},
            "step2": step2_result,
            "final_label_code": final_code,
            "global_probs": {
                "Healthy": p_healthy / total,
                "BPH": global_bph / total,
                "PCa": global_pca / total
            }
        }

# --- 3. Model Loading ---

# DEFAULT_PATH = "E:/BaiduSyncdisk/005.Bioinformatics/SCI/005/Manuscript/Scripts/models"
MODEL1_FILE = "voting_model_model12.joblib"
MODEL2_FILE = "voting_model_model22.joblib"

@st.cache_resource
def load_models():
    # p1 = os.path.join(DEFAULT_PATH, MODEL1_FILE)
    # p2 = os.path.join(DEFAULT_PATH, MODEL2_FILE)
    p1 = MODEL1_FILE
    p2 = MODEL2_FILE
    
    if not os.path.exists(p1):
        p1 = MODEL1_FILE
        p2 = MODEL2_FILE
        
    # if not os.path.exists(p1):
    #     return None, f"Model files not found in {DEFAULT_PATH} or current dir."
        
    try:
        b1 = joblib.load(p1)
        b2 = joblib.load(p2)
        return (b1, b2), None
    except Exception as e:
        return None, str(e)

# --- 4. Help & Documentation Dialog ---

@st.dialog("📘 Model Documentation & User Guide", width="large")
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

    ### 4. Performance Validation
    The model was validated in a large-scale, multi-center study (N=649) across Guangdong and Anhui provinces:
    *   **Screening Stage:** Achieved an AUC of **0.983**, significantly outperforming baseline models.
    *   **Differentiation Stage:** Achieved an AUC of **0.851** (internal) and **0.874** (external validation).
    *   **Clinical Benefit:** Decision Curve Analysis (DCA) shows superior net benefit across clinically reasonable threshold probabilities, effectively reducing unnecessary biopsies.

    ### 5. How to Use
    1.  **Input Data:** Enter the patient's clinical data in the main form. Ensure units match the labels.
    2.  **Threshold:** Adjust the "PCa Decision Threshold" in the sidebar if needed (Default 0.5). Lowering it increases sensitivity (catches more cancer but maybe more false alarms); raising it increases specificity.
    3.  **Run:** Click "Run AI Diagnosis".
    4.  **Interpret:** The report provides a global probability and a step-by-step breakdown.
    """)

# --- 5. Sidebar ---

with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Help Button (Trigger Dialog)
    if st.button("📘 Help & Documentation"):
        show_help()
        
    st.markdown("---")
    
    pca_threshold = st.slider("PCa Decision Threshold", 0.0, 1.0, 0.5, 0.01, help="Adjust sensitivity/specificity trade-off for PCa detection.")
    
    st.markdown("---")
    run_btn = st.button("🚀 Run AI Diagnosis", type="primary")

# --- 6. Main Interface ---

st.title("🧬 Prostate Cancer Hierarchical Diagnostic System")

bundles, err = load_models()
if err:
    st.error(f"⚠️ Error: {err}")
    st.stop()

classifier = HierarchicalClassifier(
    bundles[0]['model'], bundles[1]['model'],
    bundles[0]['preprocessor'], bundles[1]['preprocessor'],
    pca_threshold
)

# ==========================================
#        INPUT FORM AREA
# ==========================================

# --- SECTION 1: Traditional Markers ---
st.markdown('<div class="section-header">1. Traditional Markers (Base)</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
# TPSA and FPSA are the inputs, Ratio is calculated
tpsa = c1.number_input("tPSA (µg/L)", 0.0, 5000.0, 179.0, format="%.2f")
fpsa = c2.number_input("fPSA (µg/L)", 0.0, 800.0, 18.6, format="%.2f")

# Calculate Ratio for display
ratio_val = fpsa / tpsa if tpsa > 0 else 0.0
c3.metric("Calculated fPSA/tPSA", f"{ratio_val:.4f}")


# --- SECTION 2: Disease Screening Model ---
st.markdown('<div class="section-header">2. Disease Screening Model Parameters</div>', unsafe_allow_html=True)
st.caption("Parameters required for Healthy vs. Disease classification (excluding those already entered above).")

c1, c2, c3, c4 = st.columns(4)
# col.model1 includes: TPSA, FPSA/TPSA (Done), plus:
ly_pct = c1.number_input("LY (%)", 0.0, 100.0, 20.9, format="%.1f")
hct = c2.number_input("HCT (%)", 10.0, 70.0, 44.0, format="%.1f")
urea = c3.number_input("Urea (mmol/L)", 0.0, 60.0, 4.03, format="%.2f")
hgb = c4.number_input("HGB (g/L)", 50.0, 250.0, 143.0, format="%.1f")

c1, c2, c3, c4 = st.columns(4)
eo_abs = c1.number_input("EO# (*10^9/L)", 0.0, 5.0, 0.29, format="%.2f")
plt_cnt = c2.number_input("PLT (*10^9/L)", 0.0, 600.0, 313.0, format="%.1f")
ly_abs = c3.number_input("LY# (*10^9/L)", 0.0, 20.0, 1.7, format="%.2f")
c4.write("") # Spacer


# --- SECTION 3: Malignancy Differentiation Model ---
st.markdown('<div class="section-header">3. Malignancy Differentiation Model Parameters</div>', unsafe_allow_html=True)
st.caption("Parameters required for BPH vs. PCa classification. Interaction terms are calculated automatically.")

# Basic Clinical for Model 2
c1, c2, c3, c4 = st.columns(4)
age = c1.number_input("Age (year)", 20, 100, 65)
mch = c2.number_input("MCH (pg)", 10.0, 50.0, 30.4, format="%.1f")
mchc = c3.number_input("MCHC (g/L)", 200.0, 500.0, 325.0, format="%.1f")
neut_abs = c4.number_input("NEUT# (*10^9/L)", 0.0, 20.0, 5.5, format="%.2f")

c1, c2, c3, c4 = st.columns(4)
mono_pct = c1.number_input("MONO (%)", 0.0, 60.0, 7.4, format="%.1f")
# AR+TREM2- is a standalone variable in Model 2
ar_trem2_neg = c2.number_input("AR+TREM2- (%)", 0.0, 100.0, 0.0023, format="%.4f")
c3.write("")
c4.write("")

# Interaction Components
st.markdown("**Interaction Term Components**")
st.caption("Enter these values to calculate `APOE*AR+TREM2+` and `AFP*AR+TREM2+`.")
c1, c2, c3 = st.columns(3)
apoe = c1.number_input("APOE (mg/L)", 0.0, 2000.0, 890.31, format="%.2f")
afp = c2.number_input("AFP (µg/L)", 0.0, 500.0, 1.96, format="%.2f")
ar_trem2_pos = c3.number_input("AR+TREM2+ (%)", 0.0, 100.0, 0.0294, format="%.4f")


# --- 7. Result Display Function ---

def get_progress_bar_html(label, prob, color, threshold=None):
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

@st.dialog("📊 Diagnostic Report", width="large")
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
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        s1 = res['step1']
        is_dis = s1['is_disease']
        st.markdown(f"""
        <div class="result-card" style="border-left: 5px solid {'#ffc107' if is_dis else '#28a745'}">
            <div class="card-header">1️⃣ Screening</div>
            <div style="font-size:1.2rem; font-weight:bold;">{'Risk Detected' if is_dis else 'Healthy'}</div>
            <div class="card-sub">Healthy vs. Disease</div>
        </div>
        """, unsafe_allow_html=True)
        bar_html = get_progress_bar_html("Disease Prob.", s1['probs'][1], "#ffc107" if is_dis else "#28a745")
        st.markdown(bar_html, unsafe_allow_html=True)

    with col2:
        if is_dis:
            s2 = res['step2']
            is_pca = s2['is_pca']
            color = "#dc3545" if is_pca else "#fd7e14"
            st.markdown(f"""
            <div class="result-card" style="border-left: 5px solid {color}">
                <div class="card-header">2️⃣ Diagnosis</div>
                <div style="font-size:1.2rem; font-weight:bold;">{'PCa (Malignant)' if is_pca else 'BPH (Benign)'}</div>
                <div class="card-sub">BPH vs. PCa</div>
            </div>
            """, unsafe_allow_html=True)
            bar_html = get_progress_bar_html("Malignancy Prob.", s2['probs'][1], color, threshold=s2['threshold'])
            st.markdown(bar_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card" style="border-left: 5px solid #ccc; opacity: 0.6;">
                <div class="card-header">2️⃣ Diagnosis</div>
                <div style="font-size:1.2rem; font-weight:bold;">Skipped</div>
                <div class="card-sub">Not required</div>
            </div>
            """, unsafe_allow_html=True)

    # --- Detailed Interpretation Section ---
    st.write("") 
    final_code = res['final_label_code']
    
    if final_code == 2: # PCa
        st.error("⚠️ **High Risk**: Immunological markers suggest malignancy (PCa).")
        st.markdown("""
        <div class="interpret-box">
            <b>🔍 Result Interpretation:</b><br>
            The model has identified a high probability of <b>Prostate Cancer (PCa)</b>. 
            <br><br>
            <b>Biological Context:</b> This classification is likely driven by elevated levels of the <b>APOE*AR⁺TREM2⁺</b> interaction term. 
            According to our study, the interaction between APOE and AR⁺TREM2⁺ macrophages creates an immunosuppressive microenvironment that promotes tumor progression. 
            Unlike BPH, which may show elevated PSA, this specific immunophenotypic signature is highly specific to malignancy.
            <br><br>
            <b>🩺 Clinical Recommendation:</b> Immediate consultation with a urologist is recommended. Consider multiparametric MRI or biopsy for confirmation.
        </div>
        """, unsafe_allow_html=True)
        
    elif final_code == 1: # BPH
        st.warning("🔸 **Observation**: Markers suggest benign enlargement (BPH).")
        st.markdown("""
        <div class="interpret-box">
            <b>🔍 Result Interpretation:</b><br>
            The model predicts <b>Benign Prostatic Hyperplasia (BPH)</b>.
            <br><br>
            <b>Biological Context:</b> While the screening model detected prostatic pathology (possibly due to elevated PSA or inflammation), the malignancy differentiation model did not find the specific <b>AR⁺TREM2⁺</b> immune signature associated with cancer. 
            This suggests the condition is likely benign enlargement rather than a malignant tumor.
            <br><br>
            <b>🩺 Clinical Recommendation:</b> Regular monitoring of PSA levels and symptoms is advised. Invasive procedures (like biopsy) might be avoidable based on this risk assessment, subject to clinical judgment.
        </div>
        """, unsafe_allow_html=True)
        
    else: # Healthy
        st.success("✅ **Low Risk**: No further differential diagnosis needed.")
        st.markdown("""
        <div class="interpret-box">
            <b>🔍 Result Interpretation:</b><br>
            The model predicts the subject is <b>Healthy</b>.
            <br><br>
            <b>Biological Context:</b> The systemic inflammatory markers (like LY%, HCT) and prostate-specific antigens are within ranges typical for the control group. 
            The absence of significant deviations in the screening stage indicates a low probability of prostatic pathology.
            <br><br>
            <b>🩺 Clinical Recommendation:</b> Maintain a healthy lifestyle. Routine annual health check-ups are recommended.
        </div>
        """, unsafe_allow_html=True)

# --- 8. Execution Logic ---

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
        st.error(f"Prediction Error: {e}")
        st.write("Debug Info - Input Columns:", input_data.columns.tolist())