import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- 1. PAGE SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="Sentiment Check", 
    page_icon="🧠", 
    layout="centered"
)

MODEL_PATH = "rup69/Sentiment-Analysis"
LABELS = ["anger", "fear", "joy", "sadness", "surprise"]

# --- 2. BRING BACK YOUR HUGGINGFACE PREMIUM CSS ---
st.markdown("""
    <style>
    /* 1. Canvas Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3f4f6 100%) !important;
    }
    
    /* 2. Overriding Streamlit containers to look like your sleek card */
    div[data-testid="stVerticalBlock"] > div:has(div.form-header) {
        background: white !important;
        border-radius: 12px !important;
        padding: 35px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08) !important;
        border-top: 6px solid #6366f1 !important; /* Indigo Top Bar */
        border: 1px solid #e5e7eb !important;
    }

    /* 3. Sleek Form Header Styles */
    .form-header h1 {
        color: #4338ca !important;
        text-align: center !important;
        font-weight: 800 !important;
        margin-bottom: 5px !important;
        font-family: 'Inter', system-ui, sans-serif;
    }
    .form-header p {
        color: #6b7280 !important;
        text-align: center !important;
        margin-bottom: 20px !important;
        font-size: 1.1em;
    }

    /* 4. Primary Button Color Override */
    div.stButton > button:first-child {
        background-color: #4f46e5 !important;
        color: white !important;
        font-weight: bold !important;
        width: 100% !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 6px !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338ca !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. THE "FORM CARD" WRAPPER STARTS HERE ---
# We inject this tracking class to group everything inside our CSS card frame
st.markdown('<div class="form-header"><h1>🧠 Sentiment Check</h1><p>Advanced Neural Emotion Detection System</p></div>', unsafe_allow_html=True)

# --- 4. CACHED RESOURCE LOADING ---
@st.cache_resource
def load_pipeline():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, 
        low_cpu_mem_usage=True
    )
    return tokenizer, model

try:
    tokenizer, model = load_pipeline()
except Exception as e:
    st.error(f"Initialization Error: {e}")
    model = None

# --- 5. USER INPUT ---
st.markdown("### 1. Source Text")
text = st.text_area(
    label="Text Input Box",
    placeholder="Type your text here to analyze underlying emotional markers...", 
    height=120,
    label_visibility="collapsed"
)

# --- 6. INFERENCE & SORTING LOGIC ---
if st.button("GENERATE REPORT", type="primary"):
    if not text.strip():
        st.warning("⚠️ Please type something first!")
    elif model is None:
        st.error("❌ System Error: Model engine failed to initialize.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).numpy()[0]
        
        st.markdown("<br>### 2. Analysis Results", unsafe_allow_html=True)
        
        detected_emotions = []
        text_lower = text.lower()
        
        penalties = {
            "sleep": {"fear": 0.50}, 
            "tired": {"fear": 0.40}, 
            "hungry": {"fear": 0.40}
        }
        
        # Calculate scores
        score_list = []
        for i, label in enumerate(LABELS):
            score = float(probs[i])
            for word, penalty_map in penalties.items():
                if word in text_lower and label in penalty_map:
                    score = max(0.0, score - penalty_map[label])
            score_list.append((label, score))
            
        # Sort scores in descending order
        score_list.sort(key=lambda x: x[1], reverse=True)
        
        # Render clean elements inside the card layout
        for label, score in score_list:
            st.write(f"**{label.upper()}** ({score * 100:.1f}%)")
            st.progress(score)
            if score > 0.5:
                detected_emotions.append(label.upper())
        
        st.markdown("---")
        
        # Final Verdict Display
        if not detected_emotions:
            st.info("😐 **Verdict:** No strong classification-level emotion detected.")
        else:
            verdict_string = " + ".join([f"**{e}**" for e in detected_emotions])
            st.success(f"📊 **Primary Markers Detected:** {verdict_string}")
            
# --- 7. FOOTER OUTSIDE THE CARD ---
st.markdown(
    "<div style='text-align:center; color:#9ca3af; margin-top:30px; font-size:0.8em;'>Secure Cloud Deployment • Framework: Streamlit</div>", 
    unsafe_allow_html=True
)
