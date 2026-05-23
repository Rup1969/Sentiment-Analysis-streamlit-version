import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- 1. CONFIGURATION & PAGE FORMAT ---
st.set_page_config(
    page_title="Sentiment Check", 
    page_icon="🧠", 
    layout="centered" # Keeps the working frame bounded
)

MODEL_PATH = "rup69/Sentiment-Analysis"
LABELS = ["anger", "fear", "joy", "sadness", "surprise"]

# --- 2. EXACT FORM INTERFACE DESIGN (CSS) ---
st.markdown("""
    <style>
    /* Force canvas background gradient */
    .stApp {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3f4f6 100%) !important;
    }
    
    /* Lock the main container block into a solid white document card */
    [data-testid="stColumn"] {
        background-color: white !important;
        border-radius: 12px !important;
        padding: 40px !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08) !important;
        border-top: 6px solid #6366f1 !important; /* Premium Indigo Top Border */
        border: 1px solid #e5e7eb !important;
    }

    /* Form Header Typography */
    .form-title {
        color: #4338ca !important;
        text-align: center !important;
        font-weight: 800 !important;
        font-size: 2.2em !important;
        margin-bottom: 2px !important;
        font-family: 'Inter', system-ui, sans-serif;
    }
    .form-subtitle {
        color: #6b7280 !important;
        text-align: center !important;
        margin-bottom: 25px !important;
        font-size: 1.05em !important;
        font-weight: 500;
    }

    /* Set the button styling to span full width */
    div.stButton > button:first-child {
        background-color: #4f46e5 !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.05em !important;
        width: 100% !important;
        border: none !important;
        padding: 12px 0px !important;
        border-radius: 6px !important;
        letter-spacing: 0.5px !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #4338ca !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. CORE LAYOUT ARCHITECTURE ---
# We create a single column block to wrap all elements inside our white card container
col_card = st.columns(1)[0]

with col_card:
    # Header Section
    st.markdown('<h1 class="form-title">🧠 Sentiment Check</h1>', unsafe_allow_html=True)
    st.markdown('<p class="form-subtitle">Advanced Neural Emotion Detection System</p>', unsafe_allow_html=True)
    
    # --- 4. RESOURCE CACHE LOADING ---
    @st.cache_resource
    def load_pipeline():
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)
        return tokenizer, model

    try:
        tokenizer, model = load_pipeline()
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        model = None

    # --- 5. USER CONSOLE INPUT ---
    st.markdown("### 1. Source Text")
    text = st.text_area(
        label="Input Window",
        placeholder="Type your text here to analyze underlying emotional markers...", 
        height=130,
        label_visibility="collapsed"
    )
    
    # Execution Trigger Button
    submit_trigger = st.button("GENERATE REPORT", type="primary")
    
    # Horizontal Rule Break
    st.markdown("<hr style='margin: 25px 0; border: 0; border-top: 1px solid #e5e7eb;'>", unsafe_allow_html=True)
    
    # --- 6. MODEL CONSOLE OUTPUT ---
    st.markdown("### 2. Analysis Results")
    
    if submit_trigger:
        if not text.strip():
            st.warning("⚠️ Please type something first!")
        elif model is None:
            st.error("❌ System Error: Model engine offline.")
        else:
            # Tokenization & Safe CPU Processing
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.sigmoid(outputs.logits).numpy()[0]
            
            detected_emotions = []
            text_lower = text.lower()
            
            # Contextual Threshold Trimming
            penalties = {
                "sleep": {"fear": 0.50}, 
                "tired": {"fear": 0.40}, 
                "hungry": {"fear": 0.40}
            }
            
            # Map out values
            score_list = []
            for i, label in enumerate(LABELS):
                score = float(probs[i])
                for word, penalty_map in penalties.items():
                    if word in text_lower and label in penalty_map:
                        score = max(0.0, score - penalty_map[label])
                score_list.append((label, score))
                
            # Perform sorting directly inside the dictionary map
            score_list.sort(key=lambda x: x[1], reverse=True)
            
            # Render beautifully sorted metrics inside the white layout card
            for label, score in score_list:
                st.write(f"**{label.upper()}** ({score * 100:.1f}%)")
                st.progress(score)
                if score > 0.5:
                    detected_emotions.append(label.upper())
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Execution Verdict Window
            if not detected_emotions:
                st.info("😐 **Verdict:** No strong classification-level emotion detected.")
            else:
                verdict_string = " + ".join([f"**{e}**" for e in detected_emotions])
                st.success(f"📊 **Primary Markers Detected:** {verdict_string}")
    else:
        st.markdown("<span style='color: #9ca3af;'>*Waiting for text input report generation...*</span>", unsafe_allow_html=True)

# --- 7. APPLICATION BOTTOM CANVAS FOOTER ---
st.markdown(
    "<div style='text-align:center; color:#9ca3af; margin-top:25px; font-size:0.8em;'>Secure Cloud Deployment • Framework: Streamlit</div>", 
    unsafe_allow_html=True
)
