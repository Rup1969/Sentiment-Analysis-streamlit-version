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

# --- 2. HEADER INTERFACE ---
st.title("🧠 Sentiment Check")
st.caption("Advanced Neural Emotion Detection System • Powered by RoBERTa")
st.markdown("---")

# --- 3. CACHED RESOURCE LOADING ---
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
    st.success("🚀 Neural Network Engine Active & Ready")
except Exception as e:
    st.error(f"Initialization Error: {e}")
    model = None

# --- 4. USER INPUT ---
st.markdown("### 1. Source Text")
text = st.text_area(
    label="Text Input Box",
    placeholder="Type your text here to analyze underlying emotional markers...", 
    height=150,
    label_visibility="collapsed"
)

# --- 5. INFERENCE & LOGIC ---
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
        
        st.markdown("### 2. Analysis Results")
        
        detected_emotions = []
        text_lower = text.lower()
        
        penalties = {
            "sleep": {"fear": 0.50}, 
            "tired": {"fear": 0.40}, 
            "hungry": {"fear": 0.40}
        }
        
        # 1. Gather all scores
        score_list = []
        for i, label in enumerate(LABELS):
            score = float(probs[i])
            for word, penalty_map in penalties.items():
                if word in text_lower and label in penalty_map:
                    score = max(0.0, score - penalty_map[label])
            score_list.append((label, score))
            
        # 2. Sort descending (highest score first)
        score_list.sort(key=lambda x: x[1], reverse=True)
        
        # 3. Render elements sequentially
        for label, score in score_list:
            st.write(f"**{label.upper()}** ({score * 100:.1f}%)")
            st.progress(score)
            if score > 0.5:
                detected_emotions.append(label.upper())
        
        st.markdown("---")
        
        # --- 6. FINAL VERDICT ---
        if not detected_emotions:
            st.info("😐 **Verdict:** No strong classification-level emotion detected.")
        else:
            verdict_string = " + ".join([f"**{e}**" for e in detected_emotions])
            st.success(f"📊 **Primary Markers Detected:** {verdict_string}")
            
st.markdown(
    "<div style='text-align:center; color:#9ca3af; margin-top:30px; font-size:0.8em;'>Secure Cloud Deployment • Framework: Streamlit</div>", 
    unsafe_allow_html=True
)
