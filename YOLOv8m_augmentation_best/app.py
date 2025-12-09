import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Waste Detection AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Stats styling */
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
    }
    
    /* Success/Error messages */
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">♻️ Waste Detection AI</h1>
        <p class="subtitle">Powered by YOLOv8 - Advanced Computer Vision for Environmental Protection</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    st.markdown("---")
    
    st.markdown("### 📊 About")
    st.info("""
        This AI-powered application uses YOLOv8 to detect and classify waste in images.
        
        **Features:**
        - Real-time waste detection
        - High accuracy classification
        - Visual annotations
        - Detailed statistics
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Tips")
    st.markdown("""
        - Use clear, well-lit images
        - Ensure waste is visible
        - Higher confidence = fewer false positives
    """)

# Define model path
base_dir = r"C:\Users\user\Documents\M-AI_S3\Computer_Vision\P_CV2"
model_path = os.path.join(base_dir, 'runs', 'yolov8m_augmented5', 'weights', 'best.pt')

# Check if model exists
if not os.path.exists(model_path):
    st.markdown(f"""
        <div class="error-message">
            <strong>⚠️ Model Not Found</strong><br>
            The model file could not be located at: <code>{model_path}</code><br>
            Please verify the path and try again.
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load model
@st.cache_resource
def load_model():
    with st.spinner('🔄 Loading AI model...'):
        model = YOLO(model_path)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 Original Image", use_column_width=True)

with col2:
    if uploaded_file is not None:
        st.markdown("### 🎯 Detection Results")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔍 Analyzing image...")
        progress_bar.progress(30)
        
        # Perform prediction
        try:
            results = model.predict(
                source=image,
                conf=confidence_threshold,
                save=False,
                verbose=False
            )
            
            progress_bar.progress(70)
            status_text.text("🎨 Generating annotations...")
            
            # Display results
            for r in results:
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])
                
                progress_bar.progress(100)
                status_text.text("✅ Detection complete!")
                
                st.image(im, caption="🔍 Detected Waste", use_column_width=True)
                
                # Display statistics
                num_detections = len(r.boxes)
                
                if num_detections == 0:
                    st.markdown("""
                        <div class="info-card">
                            <p style="text-align: center; color: #666; font-size: 1.1rem;">
                                ℹ️ No waste detected in this image
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="stat-box">
                            <p class="stat-number">{num_detections}</p>
                            <p class="stat-label">Waste Item{'s' if num_detections > 1 else ''} Detected</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed detection info
                    with st.expander("📋 View Detection Details"):
                        for i, box in enumerate(r.boxes):
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = model.names[class_id]
                            
                            st.markdown(f"""
                                **Detection {i+1}:**
                                - Class: `{class_name}`
                                - Confidence: `{confidence:.2%}`
                            """)
            
        except Exception as e:
            st.error(f"❌ Error during detection: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    else:
        st.markdown("""
            <div class="info-card">
                <p style="text-align: center; color: #666; font-size: 1.1rem;">
                    👈 Upload an image to start detection
                </p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Powered by YOLOv8 & Streamlit | Built for Environmental Conservation 🌍</p>
    </div>
""", unsafe_allow_html=True)