# ============================================================
# utils.py - All Helper Functions
# ============================================================
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import os
import json

# ============================================================
# CONSTANTS
# ============================================================
AGE_CLASSES = ['T1 (1-2 days) - Premium', 'T2 (3-4 days) - Good',
               'T3 (5-7 days) - Average', 'T4 (7+ days) - Not recommended']

DISEASE_CLASSES = [
    'Algal Leaf Spot', 'Brown Blight', 'Gray Blight',
    'Helopeltis', 'Red Spider', 'Green Mirid Bug', 'Healthy Leaf'
]

QUALITY_SCORES = {
    'T1 (1-2 days) - Premium': 95,
    'T2 (3-4 days) - Good': 75,
    'T3 (5-7 days) - Average': 50,
    'T4 (7+ days) - Not recommended': 25
}

COLORS = {
    'T1 (1-2 days) - Premium': '#28a745',
    'T2 (3-4 days) - Good': '#17a2b8',
    'T3 (5-7 days) - Average': '#ffc107',
    'T4 (7+ days) - Not recommended': '#dc3545'
}

# ============================================================
# MODEL LOADING (Cached)
# ============================================================
@st.cache_resource
def load_models():
    """Load both Age and Disease models."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, 'models')
    
    age_path = os.path.join(model_dir, 'age_model.h5')
    disease_path = os.path.join(model_dir, 'disease_model.h5')
    
    age_model = None
    disease_model = None
    
    if os.path.exists(age_path):
        try:
            age_model = tf.keras.models.load_model(age_path)
        except Exception as e:
            st.warning(f"⚠️ Could not load Age model: {e}")
    
    if os.path.exists(disease_path):
        try:
            disease_model = tf.keras.models.load_model(disease_path)
        except Exception as e:
            st.warning(f"⚠️ Could not load Disease model: {e}")
    
    return age_model, disease_model

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for prediction."""
    img = img.resize(target_size)
    img_array = np.array(img)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
def predict_age(model, img_array):
    """Predict age class."""
    if model is None:
        return "Model not loaded", 0.0
    pred = model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred) * 100)
    return AGE_CLASSES[class_idx], confidence

def predict_disease(model, img_array):
    """Predict disease class."""
    if model is None:
        return "Model not loaded", 0.0
    pred = model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred) * 100)
    return DISEASE_CLASSES[class_idx], confidence

# ============================================================
# RECOMMENDATION
# ============================================================
def get_plucking_recommendation(age_class):
    """Get plucking recommendation."""
    if 'T1' in age_class or 'T2' in age_class:
        return "✅ Pluck Now!", "success", "Best quality, harvest immediately"
    elif 'T3' in age_class:
        return "⏳ Wait 2-3 Days", "warning", "Check again in 48 hours"
    else:
        return "❌ Skip Plucking", "error", "Not suitable for tea production"

def get_quality_score(age_class):
    """Get quality score."""
    return QUALITY_SCORES.get(age_class, 0)

def get_color(age_class):
    """Get color for class."""
    return COLORS.get(age_class, '#6c757d')