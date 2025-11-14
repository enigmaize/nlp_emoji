import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
# We no longer need to define a full Layer class for this attempt
# from tensorflow.keras.layers import Layer
import tensorflow.keras.utils as utils

# --- Define Custom Function for the Layer Operation ---
# Define the function that the 'NotEqual' layer executes
def not_equal_layer_fn(x):
    # Often, 'NotEqual' layers compare against 0, especially for masking padded sequences.
    # This function mimics the operation: output = tf.not_equal(input_tensor, 0)
    # Adjust the '0' if the original layer compared against a different constant.
    # For now, we assume comparison with 0.
    return tf.not_equal(x, 0)

# --- Model Loading with Custom Function and safe_mode=False ---
@st.cache_resource # <-- Corrected: Use @st.cache_resource, not @st_cache_resource
def load_resources():
    model_path = 'emotion_model_compatible.h5'

    # Check if file exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found in repository!")
        st.stop()

    # Check file size
    file_size = os.path.getsize(model_path)
    st.info(f"Model size: {file_size / (1024*1024):.2f} MB")

    # Define custom objects dictionary, mapping the layer name to the function
    # This tells Keras how to interpret the 'NotEqual' layer when loading
    custom_objects = {
        'NotEqual': not_equal_layer_fn # Map 'NotEqual' to the function
        # If you encounter other custom functions/layers, add them here
        # e.g., 'MyCustomFunction': my_custom_function,
    }

    # Load model WITH custom objects scope AND safe_mode=False to handle Lambda layers
    try:
        with utils.custom_object_scope(custom_objects):
            # Add safe_mode=False to handle the Lambda layer
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        st.success("✅ Model loaded successfully with custom NotEqual function and safe_mode=False!")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

    # Load preprocessing
    with open('tokenizer_compatible.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder_compatible.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)

    return model, tokenizer, label_encoder

# Load resources
try:
    model, tokenizer, label_encoder = load_resources()
except Exception as e:
    st.error(f"❌ Critical error loading resources: {str(e)}")
    st.stop()

# App interface...
st.title("🧠 Emotion Classification System")
st.subheader("AI-Powered Emotion Recognition from Text")

st.write("""
This model can classify text into **75 different emotions** with high accuracy.
Enter any text below to see which emotion it represents!
""")

user_input = st.text_area(
    "Enter text for emotion classification:",
    height=150,
    placeholder="Type your text here... For example: 'Examine how Envy plays a role in leadership...'"
)

if st.button("Classify Emotion"):
    if user_input.strip():
        with st.spinner('Analyzing emotion...'):
            sequence = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(sequence, maxlen=512, padding='post', truncating='post')
            prediction = model.predict(padded, verbose=0)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            predicted_emotion = label_encoder.classes_[predicted_class_idx]
            confidence = prediction[0][predicted_class_idx]

            st.success(f"**Predicted Emotion:** {predicted_emotion}")
            st.info(f"**Confidence:** {confidence:.4f}")

            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_emotions = [label_encoder.classes_[idx] for idx in top_3_indices]
            top_3_confidences = [prediction[0][idx] for idx in top_3_indices]

            st.subheader("Top 3 Predictions:")
            for i, (emotion, conf) in enumerate(zip(top_3_emotions, top_3_confidences)):
                st.write(f"{i+1}. {emotion}: {conf:.4f}")

            # Visualize probabilities
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top_3_emotions, top_3_confidences)
            ax.set_xlabel('Confidence Score')
            ax.set_title('Top 3 Emotion Probabilities')
            st.pyplot(fig)

            st.subheader("Input Text:")
            st.write(user_input)
    else:
        st.warning("Please enter some text to classify!")

st.subheader("Try these sample texts:")
samples = [
    "I feel so angry about the unfair treatment I received today",
    "The joy of seeing my family after so long was overwhelming",
    "I'm constantly worried about everything that could go wrong",
    "The envy I feel towards my successful colleague is consuming me"
]

for i, sample in enumerate(samples):
    if st.button(f"Sample {i+1}", key=f"sample_{i}"):
        st.session_state.user_input = sample

# Additional model information
st.sidebar.header("About this Model")
st.sidebar.write("""
- **Model Type**: Bidirectional LSTM with Attention
- **Classes**: 75 different emotions
- **Architecture**: Custom neural network
- **Dataset**: 280,000 AI-generated question-answer pairs
""")

st.sidebar.header("How it Works")
st.sidebar.write("""
1. Text is processed through tokenization
2. Converted to numerical sequences
3. Passed through LSTM network
4. Attention mechanism identifies key emotional phrases
5. Output shows the predicted emotion
""")
