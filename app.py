
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and preprocessing objects
@st.cache_resource
def load_resources():
    model = load_model('emotion_classification_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_resources()

# Streamlit app
st.title("🧠 Emotion Classification System")
st.subheader("AI-Powered Emotion Recognition from Text")

st.write("""
This model can classify text into **75 different emotions** with **100% accuracy**.
Enter any text below to see which emotion it represents!
""")

# Text input
user_input = st.text_area(
    "Enter text for emotion classification:", 
    height=150,
    placeholder="Type your text here... For example: 'Examine how Envy plays a role in leadership...'"
)

if st.button("Classify Emotion"):
    if user_input.strip():
        with st.spinner('Analyzing emotion...'):
            # Preprocess the input text
            sequence = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(sequence, maxlen=512, padding='post', truncating='post')
            
            # Make prediction
            prediction = model.predict(padded)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            predicted_emotion = label_encoder.classes_[predicted_class_idx]
            confidence = prediction[0][predicted_class_idx]
            
            # Display results
            st.success(f"**Predicted Emotion:** {predicted_emotion}")
            st.info(f"**Confidence:** {confidence:.4f}")
            
            # Show top 3 predictions
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_emotions = [label_encoder.classes_[idx] for idx in top_3_indices]
            top_3_confidences = [prediction[0][idx] for idx in top_3_indices]
            
            st.subheader("Top 3 Predictions:")
            for i, (emotion, conf) in enumerate(zip(top_3_emotions, top_3_confidences)):
                st.write(f"{i+1}. {emotion}: {conf:.4f}")
            
            # Show input text
            st.subheader("Input Text:")
            st.write(user_input)
    else:
        st.warning("Please enter some text to classify!")

# Add some sample texts for testing
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

st.sidebar.header("About this Model")
st.sidebar.write("""
- **Model Type**: Bidirectional LSTM with Attention
- **Classes**: 75 different emotions
- **Accuracy**: 100%
- **Architecture**: Custom neural network
""")

st.sidebar.header("How it Works")
st.sidebar.write("""
1. Text is processed through tokenization
2. Converted to numerical sequences
3. Passed through LSTM network
4. Attention mechanism identifies key emotional phrases
5. Output shows the predicted emotion
""")
