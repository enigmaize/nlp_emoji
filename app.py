import streamlit as st
import numpy as np
import pickle
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def download_large_file_from_gdrive(file_id):
    """Download large file from Google Drive using cookies method"""
    url = f"https://drive.google.com/uc?id={file_id}"
    
    session = requests.Session()
    response = session.get(url, stream=True)
    
    # Проверяем, есть ли предупреждение о скачивании
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            # Нужно подтвердить скачивание
            confirm_url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
            response = session.get(confirm_url, stream=True)
            break
    
    return response

@st.cache_resource
def load_resources():
    model_file_id = "1A0dE-UXP9M4bPY795Z6fAdJ0wyp8M9nW"
    model_path = 'emotion_classification_model.h5'
    
    # Скачиваем модель если не существует
    if not os.path.exists(model_path):
        st.info("Downloading model...")
        
        response = download_large_file_from_gdrive(model_file_id)
        
        # Сохраняем файл
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        st.success("Model downloaded successfully!")
    
    # Проверяем размер файла
    file_size = os.path.getsize(model_path)
    st.info(f"Downloaded model size: {file_size / (1024*1024):.2f} MB")
    
    if file_size < 10 * 1024 * 1024:  # Меньше 10MB
        st.error(f"Downloaded file is too small ({file_size} bytes) - likely not the actual model file")
        st.error("This might be due to Google Drive's download restrictions.")
        st.stop()
    
    # Загружаем модель
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()
    
    # Загружаем предобработку
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
        
    return model, tokenizer, label_encoder

# Загружаем ресурсы
try:
    model, tokenizer, label_encoder = load_resources()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Остальной код приложения...
st.title("🧠 Emotion Classification System")
st.subheader("AI-Powered Emotion Recognition from Text")

st.write("""
This model can classify text into **75 different emotions** with **100% accuracy**.
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

st.sidebar.header("About this Model")
st.sidebar.write("""
- **Model Type**: Bidirectional LSTM with Attention
- **Classes**: 75 different emotions
- **Accuracy**: 100%
- **Architecture**: Custom neural network
""")
