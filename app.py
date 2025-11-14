import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
# Remove the direct import of load_model, use tf.keras.models.load_model instead
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer # Import Layer base class
import tensorflow.keras.utils as utils # Import utils for registration if needed

# --- Define Custom NotEqual Layer Class (Updated for Positional Arg) ---
class NotEqual(Layer):
    def __init__(self, comparison_value, **kwargs): # Accept comparison_value as first positional arg
        """
        Initializes the NotEqual layer.

        Args:
            comparison_value: The value (or tensor) to compare the input against.
                              This is passed as a positional argument when the layer was saved.
            **kwargs: Standard Keras layer keyword arguments.
        """
        # Check if comparison_value is a dictionary (indicating it's a config being passed)
        # This handles potential cases where from_config might pass the config dict directly
        if isinstance(comparison_value, dict):
            # If so, treat it as the config and call super's __init__ first
            # Then update with the config manually, which is less common but possible
            # A more robust way is to ensure get_config/from_config handles it correctly.
            # However, the primary goal is to accept the saved positional arg.
            # Let's assume it's the value unless explicitly a config during standard load.
            # The key is that during loading, the first arg passed *is* the value.
            super(NotEqual, self).__init__(**comparison_value) # This handles kwargs like name, trainable
            self.comparison_value = None # Placeholder, config should set it
            # Rebuild comparison_value from the config that was passed as first arg
            self.comparison_value = comparison_value.get('comparison_value', 0) # Default if not found
            self._comparison_value_for_config = self.comparison_value
        else:
            # Standard case: first arg is the comparison value
            super(NotEqual, self).__init__(**kwargs)
            self.comparison_value = comparison_value
            self._comparison_value_for_config = comparison_value # Store for config

    def call(self, inputs):
        """
        Defines the computation performed by the layer.

        Args:
            inputs: Input tensor.

        Returns:
            A boolean tensor where each element is True if the corresponding
            input element is not equal to `comparison_value`, False otherwise.
        """
        # Perform the 'not equal' operation
        # comparison_value might be a scalar or a tensor compatible with inputs
        return tf.not_equal(inputs, self.comparison_value)

    def get_config(self):
        """
        Provides the config for serialization.
        """
        config = super(NotEqual, self).get_config()
        config.update({"comparison_value": self._comparison_value_for_config})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer instance from its config.
        This method is called during loading if get_config was used during saving.
        """
        # The config will contain 'comparison_value' and other Layer config items
        # We pass the comparison_value as the first positional arg to __init__
        comparison_value = config.pop('comparison_value', 0) # Get and remove from config
        # The remaining items in config are Layer kwargs (like name, trainable, etc.)
        return cls(comparison_value, **config) # Pass value first, then kwargs


# --- Model Loading with Custom Object Scope and safe_mode=False ---
@st.cache_resource
def load_resources():
    model_path = 'emotion_model_compatible.h5'

    # Check if file exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file {model_path} not found in repository!")
        st.stop()

    # Check file size
    file_size = os.path.getsize(model_path)
    st.info(f"Model size: {file_size / (1024*1024):.2f} MB")

    # Define custom objects dictionary, mapping the layer name to the class
    custom_objects = {
        'NotEqual': NotEqual # Map 'NotEqual' to the updated class definition
        # If you encounter other custom objects later, add them here
        # e.g., 'MyCustomLayer': MyCustomLayer,
        # 'my_custom_function': my_custom_function
    }

    # Load model WITH custom objects scope AND safe_mode=False to handle Lambda layers
    try:
        # Use tf.keras.models.load_model, not keras.models.load_model
        with utils.custom_object_scope(custom_objects):
            # Add safe_mode=False to handle the Lambda layer
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        st.success("✅ Model loaded successfully with updated custom NotEqual layer and safe_mode=False!")
    except Exception as e:
        st.error(f"Failed to load model even with updated custom object scope and safe_mode=False: {str(e)}")
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
    # The success message for model loading is now inside the load_resources function
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
