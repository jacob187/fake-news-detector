import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import sys

sys.path.append("../")
from utils.probability_calculations import sigmoid
from utils.clean_text import clean_text
from utils.streamlit_helpers import read_markdown_file


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🗞️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def cached_model():
    model = load_model("../models/builds/fake-news_distilbert_model.keras")
    return model


def predict_text(text: str, model):
    cleaned_text = clean_text(text)
    df = pd.DataFrame([cleaned_text], columns=["text"])
    st.session_state.user_df = df
    predictions = model.predict(df["text"])

    return predictions


def predict_class(predictions):
    return "fake" if np.argmax(predictions) == 0 else "real"


def display_home():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(
            """
            <div style="display: grid; place-items: center; height: 20vh;">
                <h1 style="color: #262730; font-size:2.5em">Fake News Detector</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:

        st.write(
            """
            <div style="display: grid; place-items: center; height: 20vh;">
                <i style="color: #6A73FF; font-weight: bold; letter-spacing: 1px;">Empower the Truth, Defeat Deception</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

    intro_text = read_markdown_file("./content/intro.md")
    st.markdown(intro_text)


def input_data():
    st.markdown("""### Enter Text Below:""")
    user_input = st.text_area(" ", height=400)
    submit_button = st.button("Submit")

    # Display the result from the AI model
    if submit_button:
        with st.spinner("Processing..."):
            model = cached_model()

        predictions = predict_text(user_input, model)
        class_confidence = sigmoid(predictions)
        result_confidence = np.max(class_confidence)
        result = predict_class(predictions)
        st.session_state.result = result
        st.session_state.result_confidence = result_confidence


def display_data():
    # Check if 'result' and 'result_confidence' exist in the session state due
    if (
        "result" in st.session_state
        and "result_confidence" in st.session_state
        and "user_df" in st.session_state
    ):
        result = st.session_state.result
        confidence_score = st.session_state.result_confidence

        tab1, tab2 = st.tabs(["Results", "Visualizations"])
        with tab1:
            st.write(f"Result: This article is {result}")
            st.write(f"Confidence score: {confidence_score:.2%}")
    else:
        pass


def main():
    display_home()
    input_data()
    display_data()


if __name__ == "__main__":
    main()
