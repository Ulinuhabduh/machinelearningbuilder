import streamlit as st

st.set_page_config(
    page_title="Machine Learning Builder",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.success("Select a feature above.")

st.sidebar.title("About")
st.sidebar.info("""
    **Machine Learning Builder** helps you build, evaluate, and use machine learning models for your projects.
    Navigate through the app using the options above and explore various features and tools.
""")

st.sidebar.title("Contact")
st.sidebar.info("""
    **Contact Us:**
    - Email: mulinuhaa@gmail.com
""")

st.title("👋 Welcome to ML Model Builder and Predictor")
st.markdown("*Author*: **M Ulin Nuha Abduh**")
st.write("""
    This app allows you to build and evaluate machine learning models, as well as make predictions using the trained models.
    Use the sidebar to navigate between pages and explore the features of this app.
""")

st.write("---")
st.write("### Why Use This App?")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://img.icons8.com/?size=100&id=103683&format=png&color=000000", width=100)
    st.subheader("User-Friendly Interface")
    st.write("Easily navigate and interact with the application using our intuitive interface.")

with col2:
    st.image("https://img.icons8.com/?size=100&id=T1KLkEPyHlcT&format=png&color=000000", width=100)
    st.subheader("Comprehensive Analysis")
    st.write("Perform extensive data analysis and visualization with integrated tools.")

with col3:
    st.image("https://img.icons8.com/?size=100&id=YNaxZEobUCzH&format=png&color=000000", width=100)
    st.subheader("Cutting-Edge Models")
    st.write("Utilize state-of-the-art machine learning models for accurate predictions.")

st.write("---")
st.write("### Key Features")
st.markdown("""
- **Data Upload**: Easily upload your dataset in CSV format.
- **Data Preprocessing**: Clean and preprocess your data with options for handling missing values, scaling, and encoding.
- **Model Training**: Choose from a variety of machine learning models and tune hyperparameters for optimal performance.
- **Model Evaluation**: Assess your model's performance using various metrics and visualizations.
- **Predictions**: Make predictions on new data with your trained model.
- **Download Options**: Save and download your processed data and trained models.
""")

st.write("---")
st.write("### Get Started")
st.write("Ready to start building your machine learning model? Use the sidebar to select a feature and begin your journey!")
st.button("Let's Go!", help="Click here to navigate to the feature selection")

st.write("---")
st.write("© 2024 M Ulin Nuha Abduh. All rights reserved.")
