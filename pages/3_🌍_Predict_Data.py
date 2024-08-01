import streamlit as st
import pandas as pd
import pickle

def predict_data():
    st.title("Predict Data With Model")

    model_file = st.file_uploader("Upload your trained model (.pkl)", type="pkl")
    
    if model_file is not None:
        model = pickle.load(model_file)
        st.success("Model loaded successfully!")

        new_data_file = st.file_uploader("Upload new data for prediction (.csv)", type="csv")
        
        if new_data_file is not None:
            new_data = pd.read_csv(new_data_file)
            st.write("Preview of new data:")
            st.write(new_data.head())

            feature_cols = st.multiselect("Select features for prediction", new_data.columns)

            if len(feature_cols) > 0:
                X_new = new_data[feature_cols]

                if st.button("Predict"):
                    predictions = model.predict(X_new)
                    new_data['Prediction'] = predictions
                    st.write("Prediction Results:")
                    st.write(new_data)

                    csv = new_data.to_csv(index=False)
                    st.download_button(
                        label="Download predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

if __name__ == "__main__":
    predict_data()