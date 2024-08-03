import pandas as pd
import streamlit as st
import sweetviz as sv
import dtale
from ydata_profiling import ProfileReport

st.title("Profiling Data")

st.sidebar.title("Contact")
st.sidebar.info(
    """
    **Contact Me:**
    - Email: mulinuhaa@gmail.com
"""
)

uploaded_file = st.file_uploader("Choose a CSV file for profiling", type="csv")

if uploaded_file is not None:
    try:
        sep = st.selectbox("Select separator", ['comma', 'semicolon', 'tab', 'spasi'])
        if sep == 'comma':
            sep = ','
            df = pd.read_csv(uploaded_file, sep=sep, on_bad_lines='skip')  
        elif sep == 'semicolon':
            sep = ';'
            df = pd.read_csv(uploaded_file, sep=sep, on_bad_lines='skip') 
        elif sep == 'tab':
            sep = '\t'
            df = pd.read_csv(uploaded_file, sep=sep, on_bad_lines='skip')  
        else:
            sep = ' '
            df = pd.read_csv(uploaded_file, sep=sep, on_bad_lines='skip') 
        
        st.write("Preview of the dataset:")
        st.write(df.head())
    except pd.errors.ParserError as e:
        st.error(f"Error parsing the CSV file: {e}. Please check the file format.")

    choose_another = st.selectbox("Choose Methods", ["", "Sweetviz", "YData-Profiling"])

    if choose_another == 'Sweetviz':
        report = sv.analyze(df)
        report_file_path = 'Sweetviz_Report.html'
        report.show_html(filepath=report_file_path, open_browser=False, layout='vertical')
        
        st.success("Sweetviz report generated successfully.")
        
        with open(report_file_path, 'rb') as file:
            btn = st.download_button(
                label="Download Sweetviz Report",
                data=file,
                file_name=report_file_path,
                mime="text/html"
            )

    elif choose_another == 'YData-Profiling':
        profile = ProfileReport(
            df,
            title='YData Profiling Report',
            explorative=True,
            variables={
                "descriptions": {
                    "text": True
                }
            }
        )
        report_file_path = 'YData_Profiling_Report.html'
        profile.to_file(report_file_path)
        
        st.success("YData Profiling report generated successfully.")
        
        with open(report_file_path, 'rb') as file:
            btn = st.download_button(
                label="Download YData-Profiling Report",
                data=file,
                file_name=report_file_path,
                mime="text/html"
            )
