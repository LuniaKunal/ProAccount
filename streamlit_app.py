# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

st.title("Bill Data Extractor")

uploaded_file = st.file_uploader("Upload a PDF bill", type=['pdf'])

if uploaded_file is not None:
    st.write("Processing PDF...")
    files = {'pdf_file': uploaded_file}
    try:
        response = requests.post("http://127.0.0.1:8000/extract_bill_data", files=files) # Flask backend URL
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        extracted_data = response.json()
        print(response.json())
        if 'error' in extracted_data:
            st.error(f"Error: {extracted_data['error']}")
            if 'raw_output' in extracted_data:
                st.write("Raw Gemini Output (for debugging):")
                st.code(extracted_data['raw_output']) # Display raw output if available
        else:
            st.success("Data extracted successfully!")
            st.write("Extracted Details:")
            output_file = extracted_data['output_file']
            df = pd.read_csv(f"D:/Automation/backend/{output_file}") # Read the CSV file into a DataFrame
            st.dataframe(df)

    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")