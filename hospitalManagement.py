import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime

# 1. Setup Database
conn = sqlite3.connect('clinic_data.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS patients 
             (pid TEXT, name TEXT, phone TEXT, visit_date TEXT, illness TEXT, report_path TEXT)''')
conn.commit()

# 2. Helper function to generate Patient ID
def generate_id():
    now = datetime.now()
    return f"PAT-{now.strftime('%Y%m%d%H%M%S')}"

st.title("🏥 Clinic Patient Management System")

# 3. Data Entry Form
with st.form("patient_form", clear_on_submit=True):
    st.header("Register New Visit")
    name = st.text_input("Patient Name")
    phone = st.text_input("Phone Number")
    visit_date = st.date_input("Date of Visit")
    illness = st.text_area("Illness Details / Symptoms")
    
    # File Upload for Reports
    uploaded_file = st.file_uploader("Upload Previous Reports (PDF/JPG)", type=['pdf', 'jpg', 'png'])
    
    submit = st.form_submit_button("Generate Patient ID & Save")

if submit:
    patient_id = generate_id()
    file_name = ""
    
    # Save file if uploaded
    if uploaded_file:
        if not os.path.exists("reports"): os.makedirs("reports")
        file_name = f"reports/{patient_id}_{uploaded_file.name}"
        with open(file_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Save to Database
    c.execute("INSERT INTO patients VALUES (?,?,?,?,?,?)", 
              (patient_id, name, phone, str(visit_date), illness, file_name))
    conn.commit()
    
    st.success(f"Patient Registered! ID: {patient_id}")

# 4. Search/View Previous Details
st.divider()
st.header("🔍 Search Patient Records")
search_id = st.text_input("Enter Patient ID to view history")
if search_id:
    data = pd.read_sql(f"SELECT * FROM patients WHERE pid='{search_id}'", conn)
    if not data.empty:
        st.write(data)
        if data['report_path'][0]:
            st.download_button("Download Last Report", data['report_path'][0])
    else:
        st.error("Patient ID not found.")
