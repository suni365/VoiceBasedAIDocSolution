import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime

# 1. Configuration & Folders
UPLOAD_DIR = "patient_reports"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# 2. Database Connection
conn = sqlite3.connect('clinic_data.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS patients 
                 (pid TEXT PRIMARY KEY, name TEXT, phone TEXT, 
                  visit_date TEXT, illness TEXT, report_path TEXT)''')
conn.commit()

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #f0f8ff; }
    .main-header { color: #004d4d; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

def login():
    st.markdown("<h1 class='main-header'>Clinic Management Login</h1>", unsafe_allow_html=True)
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and password == "clinic123":
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

def register_patient():
    st.markdown("### 📝 New Patient Registration")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Personal Information")
        name = st.text_input("Full Name")
        phone = st.text_input("Mobile Number")
        visit_date = st.date_input("Date of Visit", value=datetime.now())
        illness = st.text_area("Symptoms / Reason for Visit", height=150)

    with col2:
        st.subheader("Medical Records")
        uploaded_file = st.file_uploader("Upload Past Reports", type=['pdf', 'png', 'jpg'])
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name} ready.")

    st.divider()
    if st.button("🔥 Generate Patient ID & Save Record", use_container_width=True):
        if name and phone:
            # Generate Unique ID
            patient_id = f"PAT-{datetime.now().strftime('%Y%m%d%H%M')}"
            
            # HANDLE FILE SAVING
            saved_path = ""
            if uploaded_file:
                # Rename file to avoid duplicates: PID_Filename.ext
                file_name = f"{patient_id}_{uploaded_file.name}"
                saved_path = os.path.join(UPLOAD_DIR, file_name)
                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # SAVE TO DATABASE
            try:
                cursor.execute('''INSERT INTO patients (pid, name, phone, visit_date, illness, report_path) 
                                  VALUES (?, ?, ?, ?, ?, ?)''', 
                               (patient_id, name, phone, str(visit_date), illness, saved_path))
                conn.commit()
                st.balloons()
                st.success(f"SUCCESS! Patient ID: {patient_id}")
            except Exception as e:
                st.error(f"Database Error: {e}")
        else:
            st.warning("Please fill in Name and Phone Number.")

def main_app():
    st.sidebar.title("🩺 Navigation")
    choice = st.sidebar.selectbox("Go to", ["Register Patient", "Search Records"])

    if choice == "Register Patient":
        register_patient()
    elif choice == "Search Records":
        st.header("🔍 Patient Search")
        search_term = st.text_input("Enter Name or Phone")
        if search_term:
            query = "SELECT * FROM patients WHERE name LIKE ? OR phone LIKE ?"
            df = pd.read_sql(query, conn, params=(f'%{search_term}%', f'%{search_term}%'))
            if not df.empty:
                st.table(df) # table looks cleaner for medical lists
            else:
                st.info("No records found.")

# App Logic
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
else:
    main_app()
