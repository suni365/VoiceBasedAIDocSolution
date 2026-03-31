import streamlit as st
import sqlite3
import pandas as pd
import os
import urllib.parse
import base64
from datetime import datetime

# --- DATABASE & DIRECTORY SETUP ---
UPLOAD_DIR = "patient_reports"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

conn = sqlite3.connect('clinic_v5.db', check_same_thread=False)
cursor = conn.cursor()

# Schema for the new workflow
cursor.execute('''
CREATE TABLE IF NOT EXISTS patients (
    pid INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    phone TEXT,
    visit_date TEXT,
    basic_symptoms TEXT,
    illness_description TEXT,
    test_recommendations TEXT,
    test_results TEXT,
    test_breakdown TEXT,
    test_fees REAL,
    report_path TEXT,
    med_prescription TEXT,
    med_breakdown TEXT,
    med_fees REAL,
    consultation_fees REAL DEFAULT 500.0
)
''')
conn.commit()

# --- HELPER FUNCTIONS ---

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def send_wa_reg(phone, name, pid):
    if not phone.startswith('91'): phone = f"91{phone}"
    msg = f"*🏥 Clinic Registration*\n\nHi *{name}*,\nYou are registered. \n🆔 *ID:* {pid}\nPlease wait for the Doctor."
    return f"https://wa.me/{phone}?text={urllib.parse.quote(msg)}"

# --- UI MODULES ---

def registration_module():
    st.header("📝 Front Desk Registration")
    with st.form("reg_form"):
        name = st.text_input("Patient Name")
        phone = st.text_input("Phone Number")
        symptoms = st.text_area("Initial Symptoms (e.g. Fever, Cough)")
        c_fees = st.number_input("Consultation Fee (₹)", value=500.0)
        
        if st.form_submit_button("Register Patient"):
            if name and phone:
                cursor.execute("INSERT INTO patients (name, phone, visit_date, basic_symptoms, consultation_fees) VALUES (?,?,?,?,?)",
                               (name, phone, str(datetime.now().date()), symptoms, c_fees))
                conn.commit()
                new_id = cursor.lastrowid
                st.success(f"Registered! Patient ID: {new_id}")
                st.link_button("📲 Send WhatsApp ID", send_wa_reg(phone, name, new_id))
            else: st.error("Please fill Name and Phone.")

def doctor_module():
    st.header("👨‍⚕️ Doctor's Consultation")
    p_id = st.number_input("Enter Patient ID", min_value=1, step=1)
    if p_id:
        res = cursor.execute("SELECT name, basic_symptoms, report_path FROM patients WHERE pid=?", (p_id,)).fetchone()
        if res:
            st.subheader(f"Patient: {res[0]}")
            st.warning(f"**Reported Symptoms:** {res[1]}")
            
            # View existing reports if any
            if res[2] and os.path.exists(res[2]):
                if st.checkbox("👁️ View Existing Lab Report"):
                    display_pdf(res[2])
            
            with st.form("doc_notes"):
                descr = st.text_area("Full Illness Description")
                tests = st.text_input("Tests Recommended")
                meds = st.text_area("Medicine Prescription")
                if st.form_submit_button("Submit Clinical Notes"):
                    cursor.execute("UPDATE patients SET illness_description=?, test_recommendations=?, med_prescription=? WHERE pid=?",
                                   (descr, tests, meds, p_id))
                    conn.commit()
                    st.success("Doctor's notes saved.")
        else: st.error("ID not found.")

def lab_module():
    st.header("🔬 Laboratory")
    p_id = st.number_input("Lab: Enter Patient ID", min_value=1, step=1)
    if p_id:
        res = cursor.execute("SELECT name, test_recommendations FROM patients WHERE pid=?", (p_id,)).fetchone()
        if res:
            st.info(f"Patient: {res[0]} | **Tests Requested:** {res[1]}")
            results = st.text_area("Test Results")
            breakdown = st.text_area("Rate Breakdown (e.g. Blood: 200, Sugar: 100)")
            total_lab = st.number_input("Total Lab Amount (₹)", min_value=0.0)
            file = st.file_uploader("Upload PDF Report")
            
            if st.button("Submit Lab Data"):
                path = ""
                if file:
                    path = os.path.join(UPLOAD_DIR, f"LAB_{p_id}_{file.name}")
                    with open(path, "wb") as f: f.write(file.getbuffer())
                cursor.execute("UPDATE patients SET test_results=?, test_breakdown=?, test_fees=?, report_path=? WHERE pid=?",
                               (results, breakdown, total_lab, path, p_id))
                conn.commit()
                st.success("Lab results updated.")

def pharmacy_module():
    st.header("💊 Pharmacy")
    p_id = st.number_input("Pharma: Enter Patient ID", min_value=1, step=1)
    if p_id:
        res = cursor.execute("SELECT name, med_prescription FROM patients WHERE pid=?", (p_id,)).fetchone()
        if res:
            st.info(f"Patient: {res[0]} | **Prescription:** {res[1]}")
            breakdown = st.text_area("Medicine Details & Rates")
            total_med = st.number_input("Total Pharmacy Amount (₹)", min_value=0.0)
            if st.button("Finalize Pharmacy Bill"):
                cursor.execute("UPDATE patients SET med_breakdown=?, med_fees=? WHERE pid=?", (breakdown, total_med, p_id))
                conn.commit()
                st.success("Pharmacy charges added.")

def billing_search():
    st.header("🔍 Search & Final Bill")
    p_id = st.number_input("Search ID", min_value=1, step=1)
    if p_id:
        r = pd.read_sql("SELECT * FROM patients WHERE pid=?", conn, params=(p_id,))
        if not r.empty:
            p = r.iloc[0]
            st.write(f"### {p['name']} (Visit: {p['visit_date']})")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Diagnosis:** {p['illness_description']}")
                st.write(f"**Lab Results:** {p['test_results']}")
                if p['report_path']: st.button("View Report", on_click=display_pdf, args=(p['report_path'],))
            with c2:
                total = (p['consultation_fees'] or 0) + (p['test_fees'] or 0) + (p['med_fees'] or 0)
                st.metric("Total Payable", f"₹{total}")
                st.write(f"Breakdown: Cons(₹{p['consultation_fees']}) + Lab(₹{p['test_fees']}) + Meds(₹{p['med_fees']})")
        else: st.error("No record.")

# --- APP FLOW ---

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🏥 Clinic Management")
    if st.text_input("User") == "admin" and st.text_input("Pass", type="password") == "clinic123":
        if st.button("Login"):
            st.session_state['logged_in'] = True
            st.rerun()
else:
    menu = st.sidebar.radio("Navigation", ["Registration", "Doctor", "Lab", "Pharmacy", "Search/Billing"])
    if menu == "Registration": registration_module()
    elif menu == "Doctor": doctor_module()
    elif menu == "Lab": lab_module()
    elif menu == "Pharmacy": pharmacy_module()
    elif menu == "Search/Billing": billing_search()
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
