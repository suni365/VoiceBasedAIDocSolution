import streamlit as st
import sqlite3
import pandas as pd
import os
import urllib.parse
import base64
from datetime import datetime, timedelta

# --- DATABASE & DIRECTORY SETUP ---
UPLOAD_DIR = "patient_reports"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Using v3 database for consistency
conn = sqlite3.connect('clinic_v3.db', check_same_thread=False)
cursor = conn.cursor()

# Comprehensive Schema
cursor.execute('''
CREATE TABLE IF NOT EXISTS patients (
    pid TEXT PRIMARY KEY,
    name TEXT,
    phone TEXT,
    visit_date TEXT,
    illness TEXT,
    report_path TEXT,
    fees REAL,
    testing_done TEXT,
    testing_fees REAL,
    medicine_fees REAL,
    remarks TEXT,
    medicine_details TEXT,
    testing_results TEXT
)
''')
conn.commit()

# --- HELPER FUNCTIONS ---

def display_pdf(file_path):
    """Embeds PDF in Streamlit."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def send_whatsapp_msg(phone, name, patient_id, total_fees):
    if not phone.startswith('91'): phone = f"91{phone}"
    message = (f"*🏥 Clinic Visit Summary*\n\nHi *{name}*,\n🆔 *ID:* {patient_id}\n💰 *Total:* ₹{total_fees}\n\nGet well soon!")
    return f"https://wa.me/{phone}?text={urllib.parse.quote(message)}"

# --- UI MODULES ---

def show_dashboard():
    st.markdown("<h1>🏥 Clinic Dashboard</h1>", unsafe_allow_html=True)
    today = str(datetime.now().date())
    
    # Stats
    p_today = pd.read_sql("SELECT COUNT(*) as count FROM patients WHERE visit_date = ?", conn, params=(today,)).iloc[0]['count']
    rev_today = pd.read_sql("SELECT SUM(fees + testing_fees + medicine_fees) as rev FROM patients WHERE visit_date = ?", conn, params=(today,)).iloc[0]['rev'] or 0.0
    
    c1, c2 = st.columns(2)
    c1.metric("Patients Today", p_today)
    c2.metric("Revenue Today", f"₹{rev_today:,.2f}")
    
    st.divider()
    st.subheader("📅 Recent Activity")
    recent = pd.read_sql("SELECT name, visit_date, (fees + testing_fees + medicine_fees) as Total FROM patients ORDER BY pid DESC LIMIT 5", conn)
    st.table(recent)

def doctor_consultation():
    st.markdown("<h2>👨‍⚕️ Doctor Consultation / Registration</h2>", unsafe_allow_html=True)
    with st.form("doc_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            phone = st.text_input("Mobile Number")
            visit_date = st.date_input("Visit Date", value=datetime.now())
            illness = st.text_area("Diagnosis / Illness Details")
        with col2:
            fees = st.number_input("Consultation Fees (₹)", value=500.0)
            tests = st.text_input("Recommended Tests")
            meds = st.text_area("Prescribed Medicines")
        remarks = st.text_area("Final Remarks")
        
        if st.form_submit_button("💾 Save Record"):
            if name and phone:
                p_id = f"PAT-{datetime.now().strftime('%y%m%d%H%M%S')}"
                cursor.execute("INSERT INTO patients (pid, name, phone, visit_date, illness, fees, testing_done, medicine_details, remarks) VALUES (?,?,?,?,?,?,?,?,?)",
                               (p_id, name, phone, str(visit_date), illness, fees, tests, meds, remarks))
                conn.commit()
                st.success(f"Patient Registered! ID: {p_id}")
            else: st.error("Name and Phone required.")

def testing_lab():
    st.markdown("<h2>🔬 Laboratory Unit</h2>", unsafe_allow_html=True)
    p_id = st.text_input("Enter Patient ID for Lab Update")
    if p_id:
        row = cursor.execute("SELECT name, testing_done FROM patients WHERE pid = ?", (p_id,)).fetchone()
        if row:
            st.info(f"Patient: {row[0]} | Tests: {row[1]}")
            res = st.text_area("Testing Results")
            t_fees = st.number_input("Testing Fees (₹)", min_value=0.0)
            file = st.file_uploader("Upload Report (PDF)")
            if st.button("Update Lab Record"):
                path = ""
                if file:
                    path = os.path.join(UPLOAD_DIR, f"{p_id}_{file.name}")
                    with open(path, "wb") as f: f.write(file.getbuffer())
                cursor.execute("UPDATE patients SET testing_results=?, testing_fees=?, report_path=? WHERE pid=?", (res, t_fees, path, p_id))
                conn.commit()
                st.success("Lab record updated!")
        else: st.error("Not found.")

def pharmacy_unit():
    st.markdown("<h2>💊 Pharmacy Unit</h2>", unsafe_allow_html=True)
    p_id = st.text_input("Enter Patient ID for Pharmacy")
    if p_id:
        row = cursor.execute("SELECT name, medicine_details FROM patients WHERE pid = ?", (p_id,)).fetchone()
        if row:
            st.info(f"Patient: {row[0]} | Prescribed: {row[1]}")
            m_fees = st.number_input("Medicine Fees (₹)", min_value=0.0)
            if st.button("Update Pharmacy Record"):
                cursor.execute("UPDATE patients SET medicine_fees=? WHERE pid=?", (m_fees, p_id))
                conn.commit()
                st.success("Pharmacy charges updated!")

def search_patient():
    st.markdown("<h2>🔍 Search & Final Bill</h2>", unsafe_allow_html=True)
    query = st.text_input("Search Name/Phone/ID")
    if query:
        df = pd.read_sql("SELECT * FROM patients WHERE name LIKE ? OR phone LIKE ? OR pid LIKE ?", conn, params=(f'%{query}%',f'%{query}%',f'%{query}%'))
        for _, r in df.iterrows():
            with st.expander(f"👤 {r['name']} - {r['visit_date']}"):
                c1, c2 = st.columns(2)
                c1.write(f"**ID:** {r['pid']}\n\n**Diagnosis:** {r['illness']}\n\n**Meds:** {r['medicine_details']}")
                total = (r['fees'] or 0) + (r['testing_fees'] or 0) + (r['medicine_fees'] or 0)
                c2.metric("Total Bill", f"₹{total}")
                c2.write(f"Cons: ₹{r['fees']} | Lab: ₹{r['testing_fees']} | Meds: ₹{r['medicine_fees']}")
                
                if r['report_path'] and os.path.exists(r['report_path']):
                    col_v, col_d = st.columns(2)
                    with col_v: 
                        if st.button("👁️ View", key=f"v{r['pid']}"): display_pdf(r['report_path'])
                    with col_d: 
                        st.download_button("📥 Download", open(r['report_path'], 'rb'), file_name=os.path.basename(r['report_path']), key=f"d{r['pid']}")
                
                st.link_button("📲 WhatsApp Summary", send_whatsapp_msg(r['phone'], r['name'], r['pid'], total))

def monthly_reports():
    st.markdown("<h2>📊 Monthly Revenue Report</h2>", unsafe_allow_html=True)
    month = st.selectbox("Select Month", range(1, 13), index=datetime.now().month-1)
    year = st.number_input("Year", value=datetime.now().year)
    
    start_date = f"{year}-{month:02d}-01"
    df = pd.read_sql("SELECT * FROM patients WHERE visit_date LIKE ?", conn, params=(f'{year}-{month:02d}%',))
    
    if not df.empty:
        st.dataframe(df[['pid', 'name', 'visit_date', 'fees', 'testing_fees', 'medicine_fees']])
        total_m = (df['fees'].sum() + df['testing_fees'].sum() + df['medicine_fees'].sum())
        st.subheader(f"Total Revenue for {month}/{year}: ₹{total_m:,.2f}")
    else: st.info("No records for this month.")

# --- APP FLOW ---

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🏥 Clinic Login")
    if st.text_input("User") == "admin" and st.text_input("Pass", type="password") == "clinic123":
        if st.button("Login"):
            st.session_state['logged_in'] = True
            st.rerun()
else:
    st.sidebar.title("Clinic Management")
    menu = st.sidebar.radio("Navigation", ["Dashboard", "Dr. Consultation", "Testing/Lab", "Pharmacy", "Search Patient", "Monthly Report"])
    
    if menu == "Dashboard": show_dashboard()
    elif menu == "Dr. Consultation": doctor_consultation()
    elif menu == "Testing/Lab": testing_lab()
    elif menu == "Pharmacy": pharmacy_unit()
    elif menu == "Search Patient": search_patient()
    elif menu == "Monthly Report": monthly_reports()

    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
