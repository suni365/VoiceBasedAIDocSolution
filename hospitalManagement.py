import streamlit as st
import sqlite3
import pandas as pd
import os
import urllib.parse
import io
from datetime import datetime

# --- 1. DATABASE & DIRECTORY SETUP ---
UPLOAD_DIR = "patient_reports"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

conn = sqlite3.connect('clinic_v3.db', check_same_thread=False)
cursor = conn.cursor()

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
    remarks TEXT
)
''')
conn.commit()

# --- 2. HELPER FUNCTIONS ---

def export_monthly_report(month_year):
    query = "SELECT * FROM patients WHERE visit_date LIKE ?"
    df = pd.read_sql(query, conn, params=(f'{month_year}%',))
    if df.empty: return None
    df['Total_Paid'] = df['fees'] + df['testing_fees'] + df['medicine_fees']
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Monthly_Report')
    return output.getvalue()

def send_whatsapp_msg(phone, name, patient_id, total_fees):
    if not phone.startswith('91'): phone = f"91{phone}"
    message = (
        f"*🏥 S.P.SAMPATH M.B.B.S Visit Summary*\n\n"
        f"Hi *{name}*,\n"
        f"🆔 *Patient ID:* {patient_id}\n"
        f"💰 *Amount Paid:* ₹{total_fees}\n\n"
        f"Get well soon!"
    )
    encoded_msg = urllib.parse.quote(message)
    return f"https://wa.me/{phone}?text={encoded_msg}"

def get_daily_stats():
    today = str(datetime.now().date())
    cursor.execute("SELECT COUNT(*) FROM patients WHERE visit_date = ?", (today,))
    p_count = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(fees + testing_fees + medicine_fees) FROM patients WHERE visit_date = ?", (today,))
    total_rev = cursor.fetchone()[0] or 0.0
    return p_count, total_rev

# --- 3. UI MODULES ---

def show_dashboard():
    st.markdown("<h1>🏥 Clinic Dashboard</h1>", unsafe_allow_html=True)
    p_count, total_rev = get_daily_stats()
    col1, col2 = st.columns(2)
    col1.metric("Patients Today", p_count)
    col2.metric("Revenue Today", f"₹ {total_rev:,.2f}")
    st.divider()
    st.subheader("📅 Recent Activity")
    recent_df = pd.read_sql("SELECT pid, name, visit_date FROM patients ORDER BY pid DESC LIMIT 5", conn)
    st.table(recent_df)

def register_patient():
    st.markdown("<h2>📝 Front-Desk: New Registration</h2>", unsafe_allow_html=True)
    with st.form("reg_form", clear_on_submit=True):
        name = st.text_input("Patient Full Name")
        phone = st.text_input("Mobile Number")
        visit_date = st.date_input("Visit Date", value=datetime.now())
        fees = st.number_input("Consultation Fees (₹)", min_value=0.0, value=500.0)
        submit = st.form_submit_button("Register Patient")

        if submit:
            if not name or not phone:
                st.error("Missing Details")
            else:
                # SIMPLE ID: Using last 6 digits of timestamp
                p_id = datetime.now().strftime('%H%M%S') 
                try:
                    cursor.execute("INSERT INTO patients (pid, name, phone, visit_date, fees) VALUES (?,?,?,?,?)", 
                                   (p_id, name, phone, str(visit_date), fees))
                    conn.commit()
                    st.success(f"Registered! Short ID: {p_id}")
                except Exception as e: st.error(f"Error: {e}")

def dr_consultation():
    st.markdown("<h2>👨‍⚕️ Doctor's Consultation</h2>", unsafe_allow_html=True)
    search_q = st.text_input("🔍 Search Patient by ID or Name to start treatment")
    
    if search_q:
        cursor.execute("SELECT * FROM patients WHERE pid = ? OR name LIKE ?", (search_q, f'%{search_q}%'))
        patient = cursor.fetchone()
        
        if patient:
            st.info(f"Selected: {patient[1]} (ID: {patient[0]})")
            
            with st.form("treatment_form"):
                st.subheader("Medical Entry")
                illness = st.text_area("Diagnosis / Illness", value=patient[4] if patient[4] else "")
                testing = st.text_input("Tests Required", value=patient[7] if patient[7] else "")
                t_fees = st.number_input("Testing Fees (₹)", value=patient[8] if patient[8] else 0.0)
                m_fees = st.number_input("Medicine Fees (₹)", value=patient[9] if patient[9] else 0.0)
                remarks = st.text_area("Doctor's Final Description", value=patient[10] if patient[10] else "")
                
                uploaded_file = st.file_uploader("Update Medical Report (Optional)")
                
                if st.form_submit_button("✅ Update Treatment & Save"):
                    path = patient[5]
                    if uploaded_file:
                        path = os.path.join(UPLOAD_DIR, f"{patient[0]}_{uploaded_file.name}")
                        with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
                    
                    cursor.execute("""
                        UPDATE patients SET illness=?, report_path=?, testing_done=?, 
                        testing_fees=?, medicine_fees=?, remarks=? WHERE pid=?
                    """, (illness, path, testing, t_fees, m_fees, remarks, patient[0]))
                    conn.commit()
                    
                    st.success("Patient Record Updated Successfully!")
                    total = patient[6] + t_fees + m_fees
                    st.link_button("📲 Send Updated Summary to WhatsApp", send_whatsapp_msg(patient[2], patient[1], patient[0], total))
        else:
            st.error("Patient not found.")

def search_records():
    st.markdown("<h2>🔍 Master Records</h2>", unsafe_allow_html=True)
    query = st.text_input("Search ID, Name, or Phone")
    if query:
        df = pd.read_sql("SELECT * FROM patients WHERE pid LIKE ? OR name LIKE ?", conn, params=(f'%{query}%', f'%{query}%'))
        for _, row in df.iterrows():
            with st.expander(f"ID: {row['pid']} | {row['name']}"):
                st.write(f"**Diagnosis:** {row['illness']}")
                st.write(f"**Medicines/Tests:** {row['testing_done']}")
                st.write(f"**Remarks:** {row['remarks']}")
                total = row['fees'] + row['testing_fees'] + row['medicine_fees']
                st.write(f"**Total Paid:** ₹{total}")

# --- 4. APP FLOW ---

if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🏥 Clinic Login")
    u, p = st.text_input("User"), st.text_input("Pass", type="password")
    if st.button("Login"):
        if u == "admin" and p == "clinic123":
            st.session_state['logged_in'] = True
            st.rerun()
else:
    menu = st.sidebar.radio("Go To:", ["Dashboard", "New Registration", "Doctor Consultation", "Search Records", "Admin Reports"])
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
    
    if menu == "Dashboard": show_dashboard()
    elif menu == "New Registration": register_patient()
    elif menu == "Doctor Consultation": dr_consultation()
    elif menu == "Search Records": search_records()
    elif menu == "Admin Reports":
        st.header("📊 Admin Reports")
        m = st.date_input("Month").strftime('%Y-%m')
        if st.button("Generate Excel"):
            data = export_monthly_report(m)
            if data: st.download_button("📥 Download", data, f"Report_{m}.xlsx")
            else: st.warning("No data.")
