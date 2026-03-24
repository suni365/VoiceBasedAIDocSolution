import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime

# --- 1. CONFIGURATION & DATABASE SETUP ---
UPLOAD_DIR = "patient_reports"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

conn = sqlite3.connect('clinic_data.db', check_same_thread=False)
cursor = conn.cursor()
# Added testing_fees to track lab revenue
cursor.execute('''CREATE TABLE IF NOT EXISTS patients 
                 (pid TEXT PRIMARY KEY, name TEXT, phone TEXT, 
                  visit_date TEXT, illness TEXT, report_path TEXT, 
                  fees REAL, testing_done TEXT, testing_fees REAL)''')
conn.commit()

# --- 2. LIVE DATA CALCULATIONS ---
def get_daily_stats():
    today = str(datetime.now().date())
    
    # 1. Count Patients
    cursor.execute("SELECT COUNT(*) FROM patients WHERE visit_date = ?", (today,))
    count = cursor.fetchone()[0]
    
    # 2. Sum Consultation Fees
    cursor.execute("SELECT SUM(fees) FROM patients WHERE visit_date = ?", (today,))
    fees_sum = cursor.fetchone()[0] or 0
    
    # 3. Sum Testing/Medicine Revenue
    cursor.execute("SELECT SUM(testing_fees) FROM patients WHERE visit_date = ?", (today,))
    test_sum = cursor.fetchone()[0] or 0
    
    return count, fees_sum, test_sum

# --- 3. UI MODULES ---

def show_dashboard():
    st.markdown("<h1 class='main-header'>🏥 Clinic Dashboard</h1>", unsafe_allow_html=True)
    
    # PULL LIVE DATA
    p_count, f_total, t_total = get_daily_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Patients Today", f"{p_count}")
    with col2:
        st.metric("Consultation Fees", f"₹ {f_total:,.2f}")
    with col3:
        st.metric("Lab/Testing Revenue", f"₹ {t_total:,.2f}")
    with col4:
        st.metric("Total Revenue", f"₹ {f_total + t_total:,.2f}")

    st.divider()
    
    # Doctors & Appointments (Still Dummy for now)
    l, r = st.columns(2)
    with l:
        st.subheader("👨‍⚕️ Doctors Available")
        st.table(pd.DataFrame({"Doctor": ["Dr. Arya", "Dr. Smith"], "Status": ["Active", "On Call"]}))
    with r:
        st.subheader("📅 Recent Registrations")
        recent_df = pd.read_sql("SELECT name, visit_date, fees FROM patients ORDER BY pid DESC LIMIT 5", conn)
        st.dataframe(recent_df, use_container_width=True)

def register_patient():
    st.markdown("<h2 class='main-header'>📝 Patient Registration</h2>", unsafe_allow_html=True)
    with st.form("reg_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full Name")
            phone = st.text_input("Mobile Number")
            visit_date = st.date_input("Date", value=datetime.now())
        with c2:
            fees = st.number_input("Consultation Fees (₹)", value=500)
            t_fees = st.number_input("Testing/Medicine Fees (₹)", value=0)
            testing = st.text_input("Tests/Medicine Details")
            
        illness = st.text_area("Illness Details")
        uploaded_file = st.file_uploader("Upload Report")
        
        submit = st.form_submit_button("💾 Save & Generate ID")

        if submit and name and phone:
            patient_id = f"PAT-{datetime.now().strftime('%y%m%d%H%M')}"
            path = ""
            if uploaded_file:
                path = os.path.join(UPLOAD_DIR, f"{patient_id}_{uploaded_file.name}")
                with open(path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            cursor.execute("INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?)", 
                           (patient_id, name, phone, str(visit_date), illness, path, fees, testing, t_fees))
            conn.commit()
            st.success(f"Success! ID: {patient_id}")

def search_records():
    st.markdown("<h2 class='main-header'>🔍 Patient Records</h2>", unsafe_allow_html=True)
    search = st.text_input("Search Name/Phone")
    if search:
        df = pd.read_sql("SELECT * FROM patients WHERE name LIKE ? OR phone LIKE ?", 
                         conn, params=(f'%{search}%', f'%{search}%'))
        for _, row in df.iterrows():
            with st.expander(f"👤 {row['name']} ({row['visit_date']})"):
                st.write(f"**ID:** {row['pid']} | **Fees:** ₹{row['fees'] + row['testing_fees']}")
                st.write(f"**Diagnosis:** {row['illness']}")
                if st.session_state.get('user_role') == 'admin':
                    if st.button(f"Delete {row['pid']}", key=row['pid']):
                        cursor.execute("DELETE FROM patients WHERE pid=?", (row['pid'],))
                        conn.commit()
                        st.rerun()

# --- 4. NAVIGATION ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("Clinic Login")
    u, p = st.text_input("User"), st.text_input("Pass", type="password")
    if st.button("Login"):
        if u == "admin" and p == "clinic123":
            st.session_state.update({'logged_in': True, 'user_role': 'admin'})
            st.rerun()
else:
    choice = st.sidebar.radio("Menu", ["Dashboard", "Register Patient", "Search Records"])
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
    
    if choice == "Dashboard": show_dashboard()
    elif choice == "Register Patient": register_patient()
    elif choice == "Search Records": search_records()
