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

# --- 2. HELPER FUNCTIONS (Must be defined BEFORE they are called) ---

def export_monthly_report(month_year):
    """Generates an Excel report for a specific month (Format: 'YYYY-MM')."""
    query = "SELECT * FROM patients WHERE visit_date LIKE ?"
    df = pd.read_sql(query, conn, params=(f'{month_year}%',))
    
    if df.empty:
        return None
    
    # Calculate Total per row
    df['Total_Paid'] = df['fees'] + df['testing_fees'] + df['medicine_fees']
    
    output = io.BytesIO()
    # Note: ensure 'xlsxwriter' is installed: pip install xlsxwriter
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Monthly_Report')
    
    return output.getvalue()

def send_whatsapp_msg(phone, name, patient_id, total_fees):
    if not phone.startswith('91'):
        phone = f"91{phone}"
    message = (
        f"*🏥 S.P.SAMPATH M.B.B.S Visit Summary*\n\n"
        f"Hi *{name}*,\n"
        f"Thank you for visiting us today.\n\n"
        f"🆔 *Patient ID:* {patient_id}\n"
        f"💰 *Total Amount Paid:* ₹{total_fees}\n\n"
        f"Please keep this ID for your next visit. Get well soon!"
    )
    encoded_msg = urllib.parse.quote(message)
    return f"https://wa.me/{phone}?text={encoded_msg}"

def get_daily_stats():
    today = str(datetime.now().date())
    cursor.execute("SELECT COUNT(*) FROM patients WHERE visit_date = ?", (today,))
    p_count = cursor.fetchone()[0]
    cursor.execute("SELECT SUM(fees + testing_fees + medicine_fees) FROM patients WHERE visit_date = ?", (today,))
    total_revenue = cursor.fetchone()[0] or 0.0
    return p_count, total_revenue

# --- 3. UI MODULES ---

def show_dashboard():
    st.markdown("<h1>🏥 S.P.SAMPATH M.B.B.S Clinic Dashboard</h1>", unsafe_allow_html=True)
    p_count, total_rev = get_daily_stats()
    col1, col2 = st.columns(2)
    col1.metric("Patients Today", p_count)
    col2.metric("Total Revenue Today", f"₹ {total_rev:,.2f}")
    st.divider()
    st.subheader("📅 Recent Activity")
    recent_df = pd.read_sql("SELECT name, visit_date, (fees + testing_fees + medicine_fees) as Total FROM patients ORDER BY pid DESC LIMIT 5", conn)
    st.table(recent_df)

def register_patient():
    st.markdown("<h2>📝 Patient Registration</h2>", unsafe_allow_html=True)
    with st.form("reg_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            phone = st.text_input("Mobile Number")
            visit_date = st.date_input("Visit Date", value=datetime.now())
            illness = st.text_area("Diagnosis / Illness Details")
        with col2:
            fees = st.number_input("Consultation Fees (₹)", min_value=0.0, value=500.0)
            t_fees = st.number_input("Testing Fees (₹)", min_value=0.0, value=0.0)
            m_fees = st.number_input("Medicine Fees (₹)", min_value=0.0, value=0.0)
            testing_details = st.text_input("Tests / Medicine Names")
        uploaded_file = st.file_uploader("Upload Medical Report (Optional)")
        remarks = st.text_area("Final Description / Doctor's Remarks")
        submit = st.form_submit_button("💾 Save Patient Record")
        if submit:
            if not name or not phone:
                st.error("Name and Phone Number are required!")
            else:
                p_id = f"PAT-{datetime.now().strftime('%y%m%d%H%M%S')}"
                path = ""
                if uploaded_file:
                    path = os.path.join(UPLOAD_DIR, f"{p_id}_{uploaded_file.name}")
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                try:
                    cursor.execute("INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?)", 
                                   (p_id, name, phone, str(visit_date), illness, path, fees, testing_details, t_fees, m_fees, remarks))
                    conn.commit()
                    st.success(f"Patient Registered Successfully! ID: {p_id}")
                    grand_total = fees + t_fees + m_fees
                    wa_url = send_whatsapp_msg(phone, name, p_id, grand_total)
                    st.link_button("📲 Send Summary via WhatsApp", wa_url)
                except sqlite3.Error as e:
                    st.error(f"Database Error: {e}")

def search_records():
    st.markdown("<h2>🔍 Search Records</h2>", unsafe_allow_html=True)
    query = st.text_input("Enter Name, Phone, or Patient ID")
    if query:
        df = pd.read_sql("SELECT * FROM patients WHERE name LIKE ? OR phone LIKE ? OR pid LIKE ?", 
                         conn, params=(f'%{query}%', f'%{query}%', f'%{query}%'))
        if df.empty:
            st.info("No records found.")
        else:
            for _, row in df.iterrows():
                with st.expander(f"👤 {row['name']} - {row['visit_date']}"):
                    c1, c2 = st.columns(2)
                    c1.write(f"**ID:** {row['pid']}")
                    c1.write(f"**Phone:** {row['phone']}")
                    c1.write(f"**Illness:** {row['illness']}")
                    total = row['fees'] + row['testing_fees'] + row['medicine_fees']
                    c2.write(f"**Consultation:** ₹{row['fees']}")
                    c2.write(f"**Testing:** ₹{row['testing_fees']}")
                    c2.write(f"**Medicine:** ₹{row['medicine_fees']}")
                    c2.markdown(f"### **Total: ₹{total}**")
                    st.write(f"**Remarks:** {row['remarks']}")
                    if row['report_path'] and os.path.exists(row['report_path']):
                        st.download_button("Download Report", open(row['report_path'], 'rb'), 
                                           file_name=os.path.basename(row['report_path']))

# --- 4. APP FLOW (The logic that calls the UI) ---

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🏥 Clinic Login")
    u = st.text_input("Admin Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "clinic123":
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Invalid Login")
else:
    menu = st.sidebar.radio("Navigation", ["Dashboard", "Register Patient", "Search Records", "Admin Reports"])
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
    
    if menu == "Dashboard": 
        show_dashboard()
    elif menu == "Register Patient": 
        register_patient()   
    elif menu == "Search Records": 
        search_records()
    elif menu == "Admin Reports":
        st.header("📊 Financial & Revenue Reports")
        report_month = st.date_input("Select Month", value=datetime.now())
        formatted_month = report_month.strftime('%Y-%m')
        
        if st.button("Generate Monthly Excel Report"):
            report_data = export_monthly_report(formatted_month)
            if report_data:
                st.success(f"✅ Report for {formatted_month} is ready!")
                st.download_button(
                    label="📥 Download Excel File",
                    data=report_data,
                    file_name=f"Clinic_Revenue_{formatted_month}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No patient records found for the selected month.")
