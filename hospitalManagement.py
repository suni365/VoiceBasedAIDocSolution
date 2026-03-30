import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime

import urllib.parse

def send_whatsapp_msg(phone, name, patient_id, total_fees):
    # Ensure phone has the country code (91 for India)
    if not phone.startswith('91'):
        phone = f"91{phone}"
    
    # Create a professional message template
    message = (
        f"*🏥 Clinic Visit Summary*\n\n"
        f"Hi *{name}*,\n"
        f"Thank you for visiting us today.\n\n"
        f"🆔 *Patient ID:* {patient_id}\n"
        f"💰 *Total Fees:* ₹{total_fees}\n\n"
        f"Please keep this ID for your next visit. Get well soon!"
    )
    
    # Encode the message for a URL
    encoded_msg = urllib.parse.quote(message)
    whatsapp_url = f"https://wa.me/{phone}?text={encoded_msg}"
    return whatsapp_url

UPLOAD_DIR = "patient_reports"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Create DB connection
conn = sqlite3.connect('clinic.db', check_same_thread=False)
cursor = conn.cursor()

# Create table if not exists
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
    testing_fees REAL
)
''')

conn.commit()

# --- 1. CONFIGURATION & DATABASE SETUP ---
# UPLOAD_DIR = "patient_reports"
# if not os.path.exists(UPLOAD_DIR):
#     os.makedirs(UPLOAD_DIR)

# # conn = sqlite3.connect('clinic_data.db', check_same_thread=False)
# conn = sqlite3.connect('clinic_v2.db', check_same_thread=False)

# # 1. CREATE THE CURSOR FIRST
# cursor = conn.cursor()

# # 2. DEFINE THE MIGRATION FUNCTION
# def migrate_db():
#     try:
#         # Check if the new column exists
#         cursor.execute("SELECT testing_fees FROM patients LIMIT 1")
#     except sqlite3.OperationalError:
#         # If it fails, add the column
#         st.info("Updating database schema... adding 'testing_fees' column.")
#         cursor.execute("ALTER TABLE patients ADD COLUMN testing_fees REAL DEFAULT 0")
#         conn.commit()
#         st.success("Database updated successfully!")

# # 3. NOW CALL MIGRATION
# migrate_db()

# # 4. INITIALIZE TABLE (This will run if the DB is brand new)
# cursor.execute('''CREATE TABLE IF NOT EXISTS patients 
#                  (pid TEXT PRIMARY KEY, name TEXT, phone TEXT, 
#                   visit_date TEXT, illness TEXT, report_path TEXT, 
#                   fees REAL, testing_done TEXT, testing_fees REAL)''')
# conn.commit()

# def migrate_db():
#     try:
#         # Check if the new column exists by trying to select it
#         cursor.execute("SELECT testing_fees FROM patients LIMIT 1")
#     except sqlite3.OperationalError:
#         # If it fails, the column is missing. Let's add it!
#         st.info("Updating database schema... adding 'testing_fees' column.")
#         cursor.execute("ALTER TABLE patients ADD COLUMN testing_fees REAL DEFAULT 0")
#         conn.commit()
#         st.success("Database updated successfully!")

# migrate_db()
# cursor = conn.cursor()
# # Added testing_fees to track lab revenue
# cursor.execute('''CREATE TABLE IF NOT EXISTS patients 
#                  (pid TEXT PRIMARY KEY, name TEXT, phone TEXT, 
#                   visit_date TEXT, illness TEXT, report_path TEXT, 
#                   fees REAL, testing_done TEXT, testing_fees REAL)''')
# conn.commit()

# --- 2. LIVE DATA CALCULATIONS ---
# def get_daily_stats():
#     today = str(datetime.now().date())
    
#     # 1. Count Patients
#     cursor.execute("SELECT COUNT(*) FROM patients WHERE visit_date = ?", (today,))
#     count = cursor.fetchone()[0]
    
#     # 2. Sum Consultation Fees
#     cursor.execute("SELECT SUM(fees) FROM patients WHERE visit_date = ?", (today,))
#     fees_sum = cursor.fetchone()[0] or 0
    
#     # 3. Sum Testing/Medicine Revenue
#     cursor.execute("SELECT SUM(testing_fees) FROM patients WHERE visit_date = ?", (today,))
#     test_sum = cursor.fetchone()[0] or 0
    
#     return count, fees_sum, test_sum

# # --- 3. UI MODULES ---

# def show_dashboard():
#     st.markdown("<h1 class='main-header'>🏥 Clinic Dashboard</h1>", unsafe_allow_html=True)
    
#     # PULL LIVE DATA
#     p_count, f_total, t_total = get_daily_stats()
    
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Patients Today", f"{p_count}")
#     with col2:
#         st.metric("Consultation Fees", f"₹ {f_total:,.2f}")
#     with col3:
#         st.metric("Lab/Testing Revenue", f"₹ {t_total:,.2f}")
#     with col4:
#         st.metric("Total Revenue", f"₹ {f_total + t_total:,.2f}")

#     st.divider()
    
#     # Doctors & Appointments (Still Dummy for now)
#     l, r = st.columns(2)
#     with l:
#         st.subheader("👨‍⚕️ Doctors Available")
#         st.table(pd.DataFrame({"Doctor": ["Dr. Arya", "Dr. Smith"], "Status": ["Active", "On Call"]}))
#     with r:
#         st.subheader("📅 Recent Registrations")
#         recent_df = pd.read_sql("SELECT name, visit_date, fees FROM patients ORDER BY pid DESC LIMIT 5", conn)
#         st.dataframe(recent_df, use_container_width=True)

def get_daily_stats():
    # We will return dummy values for now to avoid SQL errors
    p_count = 12          # Total Patients Today
    fees_sum = 4500.0     # Consultation Revenue
    test_sum = 2100.0     # Lab/Medicine Revenue
    return p_count, fees_sum, test_sum

# --- 3. UI MODULES ---

def show_dashboard():
    st.markdown("<h1 class='main-header'>🏥 Clinic Dashboard</h1>", unsafe_allow_html=True)
    
    # PULL DUMMY DATA
    p_count, f_total, t_total = get_daily_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Patients Today", f"{p_count}", delta="2")
    with col2:
        st.metric("Consultation Fees", f"₹ {f_total:,.2f}", delta="10%")
    with col3:
        st.metric("Lab/Testing Revenue", f"₹ {t_total:,.2f}", delta="-5%")
    with col4:
        st.metric("Total Revenue", f"₹ {f_total + t_total:,.2f}")

    st.divider()
    
    # 4. DOCTORS & RECENT VISITS
    l, r = st.columns(2)
    with l:
        st.subheader("👨‍⚕️ Doctors Available Today")
        # Hardcoded for the UI demo
        doctors_df = pd.DataFrame({
            "Doctor Name": ["Dr. Arya (Pediatrics)", "Dr. Smith (General)"],
            "Status": ["In Consultation", "Available"],
            "Room": ["101", "102"]
        })
        st.table(doctors_df)
        
    with r:
        st.subheader("📅 Today's Appointment Queue")
        # Hardcoded dummy list
        appointments = pd.DataFrame({
            "Time": ["10:00 AM", "10:30 AM", "11:00 AM"],
            "Patient": ["Rahul Sharma", "Sita Devi", "Anjali Nair"],
            "Status": ["Completed", "Waiting", "Waiting"]
        })
        st.dataframe(appointments, use_container_width=True)

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
            if submit and name and phone:
            patient_id = f"PAT-{datetime.now().strftime('%y%m%d%H%M')}"
            # ... (keep your existing file saving logic here) ...
            
            cursor.execute("INSERT INTO patients VALUES (?,?,?,?,?,?,?,?,?)", 
                           (patient_id, name, phone, str(visit_date), illness, path, fees, testing, t_fees))
            conn.commit()
            
            # --- NEW WHATSAPP BUTTON ---
            st.success(f"Record Saved! ID: {patient_id}")
            
            total_amt = fees + t_fees
            wa_link = send_whatsapp_msg(phone, name, patient_id, total_amt)
            
            # Use a link button for the best user experience
            st.link_button("📲 Send Summary via WhatsApp", wa_link)

# def search_records():
#     st.markdown("<h2 class='main-header'>🔍 Patient Records</h2>", unsafe_allow_html=True)
#     search = st.text_input("Search Name/Phone")
#     if search:
#         df = pd.read_sql("SELECT * FROM patients WHERE name LIKE ? OR phone LIKE ?", 
#                          conn, params=(f'%{search}%', f'%{search}%'))
#         for _, row in df.iterrows():
#             with st.expander(f"👤 {row['name']} ({row['visit_date']})"):
#                 st.write(f"**ID:** {row['pid']} | **Fees:** ₹{row['fees'] + row['testing_fees']}")
#                 st.write(f"**Diagnosis:** {row['illness']}")
#                 if st.session_state.get('user_role') == 'admin':
#                     if st.button(f"Delete {row['pid']}", key=row['pid']):
#                         cursor.execute("DELETE FROM patients WHERE pid=?", (row['pid'],))
#                         conn.commit()
#                         st.rerun()

# # --- 4. NAVIGATION ---
# if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

# if not st.session_state['logged_in']:
#     st.title("Clinic Login")
#     u, p = st.text_input("User"), st.text_input("Pass", type="password")
#     if st.button("Login"):
#         if u == "admin" and p == "clinic123":
#             st.session_state.update({'logged_in': True, 'user_role': 'admin'})
#             st.rerun()
# else:
#     choice = st.sidebar.radio("Menu", ["Dashboard", "Register Patient", "Search Records"])
#     if st.sidebar.button("Logout"):
#         st.session_state['logged_in'] = False
#         st.rerun()
    
#     if choice == "Dashboard": show_dashboard()
#     elif choice == "Register Patient": register_patient()
#     elif choice == "Search Records": search_records()

def search_records():
    st.markdown("<h2 class='main-header'>🔍 Patient Records</h2>", unsafe_allow_html=True)
    
    search = st.text_input("Search by Name / Phone / Patient ID")

    if search:
        df = pd.read_sql("""
            SELECT * FROM patients 
            WHERE name LIKE ? 
            OR phone LIKE ? 
            OR pid LIKE ?
        """, conn, params=(f'%{search}%', f'%{search}%', f'%{search}%'))

        if df.empty:
            st.warning("No records found")
            return

        for _, row in df.iterrows():
            with st.expander(f"👤 {row['name']} | {row['visit_date']}"):

                # --- BASIC DETAILS ---
                st.write(f"**🆔 Patient ID:** {row['pid']}")
                st.write(f"**📞 Phone:** {row['phone']}")
                st.write(f"**🩺 Diagnosis:** {row['illness']}")

                # --- FEES ---
                st.write(f"**💰 Consultation Fees:** ₹{row['fees']}")
                st.write(f"**🧪 Testing Fees:** ₹{row['testing_fees']}")
                st.write(f"**💵 Total:** ₹{row['fees'] + row['testing_fees']}")

                # --- TEST DETAILS ---
                st.write(f"**🧾 Tests / Medicines:** {row['testing_done']}")

                # --- FILE DISPLAY ---
                if row['report_path'] and os.path.exists(row['report_path']):
                    
                    file_path = row['report_path']
                    file_ext = file_path.split('.')[-1].lower()

                    st.write("**📎 Uploaded Report:**")

                    # Show image
                    if file_ext in ['png', 'jpg', 'jpeg']:
                        st.image(file_path, caption="Patient Report", use_container_width=True)

                    # Show PDF
                    elif file_ext == 'pdf':
                        with open(file_path, "rb") as f:
                            st.download_button("📥 Download Report", f, file_name=os.path.basename(file_path))

                    else:
                        st.info("File uploaded (preview not supported)")
                        with open(file_path, "rb") as f:
                            st.download_button("📥 Download File", f, file_name=os.path.basename(file_path))

                else:
                    st.info("No report uploaded")

                # --- DELETE OPTION ---
                if st.session_state.get('user_role') == 'admin':
                    if st.button(f"❌ Delete {row['pid']}", key=row['pid']):
                        cursor.execute("DELETE FROM patients WHERE pid=?", (row['pid'],))
                        conn.commit()
                        st.success("Deleted successfully")
                        st.rerun()

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🏥 Clinic Management System")
    st.subheader("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Simple check - you can change these credentials
        if u == "admin" and p == "clinic123":
            st.session_state.update({'logged_in': True, 'user_role': 'admin'})
            st.rerun()
        else:
            st.error("Invalid credentials")
else:
    # Sidebar Menu
    st.sidebar.title(f"Welcome, {st.session_state.get('user_role', 'User')}")
    choice = st.sidebar.radio("Menu", ["Dashboard", "Register Patient", "Search Records"])
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
    
    # Routing to the functions you defined
    if choice == "Dashboard":
        show_dashboard()
    elif choice == "Register Patient":
        register_patient()
    elif choice == "Search Records":
        search_records()
