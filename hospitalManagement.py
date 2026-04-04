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

# conn = sqlite3.connect('clinic_v5.db', check_same_thread=False)
# cursor = conn.cursor()

# Drop old table and recreate with new schema
# cursor.execute("DROP TABLE IF EXISTS patients")

conn = sqlite3.connect('clinic_v5.db', check_same_thread=False)
cursor = conn.cursor()

# Create table only if it does not exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS patients (
    pid INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    phone TEXT,
    email TEXT,
    address TEXT,
    emergency_contact TEXT,
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

def calc_lab_total(breakdown):
    try:
        items = breakdown.split(',')
        return sum(float(x.split(':')[1]) for x in items if ':' in x)
    except:
        return 0.0

# --- UI MODULES ---
def registration_module():
    st.header("📝 Front Desk Registration")
    with st.form("reg_form"):
        name = st.text_input("Patient Name")
        phone = st.text_input("Phone Number")
        email = st.text_input("Email (optional)")
        address = st.text_area("Home Address (optional)")
        emergency = st.text_input("Emergency Contact (optional)")
        symptoms = st.text_area("Initial Symptoms (e.g. Fever, Cough)")
        c_fees = st.number_input("Consultation Fee (₹)", value=500.0)
        
        if st.form_submit_button("Register Patient"):
            if name and phone:
                cursor.execute("""INSERT INTO patients 
                    (name, phone, email, address, emergency_contact, visit_date, basic_symptoms, consultation_fees) 
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (name, phone, email, address, emergency, str(datetime.now().date()), symptoms, c_fees))
                conn.commit()
                new_id = cursor.lastrowid
                st.success(f"Registered! Patient ID: {new_id}")
                st.link_button("📲 Send WhatsApp ID", send_wa_reg(phone, name, new_id))
            else:
                st.error("Please fill Name and Phone.")

# def doctor_module():
#     st.header("👨‍⚕️ Doctor's Consultation")
#     p_id = st.number_input("Enter Patient ID", min_value=1, step=1)
#     if p_id:
#         res = cursor.execute("SELECT name, basic_symptoms, report_path FROM patients WHERE pid=?", (p_id,)).fetchone()
#         if res:
#             st.subheader(f"Patient: {res[0]}")
#             st.warning(f"**Reported Symptoms:** {res[1]}")
            
#             # View existing reports if any
#             if res[2] and os.path.exists(res[2]):
#                 if st.checkbox("👁️ View Existing Lab Report"):
#                     display_pdf(res[2])
            
#             # Previous visits
#             prev_visits = pd.read_sql("SELECT visit_date, illness_description, test_results FROM patients WHERE name=?", conn, params=(res[0],))
#             if not prev_visits.empty:
#                 st.write("### Previous Visits")
#                 st.dataframe(prev_visits)
            
#             with st.form("doc_notes"):
#                 descr = st.text_area("Full Illness Description")
#                 tests = st.text_input("Tests Recommended")
#                 meds = st.text_area("Medicine Prescription")
#                 if st.form_submit_button("Submit Clinical Notes"):
#                     cursor.execute("UPDATE patients SET illness_description=?, test_recommendations=?, med_prescription=? WHERE pid=?",
#                                    (descr, tests, meds, p_id))
#                     conn.commit()
#                     st.success("Doctor's notes saved.")
#         else:
#             st.error("ID not found.")

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

            # Show previous visits (excluding current one)
            prev_visits = pd.read_sql(
                "SELECT visit_date, illness_description, test_results FROM patients WHERE name=? AND pid<>?",
                conn,
                params=(res[0], p_id)
            )
            if not prev_visits.empty:
                st.write("### Previous Visits")
                st.dataframe(prev_visits)

            with st.form("doc_notes"):
                descr = st.text_area("Full Illness Description")
                tests = st.text_input("Tests Recommended")
                meds = st.text_area("Medicine Prescription")
                if st.form_submit_button("Submit Clinical Notes"):
                    cursor.execute(
                        "UPDATE patients SET illness_description=?, test_recommendations=?, med_prescription=? WHERE pid=?",
                        (descr, tests, meds, p_id)
                    )
                    conn.commit()
                    st.success("Doctor's notes saved.")
        else:
            st.error("ID not found.")


# def lab_module():
#     st.header("🔬 Laboratory")
#     p_id = st.number_input("Lab: Enter Patient ID", min_value=1, step=1)
#     if p_id:
#         res = cursor.execute("SELECT name, test_recommendations FROM patients WHERE pid=?", (p_id,)).fetchone()
#         if res:
#             st.info(f"Patient: {res[0]} | **Tests Requested:** {res[1]}")
#             results = st.text_area("Test Results")
#             breakdown = st.text_area("Rate Breakdown (e.g. Blood:200, Sugar:100)")
#             # auto_total = calc_lab_total(breakdown)
#             # total_lab = st.number_input("Total Lab Amount (₹)", min_value=0.0, value=auto_total)
#             # auto_total = float(calc_lab_total(breakdown))
#             # total_lab = st.number_input("Total Lab Amount (₹)", min_value=0.0, value=auto_total)
#             # file = st.file_uploader("Upload PDF Report")

#             auto_total = float(calc_lab_total(breakdown))
#             total_lab = st.number_input("Total Lab Amount (₹)", min_value=0.0, value=auto_total, step=0.1)
#             file = st.file_uploader("Upload PDF Report")
            
#             if st.button("Submit Lab Data"):
#                 path = ""
#                 if file:
#                     path = os.path.join(UPLOAD_DIR, f"LAB_{p_id}_{file.name}")
#                     with open(path, "wb") as f: f.write(file.getbuffer())
#                 cursor.execute("UPDATE patients SET test_results=?, test_breakdown=?, test_fees=?, report_path=? WHERE pid=?",
#                                (results, breakdown, total_lab, path, p_id))
#                 conn.commit()
#                 st.success("Lab results updated.")

def lab_module():
    st.header("🔬 Laboratory")
    p_id = st.number_input("Lab: Enter Patient ID", min_value=1, step=1)
    if p_id:
        res = cursor.execute("SELECT name, test_recommendations FROM patients WHERE pid=?", (p_id,)).fetchone()
        if res:
            st.info(f"Patient: {res[0]} | **Tests Requested:** {res[1]}")

            # Free text results
            results = st.text_area("Test Results")

            # Option 1: Structured entry (new)
            st.write("### Enter Test Breakdown")
            tests_df = st.data_editor(
                pd.DataFrame(columns=["Test", "Result", "Price"]),
                num_rows="dynamic"
            )

            if not tests_df.empty:
                tests_df["Price"] = pd.to_numeric(tests_df["Price"], errors="coerce").fillna(0)
                total_lab_structured = tests_df["Price"].sum()
            else:
                total_lab_structured = 0.0

            st.metric("Total Lab Amount (Structured)", f"₹{total_lab_structured:.2f}")

            # Option 2: Keep existing breakdown text input
            breakdown = st.text_area("Rate Breakdown (e.g. Blood:200, Sugar:100)")
            auto_total = float(calc_lab_total(breakdown))
            total_lab_breakdown = st.number_input("Total Lab Amount (Breakdown)", min_value=0.0, value=auto_total, step=0.1)

            # File upload
            file = st.file_uploader("Upload PDF Report")

           if st.button("Submit Lab Data"):
               path = ""
               if file:
                   path = os.path.join(UPLOAD_DIR, f"LAB_{p_id}_{file.name}")
                   with open(path, "wb") as f: 
                       f.write(file.getbuffer())

    # Save structured test data as JSON
               cursor.execute(
                   "UPDATE patients SET test_results=?, test_breakdown=?, test_fees=?, report_path=? WHERE pid=?",
                   (results, tests_df.to_json(), max(total_lab_structured, total_lab_breakdown), path, p_id)
               )
               conn.commit()
               st.success("Lab results updated.")

# def pharmacy_module():
#     st.header("💊 Pharmacy")
#     p_id = st.number_input("Pharma: Enter Patient ID", min_value=1, step=1)
#     if p_id:
#         res = cursor.execute("SELECT name, med_prescription FROM patients WHERE pid=?", (p_id,)).fetchone()
#         if res:
#             st.info(f"Patient: {res[0]} | **Prescription:** {res[1]}")

#             # Editable medicine table
#             meds = st.data_editor(
#                 pd.DataFrame(columns=["Medicine", "Qty", "Price", "Timing"]),
#                 num_rows="dynamic"
#             )

#             # Force numeric conversion for Qty and Price
#             if not meds.empty:
#                 meds["Qty"] = pd.to_numeric(meds["Qty"], errors="coerce").fillna(0)
#                 meds["Price"] = pd.to_numeric(meds["Price"], errors="coerce").fillna(0)

#                 # Add timing options dropdown
#                 timing_options = ["Morning", "Afternoon", "Evening", "Before Food", "After Food"]
#                 meds["Timing"] = meds["Timing"].apply(
#                     lambda x: x if x in timing_options else timing_options[0]
#                 )

#                 # Calculate total per medicine
#                 meds["Total"] = meds["Qty"] * meds["Price"]

#                 # Show updated table
#                 st.dataframe(meds)

#                 # Auto grand total
#                 total_med = meds["Total"].sum()
#             else:
#                 total_med = 0.0

#             st.metric("Total Pharmacy Amount", f"₹{total_med:.2f}")

#             if st.button("Finalize Pharmacy Bill"):
#                 cursor.execute("UPDATE patients SET med_breakdown=?, med_fees=? WHERE pid=?",
#                                (meds.to_json(), total_med, p_id))
#                 conn.commit()
#                 st.success("Pharmacy charges added.")

def pharmacy_module():
    st.header("💊 Pharmacy")
    p_id = st.number_input("Pharma: Enter Patient ID", min_value=1, step=1)
    if p_id:
        res = cursor.execute("SELECT name, med_prescription FROM patients WHERE pid=?", (p_id,)).fetchone()
        if res:
            st.info(f"Patient: {res[0]} | **Prescription:** {res[1]}")

            # Editable medicine table with dropdown for Timing
            timing_options = ["Morning", "Afternoon", "Evening", "Before Food", "After Food"]

            meds = st.data_editor(
                pd.DataFrame(columns=["Medicine", "Qty", "Price", "Timing"]),
                num_rows="dynamic",
                column_config={
                    "Timing": st.column_config.SelectboxColumn(
                        "Timing",
                        options=timing_options,
                        default=timing_options[0]
                    )
                }
            )

            # Force numeric conversion for Qty and Price
            if not meds.empty:
                meds["Qty"] = pd.to_numeric(meds["Qty"], errors="coerce").fillna(0)
                meds["Price"] = pd.to_numeric(meds["Price"], errors="coerce").fillna(0)

                # Calculate total per medicine
                meds["Total"] = meds["Qty"] * meds["Price"]

                # Show updated table
                st.dataframe(meds)

                # Auto grand total
                total_med = meds["Total"].sum()
            else:
                total_med = 0.0

            st.metric("Total Pharmacy Amount", f"₹{total_med:.2f}")

            if st.button("Finalize Pharmacy Bill"):
                cursor.execute("UPDATE patients SET med_breakdown=?, med_fees=? WHERE pid=?",
                               (meds.to_json(), total_med, p_id))
                conn.commit()
                st.success("Pharmacy charges added.")





# def billing_search():
#     st.header("🔍 Search & Final Bill")
#     p_id = st.number_input("Search ID", min_value=1, step=1)
#     if p_id:
#         r = pd.read_sql("SELECT * FROM patients WHERE pid=?", conn, params=(p_id,))
#         if not r.empty:
#             p = r.iloc[0]
#             st.write(f"### {p['name']} (Visit: {p['visit_date']})")
#             c1, c2 = st.columns(2)
#             with c1:
#                 st.write(f"**Diagnosis:** {p['illness_description']}")
#                 st.write(f"**Lab Results:** {p['test_results']}")
#                 if p['report_path']: st.button("View Report", on_click=display_pdf, args=(p['report_path'],))
#             with c2:
#                 cons_fee = float(p['consultation_fees']) if pd.notnull(p['consultation_fees']) else 0.0
#                 lab_fee  = float(p['test_fees']) if pd.notnull(p['test_fees']) else 0.0
#                 med_fee  = float(p['med_fees']) if pd.notnull(p['med_fees']) else 0.0

#                 total = cons_fee + lab_fee + med_fee
#                 # total = (p['consultation_fees'] or 0) + (p['test_fees'] or 0) + (p['med_fees'] or 0)
                
#                 # st.metric("Total Payable", f"₹{total}")
#                 # st.write(f"Breakdown: Cons(₹{p['consultation_fees']}) + Lab(₹{p['test_fees']}) + Meds(₹{p['med_fees']})")
#                 st.write(f"Breakdown: Cons(₹{cons_fee}) + Lab(₹{lab_fee}) + Meds(₹{med_fee})")
#         else:
#             st.error("No record.")

def billing_search():
    st.header("🔍 Search & Final Bill")
    p_id = st.number_input("Search ID", min_value=1, step=1)
    if p_id:
        r = pd.read_sql("SELECT * FROM patients WHERE pid=?", conn, params=(p_id,))
        if not r.empty:
            p = r.iloc[0]
            st.write(f"### {p['name']} (Visit: {p['visit_date']})")
            c1, c2 = st.columns(2)

            def safe_float(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return 0.0

            with c1:
                st.write(f"**Diagnosis:** {p['illness_description']}")
                st.write(f"**Lab Results:** {p['test_results']}")
                st.write(f"**Lab Breakdown:** {p['test_breakdown']}")
                st.write(f"**Pharmacy Breakdown:** {p['med_breakdown']}")
                if p['report_path']:
                    st.button("View Report", on_click=display_pdf, args=(p['report_path'],))

            with c2:
                cons_fee = safe_float(p['consultation_fees'])
                lab_fee  = safe_float(p['test_fees'])
                med_fee  = safe_float(p['med_fees'])

                total = cons_fee + lab_fee + med_fee

                st.metric("Total Payable", f"₹{total:.2f}")
                st.write(f"Breakdown: Cons(₹{cons_fee}) + Lab(₹{lab_fee}) + Meds(₹{med_fee})")
              
        else:
            st.error("No record.")

def dashboard_module():
    st.title("📊 Clinic Dashboard")
    total_patients = cursor.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    today_patients = cursor.execute(
        "SELECT COUNT(*) FROM patients WHERE visit_date=?",
        (str(datetime.now().date()),)
    ).fetchone()[0]

    st.metric("Total Patients", total_patients)
    st.metric("Today's Patients", today_patients)

    df = pd.read_sql("SELECT visit_date FROM patients", conn)

    if not df.empty:
        # Convert visit_date to datetime for proper time-series chart
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')

        # Drop invalid dates if any
        df = df.dropna(subset=['visit_date'])

        # Count visits per day
        visit_counts = df['visit_date'].value_counts().sort_index()

        # Ensure index is sorted chronologically
        visit_counts.index = pd.to_datetime(visit_counts.index)
        visit_counts = visit_counts.sort_index()

        st.line_chart(visit_counts)



# def dashboard_module():
#     st.title("📊 Clinic Dashboard")
#     total_patients = cursor.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
#     today_patients = cursor.execute("SELECT COUNT(*) FROM patients WHERE visit_date=?", (str(datetime.now().date()),)).fetchone()[0]
#     st.metric("Total Patients", total_patients)
#     st.metric("Today's Patients", today_patients)
#     df = pd.read_sql("SELECT visit_date FROM patients", conn)
#     if not df.empty:
#         st.line_chart(df['visit_date'].value_counts().sort_index())
# def monthly_report_module():
#     st.header("📅 Monthly Report")
#     start = st.date_input("Start Date")
#     end = st.date_input("End Date")
    
#     if st.button("Generate Report"):
#         # Fetch patients between selected dates
#         df = pd.read_sql(
#             "SELECT * FROM patients WHERE visit_date BETWEEN ? AND ?",
#             conn,
#             params=(str(start), str(end))
#         )
        
#         if not df.empty:
#             st.subheader("Patient Records")
#             st.dataframe(df)

#             # Summary metrics
#             st.metric("Total Patients", len(df))
#             st.metric("Total Consultation Fees", df['consultation_fees'].sum())
#             st.metric("Total Lab Fees", df['test_fees'].sum())
#             st.metric("Total Pharmacy Fees", df['med_fees'].sum())
#             st.metric("Grand Total Revenue", df[['consultation_fees','test_fees','med_fees']].sum().sum())

#             # Charts
#             st.subheader("Visits Over Time")
#             visit_counts = df['visit_date'].value_counts().sort_index()
#             st.line_chart(visit_counts)

#             st.subheader("Revenue Breakdown")
#             revenue_breakdown = pd.DataFrame({
#                 "Consultation": [df['consultation_fees'].sum()],
#                 "Lab": [df['test_fees'].sum()],
#                 "Pharmacy": [df['med_fees'].sum()]
#             })
#             st.bar_chart(revenue_breakdown.T)
#         else:
#             st.warning("No records found for the selected period.")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🏥 Clinic Management")
    if st.text_input("User") == "admin" and st.text_input("Pass", type="password") == "clinic123":
        if st.button("Login"):
            st.session_state['logged_in'] = True
            st.rerun()
else:
    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Registration", "Doctor", "Lab", "Pharmacy", "Search/Billing", "Monthly Report"]
    )

    if menu == "Dashboard":
        dashboard_module()
    elif menu == "Registration":
        registration_module()
    elif menu == "Doctor":
        doctor_module()
    elif menu == "Lab":
        lab_module()
    elif menu == "Pharmacy":
        pharmacy_module()
    elif menu == "Search/Billing":
        billing_search()
    elif menu == "Monthly Report":
        monthly_report_module()
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
        
