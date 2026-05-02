import streamlit as st
import sqlite3
import pandas as pd
import os 
import urllib.parse
from datetime import date
import io
import json

# ---------------- CONFIG ----------------
DB = "clinic_final.db"
UPLOAD_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- DB CONNECTION ----------------
conn = sqlite3.connect(DB, check_same_thread=False)
cursor = conn.cursor()

# ---------------- TABLES ----------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (
    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    phone TEXT,
    email TEXT,
    address TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS visits (
    visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER,
    visit_date TEXT,
    symptoms TEXT,
    diagnosis TEXT,
    tests TEXT,
    lab_json TEXT,
    lab_fee REAL,
    prescription TEXT,
    med_json TEXT,
    med_fee REAL,
    consultation_fee REAL,
    report_path TEXT,
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
)
""")
conn.commit()

cursor.execute("""
CREATE TABLE IF NOT EXISTS visit_medicines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id INTEGER,
    patient_id INTEGER,
    medicine TEXT,
    days INTEGER,
    qty INTEGER,
    price REAL,
    timing TEXT,
    status TEXT DEFAULT 'pending',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(visit_id) REFERENCES visits(visit_id),
    FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
)
""")
conn.commit()

try:
    cursor.execute("ALTER TABLE visits ADD COLUMN lab_status TEXT DEFAULT 'pending'")
except sqlite3.OperationalError:
    # Column already exists
    pass

try:
    cursor.execute("ALTER TABLE visits ADD COLUMN pharmacy_status TEXT DEFAULT 'pending'")
except sqlite3.OperationalError:
    # Column already exists
    pass

conn.commit()

# ---------------- HELPERS ----------------
def send_wa_reg(phone, name, pid):
    phone = ''.join(filter(str.isdigit, phone))
    if len(phone) == 10:
        phone = "91" + phone
    msg = f"🏥 Clinic Registration\nHi {name}, your Patient ID is {pid}"
    return f"https://wa.me/{phone}?text={urllib.parse.quote(msg)}"

def send_wa_report(phone, name):
    phone = ''.join(filter(str.isdigit, phone))
    if len(phone) == 10:
        phone = "91" + phone
    msg = f"Hi {name}, your lab report is ready. Please collect it from the clinic."
    return f"https://wa.me/{phone}?text={urllib.parse.quote(msg)}"

def show_json_table(j):
    if j:
        try:
            st.dataframe(pd.read_json(j))
        except:
            st.write(j)

def display_pdf(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button("📄 Download Lab Report", f, file_name=os.path.basename(path))


def doctor_module(conn, cursor, pid, patient_name, phone_number):
    st.title(f"👨‍⚕️ Consultation: {patient_name} (ID: {pid})")

    # 1. Clinical Inputs
    col1, col2 = st.columns(2)
    with col1:
        symptoms = st.text_area("Symptoms")
        diagnosis = st.text_area("Diagnosis")
    with col2:
        tests = st.text_area("Lab Tests Recommended")
        fee = st.number_input("Consultation Fee", min_value=0, value=300)

    st.subheader("💊 Prescription")
    
    # Medicine Input UI
    if "med_list" not in st.session_state:
        st.session_state.med_list = []

    with st.form("med_form", clear_on_submit=True):
        m_col1, m_col2, m_col3 = st.columns([3, 2, 2])
        m_name = m_col1.text_input("Medicine Name")
        m_timing = m_col2.selectbox("Timing", ["1-1-1", "1-0-1", "1-1-0","0-1-1","0-0-1", "1-0-0","0-1-0", "SOS"])
        m_days = m_col3.number_input("Days", min_value=1, value=60)
        
        if st.form_submit_button("➕ Add Medicine"):
            if m_name:
                st.session_state.med_list.append({
                    "Medicine": m_name,
                    "Timing": m_timing,
                    "Days": m_days
                })

    # Show added medicines in a table
    if st.session_state.med_list:
        meds_df = pd.DataFrame(st.session_state.med_list)
        st.table(meds_df)
        if st.button("🗑️ Clear List"):
            st.session_state.med_list = []
            st.rerun()

   # 2. Save Logic (Inside doctor_module)
    if st.button("💾 Save Consultation & Generate WhatsApp"):
        if not diagnosis:
            st.error("Please enter a diagnosis before saving.")
        else:
            today = str(date.today())
            med_json = json.dumps(st.session_state.med_list)
            
            try:
                # Inside doctor_module Save Logic
                with conn:
                    cursor.execute(
                        """INSERT INTO visits (patient_id, visit_date, symptoms, diagnosis, tests, prescription, med_json, consultation_fee, med_fee, pharmacy_status, lab_status) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (pid, today, symptoms, diagnosis, tests, "See Med Table", med_json, fee, 0.0, 'pending', 'pending')
                    )
                    visit_id = cursor.lastrowid

                    for m in st.session_state.med_list:
                        cursor.execute(
                            "INSERT INTO visit_medicines (visit_id, patient_id, medicine, timing, days, status) VALUES (?, ?, ?, ?, ?, ?)",
                            (int(visit_id), int(pid), m['Medicine'], m['Timing'], m['Days'], "pending")
                        )
            except Exception as e:
                st.error(f"Database Error: {e}")
                     

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login screen
if not st.session_state.logged_in:
    st.title("🏥 Sampath M.B.B.S Clinic ")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login") and u == "admin" and p == "clinic123":
        st.session_state.logged_in = True
        st.rerun()

# Main app after login
else:
    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Registration", "Doctor", "Lab", "Pharmacy", "Billing", "Visit Summary", "Monthly Report"]
    )

    # ---------------- DASHBOARD ----------------
    if menu == "Dashboard":
        st.title("📊 Dashboard")
        today = str(date.today())

        df = pd.read_sql("""
            SELECT p.name, v.visit_date,
            (v.consultation_fee + COALESCE(v.lab_fee,0) + COALESCE(v.med_fee,0)) AS revenue
            FROM visits v JOIN patients p ON v.patient_id = p.patient_id
        """, conn)

        st.metric("Total Visits", len(df))
        st.metric("Today's Revenue", df[df.visit_date == today]["revenue"].sum())
        st.dataframe(df[df.visit_date == today])

# ---------------- REGISTRATION ----------------
    elif menu == "Registration":
        st.title("📝 Patient Registration")

    # --- Registration Form ---
        with st.form("register_form", clear_on_submit=True):
            name = st.text_input("Name")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
          
            address = st.text_area("Address")
            submitted = st.form_submit_button("Register")
            if submitted and name and phone:
                try:
                    cursor.execute(
                        "INSERT INTO patients(name,phone,email,address) VALUES (?,?,?,?)",
                        (name, phone, email,address)
                    )
                    conn.commit()
                    pid = cursor.execute("SELECT last_insert_rowid()").fetchone()[0]
                    st.success(f"Registered – Patient ID: {pid}")
                    wa_link = send_wa_reg(phone, name, pid)
                    st.markdown(f"[📲 Send WhatsApp]({wa_link})", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Registration failed: {e}")

    # --- Patient List with Edit/Delete ---
        st.subheader("👥 Existing Patients")
        patients_df = pd.read_sql("SELECT * FROM patients", conn)

        if not patients_df.empty:
            for _, row in patients_df.iterrows():
                with st.expander(f"🧑 {row['name']} (ID: {row['patient_id']})"):
                    new_name = st.text_input("Name", row["name"], key=f"name_{row['patient_id']}")
                    new_phone = st.text_input("Phone", row["phone"], key=f"phone_{row['patient_id']}")
                    new_email = st.text_input("Email", row["email"], key=f"email_{row['patient_id']}")
                    new_address = st.text_area("Address", row["address"], key=f"addr_{row['patient_id']}")
                    # new_phone = st.text_area("Addre", row["phone"], key=f"addr_{row['patient_id']}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Update", key=f"update_{row['patient_id']}"):
                            cursor.execute(
                                "UPDATE patients SET name=?, phone=?, email=?, address=? WHERE patient_id=?",
                                (new_name, new_phone, new_email, new_address, row["patient_id"])
                            )
                            conn.commit()
                            st.success("Patient updated successfully!")
                            st.rerun()
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{row['patient_id']}"):
                            cursor.execute("DELETE FROM patients WHERE patient_id=?", (row["patient_id"],))
                            conn.commit()
                            st.warning("Patient deleted successfully!")
                            st.rerun()
        else:
            st.info("No patients registered yet.")


# ---------------- DOCTOR ----------------

    elif menu == "Doctor":
    # 1. First, get the Patient ID (You can keep this part from your old code)
        pid = st.number_input("Enter Patient ID", min_value=1)
    
    # 2. Fetch basic patient details to pass to the function
        patient = cursor.execute(
            "SELECT name, phone FROM patients WHERE patient_id=?", (pid,)
        ).fetchone()

        if patient:
        # 3. CALL THE NEW IMPROVED FUNCTION HERE
            doctor_module(
                conn=conn, 
                cursor=cursor, 
                pid=pid, 
                patient_name=patient[0], 
                phone_number=patient[1],
                # patient_phone=patient[2],
                # patient_address=patient[3]
                
            )
        else:
            st.warning("Patient not found. Please check the ID.")

   
         
# ---------------- LAB ----------------
 
    elif menu == "Lab":
        st.title("🔬 Lab Module")

        lab_df = pd.read_sql(
            """
            SELECT v.visit_id, v.visit_date, p.name, p.phone, v.tests, v.lab_json,
                   v.lab_fee, v.lab_status, v.report_path
            FROM visits v
            JOIN patients p ON v.patient_id = p.patient_id
            WHERE v.tests IS NOT NULL AND v.tests != '' AND v.lab_status='pending'
            """,
            conn
         )

        total_expense = 0

        if not lab_df.empty:
            for _, row in lab_df.iterrows():
                with st.expander(f"{row['name']} – Visit {row['visit_id']}"):
                    st.write(f"Date: {row['visit_date']}")
                    st.write(f"Tests Recommended: {row['tests']}")

                # Enter lab details and fee
                    test_details = st.text_area("Enter Test Details", row["lab_json"] or "", key=f"labdet_{row['visit_id']}")
                    lab_fee = st.number_input("Lab Fee", value=row["lab_fee"] if row["lab_fee"] else 0.0, key=f"labfee_{row['visit_id']}")

                    if st.button("💾 Save Lab Record", key=f"savelab_{row['visit_id']}"):
                        cursor.execute(
                            "UPDATE visits SET lab_json=?, lab_fee=? WHERE visit_id=?",
                            (test_details, lab_fee, row["visit_id"])
                        )
                        conn.commit()
                        st.success("Lab details saved successfully!")
                        st.rerun()

                # Upload report
                    report_file = st.file_uploader("Upload Lab Report", type=["pdf"], key=f"lab_{row['visit_id']}")
                    if report_file:
                        report_path = os.path.join(UPLOAD_DIR, f"report_{row['visit_id']}.pdf")
                        with open(report_path, "wb") as f:
                            f.write(report_file.read())
                        cursor.execute("UPDATE visits SET report_path=?, lab_status='completed' WHERE visit_id=?", (report_path, row["visit_id"]))
                        conn.commit()
                        st.success("Report uploaded and marked completed!")
                        st.rerun()

                # WhatsApp link to send report
                    if row["report_path"]:
                        wa_link = send_wa_report(row["phone"], row["name"])
                        st.markdown(f"[📲 Send Report via WhatsApp]({wa_link})", unsafe_allow_html=True)

                # Add to total expense
                    if lab_fee:
                        total_expense += lab_fee

        # Show total lab expense
            st.subheader("💰 Total Lab Expense")
            st.metric("Total", total_expense)
        else:
            st.info("No pending lab tests.")

    elif menu == "Pharmacy":
        st.title("💊 Pharmacy")
        patients_df = pd.read_sql("SELECT patient_id, name FROM patients", conn)
        patient_choice = st.selectbox("Select Patient", patients_df["name"])
    
        if patient_choice:
            pid = int(patients_df.loc[patients_df["name"] == patient_choice, "patient_id"].values[0])
            visits_df = pd.read_sql(
                "SELECT visit_id, visit_date FROM visits WHERE patient_id=?", 
                conn, params=(pid,)
            )

            if not visits_df.empty:
                visit_map = {f"Visit {row['visit_id']} ({row['visit_date']})": row['visit_id'] for _, row in visits_df.iterrows()}
                visit_choice = st.selectbox("Select Visit", options=list(visit_map.keys()))
                vid = int(visit_map[visit_choice]) # Force Integer

            # Fetch ONLY pending medicines for this specific visit
                meds = pd.read_sql(
                    "SELECT id, medicine, days, timing FROM visit_medicines WHERE visit_id=? AND status='pending'",
                    conn, params=(vid,)
                )

                if meds.empty:
                    st.info("No pending medicines for this visit.")
                else:
                    total_session_fee = 0.0
                    with st.form("dispense_form"):
                        for idx, row in meds.iterrows():
                            col1, col2 = st.columns(2)
                        # We use idx to keep keys unique
                            u_qty = col1.number_input(f"Qty: {row['medicine']}", min_value=1, value=int(row['days']), key=f"q_{row['id']}")
                            u_price = col2.number_input(f"Price: {row['medicine']}", min_value=0.0, value=10.0, key=f"p_{row['id']}")
                            total_session_fee += (u_qty * u_price)
                    
                        submit = st.form_submit_button("✅ Dispense & Update Billing")

                    if submit:
                        for idx, row in meds.iterrows():
                            final_qty = st.session_state[f"q_{row['id']}"]
                            final_price = st.session_state[f"p_{row['id']}"]
                        
                            cursor.execute(
                                "UPDATE visit_medicines SET status='dispensed', qty=?, price=? WHERE id=?",
                                (final_qty, final_price, row['id'])
                            )
                    
                    # Update the main visits table with the total med fee
                        cursor.execute(
                            "UPDATE visits SET med_fee = IFNULL(med_fee, 0) + ?, pharmacy_status='completed' WHERE visit_id=?",
                            (total_session_fee, vid)
                        )
                        conn.commit()
                        st.success(f"Dispensed! Total: ₹{total_session_fee}")
                        st.rerun()

    # elif menu == "Billing":
    #     st.title("🧾 Billing")
    #     patients_df = pd.read_sql("SELECT patient_id, name FROM patients", conn)
    #     patient_choice = st.selectbox("Select Patient", patients_df["name"])
    #     if patient_choice:
    #         pid = patients_df.loc[patients_df["name"] == patient_choice, "patient_id"].values[0]
    #         visits_df = pd.read_sql(
    #             "SELECT visit_id, visit_date, consultation_fee, med_fee FROM visits WHERE patient_id=?",
    #             conn, params=(pid,)
    #         )
    #         if not visits_df.empty:
    #             visit_choice = st.selectbox(
    #                 "Select Visit",
    #                 visits_df.apply(lambda r: f"Visit {r['visit_id']} ({r['visit_date']})", axis=1),
    #             )
    #             vid = int(visit_choice.split()[1])

    #             row = visits_df.loc[visits_df["visit_id"] == vid].iloc[0]
    #             consultation = row["consultation_fee"] or 0
    #             med_total = row["med_fee"] or 0
    #             total = consultation + med_total

    #             st.metric("Total Payable", f"₹{total}")
    #             st.write(f"Consultation Fee: ₹{consultation}")
    #             st.write(f"Medicine Fee: ₹{med_total}")

    #             if st.button("Finalize Bill"):
    #                 cursor.execute(
    #                     "UPDATE visits SET billing_status='Paid' WHERE visit_id=?",
    #                     (vid,)
    #                 )
    #                 conn.commit()
    #                 st.success("Bill finalized successfully!")


    elif menu == "Billing":
        st.title("🧾 Billing")
    
    # 1. Initialize an empty dataframe to prevent NameError
        visits_df = pd.DataFrame() 

        patients_df = pd.read_sql("SELECT patient_id, name FROM patients", conn)
        patient_choice = st.selectbox("Select Patient", [""] + list(patients_df["name"])) # Added empty default

        if patient_choice and patient_choice != "":
        # Get the PID safely
            pid_val = patients_df.loc[patients_df["name"] == patient_choice, "patient_id"].values[0]
            pid = int(pid_val)

        # 2. Populate the visits_df inside the choice block
            visits_df = pd.read_sql(
                "SELECT visit_id, visit_date, consultation_fee, med_fee, lab_fee FROM visits WHERE patient_id=?",
                conn, params=(pid,)
            )

            if not visits_df.empty:
            # 3. Create the selection map
                visit_map = {f"Visit {row['visit_id']} ({row['visit_date']})": row['visit_id'] for _, row in visits_df.iterrows()}
                visit_choice = st.selectbox("Select Visit", options=list(visit_map.keys()))
            
            # Force vid to be a clean integer
                vid = int(visit_map[visit_choice])

            # 4. Fetch the most recent data for this specific visit
                visit_data = pd.read_sql(
                    "SELECT consultation_fee, med_fee, lab_fee FROM visits WHERE visit_id=?", 
                    conn, params=(vid,)
                ).iloc[0]

                cons = visit_data["consultation_fee"] or 0
                meds = visit_data["med_fee"] or 0
                labs = visit_data["lab_fee"] or 0
                total = cons + meds + labs

                st.subheader("Payment Breakdown")
                col1, col2, col3 = st.columns(3)
                col1.metric("Consultation", f"₹{cons}")
                col2.metric("Pharmacy", f"₹{meds}")
                col3.metric("Lab", f"₹{labs}")

                st.divider()
                    st.header(f"Total Payable: ₹{total}")

                if st.button("Finalize Bill & Mark Paid"):
                    cursor.execute(
                        "UPDATE visits SET billing_status='Paid' WHERE visit_id=?",
                        (vid,)
                    )
                    conn.commit()
                    st.success("Transaction completed!")
            else:
                st.info("No visit history found for this patient.")
    
# ---------------- MONTHLY REPORT ----------------
    elif menu == "Monthly Report":
        st.title("📅 Monthly Report")
        start = st.date_input("Start")
        end = st.date_input("End")

        df = pd.read_sql("""
            SELECT visit_date,
            consultation_fee + COALESCE(lab_fee,0) + COALESCE(med_fee,0) AS revenue
            FROM visits
            WHERE visit_date BETWEEN ? AND ?
        """, conn, params=(str(start), str(end)))

        st.metric("Total Revenue", df["revenue"].sum())
        st.line_chart(df.groupby("visit_date")["revenue"].sum())

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
