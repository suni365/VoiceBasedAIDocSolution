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

    # 2. Save Logic
    if st.button("💾 Save Consultation & Generate WhatsApp"):
        if not diagnosis:
            st.error("Please enter a diagnosis before saving.")
        else:
            today = str(date.today())
            med_json = json.dumps(st.session_state.med_list)
            
            # Insert into visits (med_fee starts at 0 for Pharmacy to fill)
            cursor.execute(
                """INSERT INTO visits (patient_id, visit_date, symptoms, diagnosis, tests, prescription, med_json, consultation_fee, med_fee) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (pid, today, symptoms, diagnosis, tests, "See Med Table", med_json, fee, 0)
            )
            visit_id = cursor.lastrowid

            # Insert individual medicines for Pharmacy module
            for m in st.session_state.med_list:
                cursor.execute(
                    "INSERT INTO visit_medicines (visit_id, patient_id, medicine, timing, days, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (visit_id, pid, m['Medicine'], m['Timing'], m['Days'], "pending")
                )
            
            conn.commit()

            # 3. Construct WhatsApp Message
            med_summary = "\n".join([f"- {m['Medicine']} ({m['Timing']} for {m['Days']} days)" for m in st.session_state.med_list])
            
            whatsapp_msg = (
                f"*Visit Summary - {today}*\n\n"
                f"*Patient ID:* {pid}\n"
                f"*Patient Name:* {patient_name}\n"
                f"*Diagnosis:* {diagnosis}\n"
                f"*Recommended Tests:* {tests if tests else 'None'}\n\n"
                f"*Prescription:*\n{med_summary}\n\n"
                f"Please visit the Pharmacy/Lab for further processing."
            )
            
            # Encode for URL
            encoded_msg = urllib.parse.quote(whatsapp_msg)
            wa_link = f"https://wa.me/{phone_number}?text={encoded_msg}"
            
            st.success("✅ Consultation Saved Successfully!")
            st.markdown(f"""
                <a href="{wa_link}" target="_blank">
                    <button style="background-color: #25D366; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                        📲 Send Details via WhatsApp
                    </button>
                </a>
                """, unsafe_allow_html=True)
            
            # Clear state for next patient
            st.session_state.med_list = []

    # 4. Professional History View
    st.divider()
    st.subheader("📜 Previous Visit History")
    history = pd.read_sql(f"SELECT * FROM visits WHERE patient_id={pid} ORDER BY visit_id DESC", conn)
    
    if not history.empty:
        for _, row in history.iterrows():
            with st.expander(f"🗓️ Visit on {row['visit_date']} (ID: {row['visit_id']})"):
                h_col1, h_col2 = st.columns(2)
                h_col1.write(f"**Symptoms:** {row['symptoms']}")
                h_col1.write(f"**Diagnosis:** {row['diagnosis']}")
                h_col2.write(f"**Tests:** {row['tests']}")
                
                if row["med_json"]:
                    st.write("**Medicines:**")
                    try:
                        h_meds = pd.DataFrame(json.loads(row["med_json"]))
                        st.table(h_meds)
                    except:
                        st.write("No medicine data found.")

# ---------------- LOGIN ----------------
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# if not st.session_state.logged_in:
#     st.title("🏥 Cl Management System")
#     u = st.text_input("Username")
#     p = st.text_input("Password", type="password")
#     if st.button("Login") and u == "admin" and p == "clinic123":
#         st.session_state.logged_in = True
#         st.rerun()

# else:
#     menu = st.sidebar.radio(
#         "Navigation",
#         ["Dashboard", "Registration", "Doctor", "Lab", "Pharmacy", "Billing", "Visit Summary", "Monthly Report"]
#     )




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

    # elif menu == "Pharmacy":
    #     st.title("💊 Pharmacy")
    #     patients_df = pd.read_sql("SELECT patient_id, name FROM patients", conn)
    #     patient_choice = st.selectbox("Select Patient", patients_df["name"])
    #     if patient_choice:
    #         pid = patients_df.loc[patients_df["name"] == patient_choice, "patient_id"].values[0]
    #         visits_df = pd.read_sql(
    #             "SELECT visit_id, visit_date, diagnosis FROM visits WHERE patient_id=?",
    #             conn, params=(pid,)
    #         )

    #         if not visits_df.empty:
    #             visit_choice = st.selectbox(
    #                 "Select Visit",
    #                 visits_df.apply(lambda r: f"Visit {r['visit_id']} ({r['visit_date']})", axis=1),
    #             )
    #             vid = int(visit_choice.split()[1])
    #             meds = pd.read_sql(
    #             "SELECT id, medicine, days, timing FROM visit_medicines WHERE visit_id=? AND status='pending'",
    #             conn, params=(vid,)
    #             )

    #             if meds.empty:
    #                 st.info("No pending medicines for this visit.")
    #             else:
    #                 total = 0
    #                 for idx, row in meds.iterrows():
    #                     qty = st.number_input(f"Qty for {row['medicine']}", min_value=1, value=row["days"], key=f"qty_{idx}")
    #                     price = st.number_input(f"Price per unit for {row['medicine']}", min_value=0, value=10, key=f"price_{idx}")
    #                     if st.button(f"Dispense {row['medicine']}", key=f"dispense_{idx}"):
    #                         total = qty * price
    #                         cursor.execute(
    #                             "UPDATE visit_medicines SET status=?, qty=?, price=? WHERE id=?",
    #                             ("Dispensed", qty, price, row["id"])
    #                         )
    #                         cursor.execute(
    #                             "UPDATE visits SET med_fee = med_fee + ? WHERE visit_id=?",
    #                             (total, vid)
    #                         )
    #                         conn.commit()
    #                         st.success(f"{row['medicine']} dispensed successfully!")
    #                         st.rerun()

    #                 # for idx, row in meds.iterrows():
    #                 #     qty = st.number_input(f"Qty for {row['medicine']}", min_value=1, value=row["days"], key=f"qty_{idx}")
    #                 #     price = st.number_input(f"Price per unit for {row['medicine']}", min_value=0, value=10, key=f"price_{idx}")
    #                 #     total += qty * price

    #                 #     if st.button(f"Dispense {row['medicine']}", key=f"dispense_{idx}"):
    #                 #         cursor.execute(
    #                 #             "UPDATE visit_medicines SET status=?, qty=?, price=? WHERE id=?",
    #                 #             ("Dispensed", qty, price, row["id"])
    #                 #         )
    #                 #         cursor.execute(
    #                 #             "UPDATE visits SET med_fee = med_fee + ? WHERE visit_id=?",
    #                 #             (qty * price, vid)
    #                 #         )
    #                 #         conn.commit()
    #                 #         st.success(f"{row['medicine']} dispensed successfully!")
    #                 #         st.rerun()

    #                 st.metric("Total Medicine Fee (this visit)", f"₹{total}")

    elif menu == "Pharmacy":
        st.title("💊 Pharmacy")
        patients_df = pd.read_sql("SELECT patient_id, name FROM patients", conn)
        patient_choice = st.selectbox("Select Patient", patients_df["name"])
        if patient_choice:
            pid = patients_df.loc[patients_df["name"] == patient_choice, "patient_id"].values[0]
            visits_df = pd.read_sql(
                "SELECT visit_id, visit_date, diagnosis FROM visits WHERE patient_id=?",
                conn, params=(pid,)
            )

            if not visits_df.empty:
                visit_choice = st.selectbox(
                    "Select Visit",
                    visits_df.apply(lambda r: f"Visit {r['visit_id']} ({r['visit_date']})", axis=1),
                )
                vid = int(visit_choice.split()[1])
                meds = pd.read_sql(
                    "SELECT id, medicine, days, timing, qty, price, status FROM visit_medicines WHERE visit_id=? AND status='pending'",
                    conn, params=(vid,)
                )

                if meds.empty:
                    st.info("No pending medicines for this visit.")
                else:
                    st.subheader("📝 Enter Qty & Price")
                    total_fee = 0

                    # Collect qty/price for each medicine
                    for idx, row in meds.iterrows():
                        qty = st.number_input(
                            f"Qty for {row['medicine']}", 
                            min_value=1, 
                            value=row["days"], 
                            key=f"qty_{idx}"
                        )
                        price = st.number_input(
                            f"Price per unit for {row['medicine']}", 
                            min_value=0, 
                            value=10, 
                            key=f"price_{idx}"
                        )
                        total_fee += qty * price

                    st.metric("Total Medicine Fee (this visit)", f"₹{total_fee}")

                    # Dispense all medicines together
                    if st.button("✅ Dispense All Medicines"):
                        for idx, row in meds.iterrows():
                            qty = st.session_state[f"qty_{idx}"]
                            price = st.session_state[f"price_{idx}"]
                            cursor.execute(
                                "UPDATE visit_medicines SET status=?, qty=?, price=? WHERE id=?",
                                ("dispensed", qty, price, row["id"])
                            )
                            cursor.execute(
                                "UPDATE visits SET med_fee = med_fee + ? WHERE visit_id=?",
                                (qty * price, vid)
                            )
                        conn.commit()
                        st.success("All medicines dispensed successfully!")
                        st.rerun()

    elif menu == "Billing":
        st.title("🧾 Billing")
        patients_df = pd.read_sql("SELECT patient_id, name FROM patients", conn)
        patient_choice = st.selectbox("Select Patient", patients_df["name"])
        if patient_choice:
            pid = patients_df.loc[patients_df["name"] == patient_choice, "patient_id"].values[0]
            visits_df = pd.read_sql(
                "SELECT visit_id, visit_date, consultation_fee, med_fee FROM visits WHERE patient_id=?",
                conn, params=(pid,)
            )
            if not visits_df.empty:
                visit_choice = st.selectbox(
                    "Select Visit",
                    visits_df.apply(lambda r: f"Visit {r['visit_id']} ({r['visit_date']})", axis=1),
                )
                vid = int(visit_choice.split()[1])

                row = visits_df.loc[visits_df["visit_id"] == vid].iloc[0]
                consultation = row["consultation_fee"] or 0
                med_total = row["med_fee"] or 0
                total = consultation + med_total

                st.metric("Total Payable", f"₹{total}")
                st.write(f"Consultation Fee: ₹{consultation}")
                st.write(f"Medicine Fee: ₹{med_total}")

                if st.button("Finalize Bill"):
                    cursor.execute(
                        "UPDATE visits SET billing_status='Paid' WHERE visit_id=?",
                        (vid,)
                    )
                    conn.commit()
                    st.success("Bill finalized successfully!")


   
          
# ---------------- VISIT SUMMARY ----------------
    elif menu == "Visit Summary":
        st.title("📋 Visit Summary")

    # Select patient
        patients_df = pd.read_sql("SELECT patient_id, name FROM patients", conn)
        patient_choice = st.selectbox("Select Patient", patients_df["name"]) if not patients_df.empty else None

        if patient_choice:
            pid = patients_df.loc[patients_df["name"] == patient_choice, "patient_id"].values[0]

        # Show visits for this patient
            visits_df = pd.read_sql(
                """
                SELECT v.visit_id, v.visit_date, v.symptoms, v.diagnosis, v.tests,
                       v.prescription, v.consultation_fee, v.lab_fee, v.med_fee,
                       v.lab_json, v.med_json, v.report_path, v.lab_status, v.pharmacy_status
                FROM visits v
                WHERE v.patient_id=?
                """,
                conn, params=(pid,)
            )

            if not visits_df.empty:
                visit_choice = st.selectbox(
                    "Select Visit",
                    visits_df.apply(lambda r: f"Visit {r['visit_id']} ({r['visit_date']})", axis=1)
                )
                vid = int(visit_choice.split()[1])

                row = visits_df.loc[visits_df["visit_id"] == vid].iloc[0]

            # Doctor details
                st.subheader("🩺 Doctor Consultation")
                st.write(f"Symptoms: {row['symptoms']}")
                st.write(f"Diagnosis: {row['diagnosis']}")
                st.write(f"Tests Recommended: {row['tests']}")
                st.write(f"Prescription: {row['prescription']}")
                st.write(f"Consultation Fee: ₹{row['consultation_fee'] or 0}")

            # Lab details
                st.subheader("🔬 Lab Results")
                st.write(f"Lab Status: {row['lab_status']}")
                st.write(f"Lab Fee: ₹{row['lab_fee'] or 0}")
                show_json_table(row["lab_json"])
                display_pdf(row["report_path"])

            # Pharmacy details
                st.subheader("💊 Pharmacy")
                st.write(f"Pharmacy Status: {row['pharmacy_status']}")
                st.write(f"Medicine Fee: ₹{row['med_fee'] or 0}")
                show_json_table(row["med_json"])

            # Billing summary
                st.subheader("🧾 Billing Summary")
                consultation = row["consultation_fee"] or 0
                lab = row["lab_fee"] or 0
                medicine = row["med_fee"] or 0
                total = consultation + lab + medicine
                st.metric("Total Payable", f"₹{total}")
            else:
                st.info("No visits found for this patient.")

    
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
