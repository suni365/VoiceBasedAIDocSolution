import streamlit as st
import sqlite3
import pandas as pd
import os 
import urllib.parse
from datetime import date

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

# ---------------- LOGIN ----------------
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False

# if not st.session_state.logged_in:
#     st.title("🏥 Clinic Management System")
#     u = st.text_input("Username")
#     p = st.text_input("Password", type="password")
#     if st.button("Login") and u == "admin" and p == "clinic123":
#         st.session_state.logged_in = True
#         st.rerun()

# else:
#     menu = st.sidebar.radio(
#         "Navigation",
#         ["Dashboard", "Registration", "Doctor", "Lab", "Pharmacy", "Billing", "Monthly Report"]
#     )


#     # ---------------- DASHBOARD ----------------
# if menu == "Dashboard":
#     st.title("📊 Dashboard")
#     today = str(date.today())

#     df = pd.read_sql("""
#         SELECT p.name, v.visit_date,
#         (v.consultation_fee + COALESCE(v.lab_fee,0) + COALESCE(v.med_fee,0)) AS revenue
#         FROM visits v JOIN patients p ON v.patient_id = p.patient_id
#     """, conn)

#     st.metric("Total Visits", len(df))
#     st.metric("Today's Revenue", df[df.visit_date == today]["revenue"].sum())
#     st.dataframe(df[df.visit_date == today])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Login screen
if not st.session_state.logged_in:
    st.title("🏥 SAMPATH M.B.B.S Clinic")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login") and u == "admin" and p == "clinic123":
        st.session_state.logged_in = True
        st.rerun()

# Main app after login
else:
    menu = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Registration", "Doctor", "Lab", "Pharmacy", "Billing", "Monthly Report"]
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
            email = st.text_input("Email")
            address = st.text_area("Address")
            phone = st.text_area("Phone")
            submitted = st.form_submit_button("Register")
            if submitted and name and phone:
                try:
                    cursor.execute(
                        "INSERT INTO patients(name,phone,email,phone,address) VALUES (?,?,?,?,?)",
                        (name, phone, email, phone, address)
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
        st.title("👨‍⚕️ Doctor Consultation")
        pid = st.number_input("Patient ID", 1)

        patient = cursor.execute(
            "SELECT * FROM patients WHERE patient_id=?", (pid,)
        ).fetchone()

        if patient:
            st.subheader(patient[1])
            st.write(f"📞 {patient[2]} | 🏠 {patient[4]}")

        # Show full visit history with more details
            history = pd.read_sql(
                """
                SELECT visit_id, visit_date, symptoms, diagnosis, tests,
                    prescription, med_json, med_fee, lab_json, lab_fee,
                    consultation_fee
                FROM visits
                WHERE patient_id=?
                """,
                conn, params=(pid,)
            )

            if not history.empty:
                st.subheader("📜 Previous Visit History")
                st.dataframe(history)
            else:
                st.info("No previous visits found")

        # New consultation entry
            st.subheader("🩺 New Consultation")
            symptoms = st.text_area("Symptoms")
            diagnosis = st.text_area("Diagnosis")
            tests = st.text_input("Tests Recommended")
            prescription = st.text_area("Prescription")
            fee = st.number_input("Consultation Fee", value=300.0)

            meds = st.data_editor(
                pd.DataFrame(columns=["Medicine", "Qty", "Price", "Timing (1-1-1)"]),
                num_rows="dynamic"
            )

            if st.button("💾 Save Consultation"):
                today = str(date.today())
                cursor.execute(
                    """ 
                    INSERT INTO visits(patient_id, visit_date, symptoms, diagnosis, tests,
                                       prescription, med_json, consultation_fee)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (pid, today, symptoms, diagnosis, tests,
                    prescription, meds.to_json(), fee)
                )
                conn.commit()
                st.success("Consultation saved successfully!")
                st.rerun()

            # if st.button("💾 Save Consultation"):
            #     today = str(date.today())
            #     try:
            #         cursor.execute(
            #             """
            #             INSERT INTO visits(patient_id, visit_date, symptoms, diagnosis, tests,
            #                            prescription, consultation_fee)
            #             VALUES (?, ?, ?, ?, ?, ?, ?)
            #             """,
            #             (pid, today, symptoms, diagnosis, tests, prescription, fee)
            #         )
            #         conn.commit()
            #         st.success("Consultation saved successfully!")
            #         st.rerun()
                except Exception as e:
                    st.error(f"Failed to save consultation: {e}")

        # Optional: show lab results and medicines if available
            st.subheader("🔬 Lab Results & Medicines")
            if not history.empty:
                for _, row in history.iterrows():
                    with st.expander(f"Visit {row['visit_id']} on {row['visit_date']}"):
                        st.write(f"Diagnosis: {row['diagnosis']}")
                        st.write(f"Symptoms: {row['symptoms']}")
                        st.write(f"Tests: {row['tests']}")
                        st.write(f"Prescription: {row['prescription']}")
                        st.write(f"Consultation Fee: {row['consultation_fee']}")
                        st.write(f"Medicine Fee: {row['med_fee']}")
                        st.write(f"Lab Fee: {row['lab_fee']}")
                        show_json_table(row["med_json"])
                        show_json_table(row["lab_json"])

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
        vid = st.number_input("Visit ID", 1)

        meds = st.data_editor(
        pd.DataFrame(columns=["Medicine", "Qty", "Price", "Timing (1-1-1)"]),
        num_rows="dynamic"
        )
        if not meds.empty:
            meds["Total"] = meds["Qty"] * meds["Price"]
            total = meds["Total"].sum()
        else:
            total = 0
            st.metric("Total", f"₹{total}")
            if st.button("Save Pharmacy"):
                cursor.execute("""
                    UPDATE visits SET med_json=?, med_fee=? WHERE visit_id=?
                """, (meds.to_json(), total, vid))

                conn.commit()
                st.success("Pharmacy saved")

# ---------------- BILLING ----------------
    elif menu == "Billing":
        st.title("🧾 Billing")
        vid = st.number_input("Visit ID", 1)

        v = cursor.execute("SELECT * FROM visits WHERE visit_id=?", (vid,)).fetchone()
        if v:
            total = (v[11] or 0) + (v[7] or 0) + (v[9] or 0)
            st.metric("Total Payable", f"₹{total}")

            st.write("### Lab")
            show_json_table(v[6])
            display_pdf(v[12])

            st.write("### Pharmacy")
            show_json_table(v[8])

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
