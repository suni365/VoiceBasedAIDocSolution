import streamlit as st
import sqlite3
import pandas as pd
import os
import json
from datetime import datetime, date

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

# ---------------- HELPERS ----------------
def show_json_table(j):
    try:
        df = pd.read_json(j)
        st.dataframe(df)
    except:
        st.write(j)

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🏥 Clinic Management System")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "clinic123":
            st.session_state.login = True
            st.rerun()
        else:
            st.error("Invalid login")

# ================= MAIN APP =================
else:
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Registration", "Doctor", "Lab", "Pharmacy", "Billing", "Monthly Report"]
    )

    # ---------------- DASHBOARD ----------------
    if menu == "Dashboard":
        st.title("📊 Dashboard")

        today = str(date.today())

        df = pd.read_sql("""
        SELECT p.name, v.visit_date,
        (v.consultation_fee + COALESCE(v.lab_fee,0) + COALESCE(v.med_fee,0)) AS revenue
        FROM visits v JOIN patients p ON p.patient_id = v.patient_id
        """, conn)

        st.metric("Total Visits", len(df))
        st.metric("Today's Revenue", df[df.visit_date == today]["revenue"].sum())

        st.dataframe(df[df.visit_date == today])

    # ---------------- REGISTRATION ----------------
    if menu == "Registration":
        st.title("📝 Patient Registration")

        name = st.text_input("Name")
        phone = st.text_input("Phone")
        email = st.text_input("Email")
        address = st.text_area("Address")

        if st.button("Register"):
            cursor.execute(
                "INSERT INTO patients(name,phone,email,address) VALUES (?,?,?,?)",
                (name, phone, email, address))
            conn.commit()
            st.success("Patient Registered")

    # ---------------- DOCTOR ----------------
    if menu == "Doctor":
        st.title("👨‍⚕️ Doctor Consultation")

        pid = st.number_input("Patient ID", 1)
        patient = cursor.execute("SELECT * FROM patients WHERE patient_id=?", (pid,)).fetchone()

        if patient:
            st.subheader(patient[1])
            st.write(f"📞 {patient[2]} | 🏠 {patient[4]}")

            history = pd.read_sql(
                "SELECT visit_date, diagnosis FROM visits WHERE patient_id=?",
                conn, params=(pid,))
            st.write("### Previous Visits")
            st.dataframe(history)

            symptoms = st.text_area("Symptoms")
            diagnosis = st.text_area("Diagnosis")
            tests = st.text_input("Tests Recommended")
            prescription = st.text_area("Prescription")
            fee = st.number_input("Consultation Fee", value=500.0)

            if st.button("Save Visit"):
                cursor.execute("""
                INSERT INTO visits
                (patient_id, visit_date, symptoms, diagnosis, tests,
                 prescription, consultation_fee)
                VALUES (?,?,?,?,?,?,?)
                """, (pid, str(date.today()), symptoms, diagnosis, tests, prescription, fee))
                conn.commit()
                st.success("Visit Saved")

    # ---------------- LAB ----------------
    if menu == "Lab":
        st.title("🔬 Lab")

        vid = st.number_input("Visit ID", 1)
        results = st.text_area("Results")

        df = st.data_editor(
            pd.DataFrame(columns=["Test", "Result", "Price"]),
            num_rows="dynamic"
        )

        total = df["Price"].astype(float).sum() if not df.empty else 0

        file = st.file_uploader("Upload Report PDF")

        if st.button("Save Lab"):
            path = ""
            if file:
                path = os.path.join(UPLOAD_DIR, file.name)
                open(path, "wb").write(file.getbuffer())

            cursor.execute("""
            UPDATE visits SET lab_json=?, lab_fee=?, report_path=? WHERE visit_id=?
            """, (df.to_json(), total, path, vid))
            conn.commit()
            st.success("Lab Updated")

    # ---------------- PHARMACY ----------------
    if menu == "Pharmacy":
        st.title("💊 Pharmacy")

        vid = st.number_input("Visit ID", 1)

        meds = st.data_editor(
            pd.DataFrame(columns=["Medicine", "Qty", "Price", "Timing (1-1-1)"]),
            num_rows="dynamic"
        )

        meds["Total"] = meds["Qty"].astype(float) * meds["Price"].astype(float)
        total = meds["Total"].sum()

        st.metric("Total", f"₹{total}")

        if st.button("Save Pharmacy"):
            cursor.execute("""
            UPDATE visits SET med_json=?, med_fee=? WHERE visit_id=?
            """, (meds.to_json(), total, vid))
            conn.commit()
            st.success("Pharmacy Updated")

    # ---------------- BILLING ----------------
    if menu == "Billing":
        st.title("🧾 Billing")

        vid = st.number_input("Visit ID", 1)
        v = pd.read_sql("SELECT * FROM visits WHERE visit_id=?", conn, params=(vid,))

        if not v.empty:
            v = v.iloc[0]
            total = v["consultation_fee"] + (v["lab_fee"] or 0) + (v["med_fee"] or 0)
            st.metric("Total Payable", f"₹{total}")

            st.write("### Lab")
            show_json_table(v["lab_json"])
            st.write("### Pharmacy")
            show_json_table(v["med_json"])

    # ---------------- MONTHLY REPORT ----------------
    if menu == "Monthly Report":
        st.title("📅 Monthly Report")

        start = st.date_input("Start")
        end = st.date_input("End")

        df = pd.read_sql("""
        SELECT visit_date,
        consultation_fee + COALESCE(lab_fee,0) + COALESCE(med_fee,0) AS revenue
        FROM visits WHERE visit_date BETWEEN ? AND ?
        """, conn, params=(str(start), str(end)))

        st.metric("Total Revenue", df["revenue"].sum())
        st.line_chart(df.groupby("visit_date").sum())

    if st.sidebar.button("Logout"):
        st.session_state.login = False
        st.rerun()
