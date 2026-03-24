import streamlit as st
import sqlite3
import pandas as pd

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f8ff; /* Light medical blue background */
    }
    .main-header {
        color: #004d4d;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def login():
    st.markdown("<h1 class='main-header'>Clinic Management Login</h1>", unsafe_allow_html=True)
    with st.container():
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if user == "admin" and password == "clinic123": # Simple check
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

def main_app():
    st.sidebar.title("🩺 Navigation")
    menu = ["Register Patient", "Search Records", "Today's Appointments"]
    choice = st.sidebar.selectbox("Go to", menu)

    if choice == "Search Records":
        st.header("🔍 Patient Search")
        # Search by Name OR Phone
        search_term = st.text_input("Search by Patient Name or Phone Number")
        
        if search_term:
            query = f"SELECT * FROM patients WHERE name LIKE '%{search_term}%' OR phone LIKE '%{search_term}%'"
            df = pd.read_sql(query, conn)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No records found.")

# Logic to switch between Login and App
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
else:
    main_app()
