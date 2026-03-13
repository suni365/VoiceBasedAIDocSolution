import streamlit as st
import re

st.title("AI Log Analyzer (Local Version)")

# Text area for pasting logs
log_text = st.text_area("Paste your log here")

# File uploader for log files
uploaded_file = st.file_uploader("Or upload a log file")
if uploaded_file:
    log_text = uploaded_file.read().decode("utf-8")

# Function to extract error lines
def extract_errors(log_text):
    lines = log_text.split("\n")
    errors = []
    for line in lines:
        if re.search(r"error|exception|fail|traceback", line, re.IGNORECASE):
            errors.append(line)
    return errors

# Simple local analysis function
def analyze_errors(errors):
    if not errors:
        return "No errors detected in the log."

    analysis = []
    for i, line in enumerate(errors, start=1):
        component = "Unknown"
        suggested_fix = "Check the log context and stack trace."

        # Try to detect component from line (simple heuristic)
        if ":" in line:
            parts = line.split(":")
            if len(parts) > 1:
                component = parts[0].strip()

        analysis.append(
            f"Error {i}:\n"
            f"  Log: {line}\n"
            f"  Component: {component}\n"
            f"  Suggested Fix: {suggested_fix}\n"
        )
    return "\n".join(analysis)

if st.button("Analyze Log"):
    errors = extract_errors(log_text)
    result = analyze_errors(errors)
    st.text_area("Analysis Result", result, height=400)
