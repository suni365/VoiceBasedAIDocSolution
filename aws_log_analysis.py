import streamlit as st
import re

st.title("Smart Log Analyzer")

log_text = st.text_area("Paste your log here")

uploaded_file = st.file_uploader("Or upload a log file")
if uploaded_file:
    log_text = uploaded_file.read().decode("utf-8")

def parse_log_line(line):
    """
    Parses a single log line to extract:
    - Timestamp
    - Component (package/class)
    - Error message
    """
    timestamp = ""
    component = ""
    message = line.strip()

    # Try to extract timestamp (common log format)
    ts_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+)', line)
    if ts_match:
        timestamp = ts_match.group(1)
        message = line[len(timestamp):].strip()

    # Try to extract component/class from stack trace
    comp_match = re.search(r'at ([\w\.]+\w)\(', message)
    if comp_match:
        component = comp_match.group(1)

    # Try to extract actual error message (for Caused by lines)
    error_match = re.search(r'Caused by: (.*)', message)
    if error_match:
        message = error_match.group(1)

    return timestamp, component, message

def extract_errors(log_text):
    lines = log_text.split("\n")
    errors = []
    for line in lines:
        if re.search(r'error|exception|fail|traceback', line, re.IGNORECASE):
            errors.append(parse_log_line(line))
    return errors

def analyze_errors(errors):
    if not errors:
        return "No errors detected."

    analysis = []
    for i, (ts, comp, msg) in enumerate(errors, start=1):
        suggested_fix = "Check stack trace and code."
        # Add some basic known patterns
        if "SQLServerException" in msg:
            suggested_fix = "Check database column names and query."
        elif "NullPointerException" in msg:
            suggested_fix = "Check for null values before using the object."
        elif "TimeoutException" in msg:
            suggested_fix = "Check service connectivity or increase timeout."

        analysis.append(
            f"Error {i}:\n"
            f"  Timestamp: {ts or 'N/A'}\n"
            f"  Component: {comp or 'N/A'}\n"
            f"  Error Message: {msg}\n"
            f"  Suggested Fix: {suggested_fix}\n"
        )
    return "\n\n".join(analysis)

if st.button("Analyze Log"):
    errors = extract_errors(log_text)
    result = analyze_errors(errors)
    st.text_area("Analysis Result", result, height=600)
