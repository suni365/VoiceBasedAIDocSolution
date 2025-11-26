# A-ZChatWithVoice_fixed.py
elif seg.startswith("SE*"):
current_txn.append(seg)
transactions.append(current_txn)
inside_txn = False
elif inside_txn:
current_txn.append(seg)


results = []
for txn in transactions:
if any(seg.startswith(search_segment) for seg in txn):
results.extend([seg for seg in txn if seg.startswith(target_segment_type + "*")])


if results:
st.success(f"✅ Found {len(results)} '{target_segment_type}' segments:")
for seg in results:
st.text(seg)
else:
st.warning("No matches found.")


# --------------------------
# XML search helpers UI (kept)
# --------------------------
st.subheader("🔍 XML Search with Full Context")
xml_file = st.file_uploader("📂 Upload XML File", type=["xml"]) if 'xml_uploader_shown' not in st.session_state else None


if xml_file:
st.success("✅ XML file uploaded successfully!")
xml_content = xml_file.getvalue()
source_tag = st.text_input("Enter source tag name (e.g., PolicyNumber):")
source_value = st.text_input("Enter source tag value (e.g., H123456789):")
target_path = st.text_input("Enter target tag/path (optional, e.g., ClaimID, StartDate):")


if st.button("Search XML"):
if source_tag and source_value:
try:
results = search_large_xml(xml_content, source_tag, source_value, target_path)
if results:
st.success(f"✅ Found {len(results)} match(es):")
for idx, res in enumerate(results, start=1):
st.markdown(f"**Result {idx}:**")
st.code(res, language="xml")
else:
st.warning("⚠️ No matching data found.")
except etree.XMLSyntaxError as xe:
st.error(f"❌ XML Syntax Error: {xe}")
except Exception as e:
st.error(f"❌ Error during XML search: {e}")
else:
st.error("Please fill both Source Tag and Source Value before searching.")


# --------------------------
# XML helper functions kept at bottom
# --------------------------


def search_large_xml(xml_content, source_tag, source_value, target_path=None):
parser = etree.XMLParser(remove_blank_text=True)
tree = etree.parse(BytesIO(xml_content), parser)
root = tree.getroot()
results = []


for elem in root.iter(source_tag):
if elem.text and elem.text.strip() == source_value.strip():
parent = elem
while parent.getparent() is not None:
parent = parent.getparent()
if target_path:
for t in parent.iter(target_path):
results.append(etree.tostring(t, pretty_print=True, encoding='unicode'))
else:
results.append(etree.tostring(parent, pretty_print=True, encoding='unicode'))
return results
