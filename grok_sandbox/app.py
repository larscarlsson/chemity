import streamlit as st
import os
import google.generativeai as genai
import PyPDF2
import pandas as pd
import json
from io import BytesIO
from pdf2image import convert_from_bytes

# Hardcoded RoHS restricted substances (Directive 2011/65/EU)
ROHS_SUBSTANCES = [
    {"name": "Lead", "threshold": "0.1%"},
    {"name": "Mercury", "threshold": "0.1%"},
    {"name": "Cadmium", "threshold": "0.01%"},
    {"name": "Hexavalent Chromium", "threshold": "0.1%"},
    {"name": "Polybrominated Biphenyls (PBB)", "threshold": "0.1%"},
    {"name": "Polybrominated Diphenyl Ethers (PBDE)", "threshold": "0.1%"},
    {"name": "Bis(2-ethylhexyl) Phthalate (DEHP)", "threshold": "0.1%"},
    {"name": "Butyl Benzyl Phthalate (BBP)", "threshold": "0.1%"},
    {"name": "Dibutyl Phthalate (DBP)", "threshold": "0.1%"},
    {"name": "Diisobutyl Phthalate (DIBP)", "threshold": "0.1%"}
]

# Configure Gemini API and check connection
api_key = os.getenv("GEMINI_API_KEY")
gemini_connected = False
initial_response = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        initial_response = model.generate_content("Test connection")
        gemini_connected = True
    except Exception as e:
        st.error(f"Failed to connect to Gemini API: {str(e)}")
else:
    st.error("Gemini API key not set. Please set the GEMINI_API_KEY environment variable.")

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = {
        "documents": [],
        "components": [],
        "reach_substances": [],
        "rohs_substances": [],
        "scip_number": "",
        "country_origin": "",
        "taric_code": "",
        "safe_use": False,
        "safe_use_doc": None,
        "compliance_docs": [],
        "reach_guidance": "",
        "rohs_guidance": "",
        "candidate_list": None,
        "extraction_issues": [],
        "translated_texts": {},
        "raw_extracted_texts": {}
    }
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=[
        "Field", "Value", "Step", "Source", "Validation"
    ])

# Function to extract text from PDF
def extract_pdf_text(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages, 1):
            page_text = page.extract_text() or ""
            text += f"\n--- Page {page_num} ---\n{page_text}"
        return text
    except:
        return ""

# Function to extract text from PDF images, focusing on page 2
def extract_text_from_images(file, target_page=2):
    try:
        images = convert_from_bytes(file.read())
        text = ""
        for page_num, image in enumerate(images, 1):
            if page_num == target_page and gemini_connected:
                response = model.generate_content(["Extract text from this image, focusing on any ingredient lists (e.g., INCI names) at the bottom of the page", image])
                page_text = response.text
                text += f"\n--- Page {page_num} ---\n{page_text}"
            else:
                text += f"\n--- Page {page_num} ---\n[Skipped; focusing on page 2]"
        return text
    except:
        return ""

# Function to sanitize JSON string
def sanitize_json_string(text):
    text = text.replace("'", '"')
    text = text.replace(",}", "}")
    text = text.replace(",]", "]")
    return text

# Function to summarize guidance using Gemini API
def summarize_guidance(text, context):
    if not gemini_connected:
        return "Guidance summarization unavailable without Gemini API."
    prompt = f"""
    Summarize the following {context} guidance text for display in a supplier compliance app.
    Focus on key requirements relevant to {context} compliance (e.g., substance restrictions, documentation needs).
    Keep the summary concise (max 100 words).
    Text: {text}
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return f"Failed to summarize {context} guidance."

# Function to translate and extract components using Gemini API
def translate_and_extract_components(text, filename):
    if not gemini_connected:
        return {
            "components": [],
            "translated_text": "",
            "raw_extracted_text": text,
            "notes": f"Extraction unavailable for {filename} without Gemini API",
            "scip_number": "",
            "country_origin": "",
            "taric_code": "",
            "safe_use": False,
            "compliance_docs": []
        }
    prompt = f"""
    You are processing a product datasheet named '{filename}', likely in Swedish, for a supplier compliance app, focusing on cosmetic products like sunscreens.
    Step 1: Translate the entire text to English, preserving all details, especially ingredient lists (INCI names like "Aqua", "Octocrylene", "Glycerin") at the bottom of page 2.
    Step 2: Extract a comprehensive list of all possible components, materials, ingredients, and packaging details.
    Prioritize:
    - Ingredients: INCI names from lists, tables, or images on page 2 (bottom). Look for terms like "Ingredients", "INCI", or lists of chemical names.
    - Packaging: Formats and sizes (e.g., "100 ml tube", "1 L cartridge").
    - Materials: General materials (e.g., "cream base", "plastic").
    - Components: Physical parts (e.g., "tube", "cartridge").
    Use industry knowledge to infer potential ingredients if not explicitly stated (e.g., sunscreens may contain UV filters like "Octocrylene", "Avobenzone", "Titanium Dioxide").
    Also extract:
    - SCIP number (if present).
    - Country of origin.
    - TARIC code.
    - Safe use information (e.g., usage instructions).
    - Compliance documents or references (e.g., safety data sheet, certificates), each as a dictionary with a "name" key (e.g., {{"name": "Safety Data Sheet"}}).
    For each component, include:
    - name: (e.g., "Aqua", "100 ml tube").
    - type: ("Ingredient", "Material", "Component", "Packaging").
    - source: ("Explicit" for mentioned, "Inferred" for implied).
    Return a JSON object with:
    - translated_text: Full translated English text, separated by page.
    - components: List of extracted items.
    - raw_extracted_text: Original input text for debugging.
    - notes: Extraction issues or inferences (e.g., "No ingredients list found on page 2; check SDS").
    - scip_number: Extracted SCIP number or empty string.
    - country_origin: Extracted country or empty string.
    - taric_code: Extracted TARIC code or empty string.
    - safe_use: Boolean indicating if safe use info is provided.
    - compliance_docs: List of dictionaries with "name" key.
    Ensure all string values use double quotes to avoid JSON parsing errors.
    Text: {text}
    """
    try:
        response = model.generate_content(prompt)
        sanitized_text = sanitize_json_string(response.text.strip("```json\n").strip("```"))
        result = json.loads(sanitized_text)
        if not any(c["type"] == "Ingredient" for c in result.get("components", [])):
            notes = result.get("notes", "")
            result["notes"] = f"{notes}; No ingredients list found in {filename}, including page 2; consider uploading full PDF or safety data sheet (SDS)"
        # Ensure compliance_docs contains dictionaries
        if "compliance_docs" in result:
            result["compliance_docs"] = [
                {"name": doc} if isinstance(doc, str) else doc 
                for doc in result["compliance_docs"]
            ]
        result["raw_extracted_text"] = text
        return result
    except Exception as e:
        st.session_state.data["extraction_issues"].append(f"JSON parsing error for {filename}: {str(e)}. Raw response: {response.text}")
        return {
            "components": [],
            "translated_text": "",
            "raw_extracted_text": text,
            "notes": f"Failed to extract components from {filename} due to JSON parsing error: {str(e)}",
            "scip_number": "",
            "country_origin": "",
            "taric_code": "",
            "safe_use": False,
            "compliance_docs": []
        }

# Function to validate substances against REACH and RoHS
def validate_substances(components, candidate_list_df):
    validations = []
    reach_substances = []
    rohs_substances = []
    
    candidate_names = candidate_list_df.get("Substance Name", pd.Series([])).str.lower().tolist() if candidate_list_df is not None else []
    rohs_names = [s["name"].lower() for s in ROHS_SUBSTANCES]
    
    for component in components:
        name = component.get("name", "").lower()
        comp_type = component.get("type", "").lower()
        validation = ""
        
        if comp_type == "ingredient" or comp_type == "substance":
            if candidate_names and name in candidate_names:
                validation = "Valid REACH candidate list substance"
                reach_substances.append(component)
            elif name in rohs_names:
                validation = f"RoHS restricted substance (threshold: {next(s['threshold'] for s in ROHS_SUBSTANCES if s['name'].lower() == name)})"
                rohs_substances.append(component)
            else:
                validation = "Warning: Substance/Ingredient not found in REACH candidate list or RoHS restricted list"
        else:
            validation = f"{comp_type.capitalize()}; no REACH/RoHS validation required"
        
        validations.append(validation)
    
    return validations, reach_substances, rohs_substances

# Update results dataframe
def update_results_df():
    results = [
        {"Field": "Components/Ingredients/Packaging", "Value": json.dumps(st.session_state.data["components"]), 
         "Step": 1, "Source": "Document/User", "Validation": "Reviewed in Step 1b"},
        {"Field": "REACH Substances", "Value": json.dumps(st.session_state.data["reach_substances"]), 
         "Step": 2, "Source": "Validated", "Validation": "; ".join(validate_substances(st.session_state.data["components"], st.session_state.data["candidate_list"])[0])},
        {"Field": "RoHS Substances", "Value": json.dumps(st.session_state.data["rohs_substances"]), 
         "Step": 2, "Source": "Validated", "Validation": "; ".join(["RoHS validated" for _ in st.session_state.data["rohs_substances"]])},
        {"Field": "SCIP Number", "Value": st.session_state.data["scip_number"], 
         "Step": 4, "Source": "Document/User", "Validation": "Required if REACH substances present" if st.session_state.data["reach_substances"] and not st.session_state.data["scip_number"] else "Valid"},
        {"Field": "Country of Origin", "Value": st.session_state.data["country_origin"], 
         "Step": 5, "Source": "Document/User", "Validation": "Required if REACH substances present" if st.session_state.data["reach_substances"] and not st.session_state.data["country_origin"] else "Valid"},
        {"Field": "TARIC Code", "Value": st.session_state.data["taric_code"], 
         "Step": 6, "Source": "Document/User", "Validation": "Required if REACH substances present" if st.session_state.data["reach_substances"] and not st.session_state.data["taric_code"] else "Valid"},
        {"Field": "Safe Use Provided", "Value": str(st.session_state.data["safe_use"]), 
         "Step": 7, "Source": "Document/User", "Validation": "Required if REACH substances present" if st.session_state.data["reach_substances"] and not (st.session_state.data["safe_use"] or st.session_state.data["safe_use_doc"]) else "Valid"},
        {"Field": "Safe Use Document", "Value": st.session_state.data["safe_use_doc"] or "None", 
         "Step": 7, "Source": "User", "Validation": "Optional"},
        {"Field": "Compliance Documents", "Value": json.dumps(st.session_state.data["compliance_docs"]), 
         "Step": 7, "Source": "Document", "Validation": "Valid"}
    ]
    st.session_state.results_df = pd.DataFrame(results)

# Main app
st.title("Supplier Module for REACH & RoHS Compliance")

# Diagnostics
st.header("Diagnostics")
if gemini_connected:
    st.success("Gemini API key is set and connection established.")
    if initial_response:
        st.success("Gemini API test response received successfully.")
else:
    st.error("Gemini API is not available. Extraction will be limited.")
if st.session_state.data["reach_guidance"]:
    st.success("REACH guidance PDF processed successfully.")
if st.session_state.data["rohs_guidance"]:
    st.success("RoHS guidance PDF processed successfully.")
if st.session_state.data["candidate_list"] is not None:
    st.success("REACH candidate list XLSX loaded successfully.")
if st.session_state.data["extraction_issues"]:
    st.warning("Extraction issues detected:")
    for issue in st.session_state.data["extraction_issues"]:
        st.write(f"- {issue}")

# Raw Extracted Text for Debugging
if st.session_state.data["raw_extracted_texts"]:
    with st.expander("Raw Extracted Text (Debug)", expanded=False):
        for filename, text in st.session_state.data["raw_extracted_texts"].items():
            st.subheader(f"Raw Text: {filename}")
            st.text(text if text else "No text extracted.")

# Results Table
st.header("Results Overview (Editable)")
update_results_df()
edited_df = st.data_editor(
    st.session_state.results_df,
    column_config={
        "Field": st.column_config.TextColumn("Field", disabled=True),
        "Value": st.column_config.TextColumn("Value"),
        "Step": st.column_config.NumberColumn("Step", disabled=True),
        "Source": st.column_config.TextColumn("Source", disabled=True),
        "Validation": st.column_config.TextColumn("Validation", disabled=True)
    },
    hide_index=True
)

# Update session state from edited dataframe
for _, row in edited_df.iterrows():
    field = row["Field"]
    value = row["Value"]
    if field == "Components/Ingredients/Packaging":
        try:
            st.session_state.data["components"] = json.loads(value)
        except:
            pass
    elif field == "REACH Substances":
        try:
            st.session_state.data["reach_substances"] = json.loads(value)
        except:
            pass
    elif field == "RoHS Substances":
        try:
            st.session_state.data["rohs_substances"] = json.loads(value)
        except:
            pass
    elif field == "SCIP Number":
        st.session_state.data["scip_number"] = value
    elif field == "Country of Origin":
        st.session_state.data["country_origin"] = value
    elif field == "TARIC Code":
        st.session_state.data["taric_code"] = value
    elif field == "Safe Use Provided":
        st.session_state.data["safe_use"] = value.lower() == "true"
    elif field == "Safe Use Document":
        st.session_state.data["safe_use_doc"] = value if value != "None" else None
    elif field == "Compliance Documents":
        try:
            st.session_state.data["compliance_docs"] = json.loads(value)
            st.session_state.data["compliance_docs"] = [
                {"name": doc} if isinstance(doc, str) else doc 
                for doc in st.session_state.data["compliance_docs"]
            ]
        except:
            pass

# Steps
st.header("Process Steps")
with st.expander("Step 1: Upload Guidance and Verification Documents", expanded=True):
    st.subheader("Guidance Documents")
    reach_pdf = st.file_uploader("Upload REACH Guidance PDF", type=["pdf"], key="reach_pdf")
    rohs_pdf = st.file_uploader("Upload RoHS Guidance PDF", type=["pdf"], key="rohs_pdf")
    candidate_xlsx = st.file_uploader("Upload REACH Candidate List XLSX", type=["xlsx"], key="candidate_xlsx")
    
    if reach_pdf:
        reach_text = extract_pdf_text(reach_pdf)
        st.session_state.data["reach_guidance"] = summarize_guidance(reach_text, "REACH")
        st.success("REACH guidance uploaded and summarized.")
    
    if rohs_pdf:
        rohs_text = extract_pdf_text(rohs_pdf)
        st.session_state.data["rohs_guidance"] = summarize_guidance(rohs_text, "RoHS")
        st.success("RoHS guidance uploaded and summarized.")
    
    if candidate_xlsx:
        try:
            st.session_state.data["candidate_list"] = pd.read_excel(candidate_xlsx)
            st.success("REACH candidate list loaded.")
        except:
            st.error("Failed to load candidate list XLSX.")
    
    st.subheader("Verification Documents")
    uploaded_files = st.file_uploader("Upload verification PDFs", type=["pdf"], accept_multiple_files=True, key="uploader")
    if uploaded_files:
        st.session_state.data["documents"] = []
        st.session_state.data["components"] = []
        st.session_state.data["extraction_issues"] = []
        st.session_state.data["translated_texts"] = {}
        st.session_state.data["raw_extracted_texts"] = {}
        for file in uploaded_files:
            file.seek(0)
            text = extract_pdf_text(file)
            extraction_method = "text"
            if not text or "de kropprder" in text or "stokoderm" in file.name.lower():
                file.seek(0)
                text = extract_text_from_images(file, target_page=2)
                extraction_method = "image"
            extracted_data = translate_and_extract_components(text, file.name)
            st.session_state.data["documents"].append({
                "file_name": file.name,
                "extracted_data": extracted_data,
                "extraction_method": extraction_method
            })
            st.session_state.data["translated_texts"][file.name] = extracted_data.get("translated_text", "")
            st.session_state.data["raw_extracted_texts"][file.name] = extracted_data.get("raw_extracted_text", "")
            st.session_state.data["components"].extend(extracted_data.get("components", []))
            if extracted_data.get("notes"):
                st.session_state.data["extraction_issues"].append(f"{file.name} ({extraction_method}): {extracted_data['notes']}")
            if "scip_number" in extracted_data:
                st.session_state.data["scip_number"] = extracted_data["scip_number"]
            if "country_origin" in extracted_data:
                st.session_state.data["country_origin"] = extracted_data["country_origin"]
            if "taric_code" in extracted_data:
                st.session_state.data["taric_code"] = extracted_data["taric_code"]
            if "safe_use" in extracted_data:
                st.session_state.data["safe_use"] = extracted_data["safe_use"]
            if "compliance_docs" in extracted_data:
                st.session_state.data["compliance_docs"].extend([
                    {"name": doc} if isinstance(doc, str) else doc 
                    for doc in extracted_data["compliance_docs"]
                ])
        st.success(f"{len(uploaded_files)} verification documents uploaded and processed!")
        if st.session_state.data["extraction_issues"]:
            st.warning("Issues detected during extraction. The ingredients list may be in an image, table, or unprovided page. Consider uploading the full PDF or safety data sheet (SDS) from https://www.scip.com/sv-se/sakerhetsdatablad.")

with st.expander("Step 1b: Review and Edit Components/Ingredients/Packaging"):
    st.subheader("Extracted Components, Ingredients, and Packaging")
    components_df = pd.DataFrame(st.session_state.data["components"])
    if not components_df.empty:
        edited_components = st.data_editor(
            components_df,
            column_config={
                "name": st.column_config.TextColumn("Name"),
                "type": st.column_config.SelectboxColumn(
                    "Type",
                    options=["Ingredient", "Material", "Component", "Packaging"]
                ),
                "source": st.column_config.TextColumn("Source", disabled=True)
            },
            hide_index=True,
            num_rows="dynamic"
        )
        st.session_state.data["components"] = edited_components.to_dict("records")
    
    st.subheader("Add New Component/Ingredient/Packaging")
    col1, col2, col3 = st.columns(3)
    new_name = col1.text_input("Name", key="new_comp_name")
    new_type = col2.selectbox("Type", options=["Ingredient", "Material", "Component", "Packaging"], key="new_comp_type")
    if col3.button("Add"):
        if new_name:
            st.session_state.data["components"].append({
                "name": new_name,
                "type": new_type,
                "source": "User"
            })
            st.success(f"Added {new_name} as {new_type}")
    
    if st.button("Confirm Components and Validate"):
        validations, reach_substances, rohs_substances = validate_substances(
            st.session_state.data["components"], st.session_state.data["candidate_list"]
        )
        st.session_state.data["reach_substances"] = reach_substances
        st.session_state.data["rohs_substances"] = rohs_substances
        st.session_state.data["components"] = [
            {**comp, "validation": val} for comp, val in zip(st.session_state.data["components"], validations)
        ]
        st.success("Components/Ingredients validated against REACH and RoHS lists.")
        update_results_df()

with st.expander("Step 2: Review Validated Substances"):
    if st.session_state.data["reach_guidance"]:
        st.markdown(f"**REACH Guidance**: {st.session_state.data['reach_guidance']}")
    if st.session_state.data["rohs_guidance"]:
        st.markdown(f"**RoHS Guidance**: {st.session_state.data['rohs_guidance']}")
    
    st.subheader("REACH Candidate List Substances")
    for idx, substance in enumerate(st.session_state.data["reach_substances"]):
        col1, col2 = st.columns(2)
        name = col1.text_input("Substance Name", value=substance.get("name", ""), key=f"reach_name_{idx}", disabled=True)
        conc = col2.text_input("Concentration", value=substance.get("concentration", ""), key=f"reach_conc_{idx}")
        st.session_state.data["reach_substances"][idx]["concentration"] = conc
    
    st.subheader("RoHS Substances")
    for idx, substance in enumerate(st.session_state.data["rohs_substances"]):
        col1, col2 = st.columns(2)
        name = col1.text_input("Substance Name", value=substance.get("name", ""), key=f"rohs_name_{idx}", disabled=True)
        conc = col2.text_input("Concentration", value=substance.get("concentration", ""), key=f"rohs_conc_{idx}")
        st.session_state.data["rohs_substances"][idx]["concentration"] = conc
    update_results_df()

with st.expander("Step 3: Substance Details"):
    if st.session_state.data["reach_guidance"]:
        st.markdown(f"**REACH Guidance**: {st.session_state.data['reach_guidance']}")
    for idx, substance in enumerate(st.session_state.data["reach_substances"] + st.session_state.data["rohs_substances"]):
        location = st.text_input(f"Location in product for {substance.get('name')}", key=f"loc_{idx}")
        if "location" not in substance:
            substance["location"] = location
        else:
            substance["location"] = location
    update_results_df()

with st.expander("Step 4: SCIP Number (if applicable)"):
    if st.session_state.data["reach_guidance"]:
        st.markdown(f"**REACH Guidance**: {st.session_state.data['reach_guidance']}")
    scip = st.text_input("SCIP Number", value=st.session_state.data["scip_number"], key="scip")
    st.session_state.data["scip_number"] = scip
    if st.session_state.data["reach_substances"] and not scip:
        st.warning("SCIP number is required for EU suppliers with candidate list substances.")
    update_results_df()

with st.expander("Step 5: Country of Origin"):
    if st.session_state.data["reach_guidance"]:
        st.markdown(f"**REACH Guidance**: {st.session_state.data['reach_guidance']}")
    country = st.text_input("Country of Origin", value=st.session_state.data["country_origin"], key="country")
    st.session_state.data["country_origin"] = country
    if st.session_state.data["reach_substances"] and not country:
        st.warning("Country of origin is required for products with candidate list substances.")
    update_results_df()

with st.expander("Step 6: TARIC Code"):
    if st.session_state.data["reach_guidance"]:
        st.markdown(f"**REACH Guidance**: {st.session_state.data['reach_guidance']}")
    taric = st.text_input("TARIC Code", value=st.session_state.data["taric_code"], key="taric")
    st.session_state.data["taric_code"] = taric
    if st.session_state.data["reach_substances"] and not taric:
        st.warning("TARIC code is required for products with candidate list substances.")
    update_results_df()

with st.expander("Step 7: Verification Documents"):
    if st.session_state.data["reach_guidance"] or st.session_state.data["rohs_guidance"]:
        st.markdown(f"**REACH Guidance**: {st.session_state.data['reach_guidance']}")
        st.markdown(f"**RoHS Guidance**: {st.session_state.data['rohs_guidance']}")
    
    st.subheader("Uploaded Compliance Documents")
    for doc in st.session_state.data["compliance_docs"]:
        doc_name = doc.get("name", doc) if isinstance(doc, dict) else doc
        st.write(f"- {doc_name}")
    
    st.subheader("Safe Use Information")
    safe_use = st.checkbox("Safe use information provided", value=st.session_state.data["safe_use"], key="safe_use")
    safe_use_doc = st.file_uploader("Upload safe use document (optional)", type=["pdf"], key="safe_use_doc")
    if safe_use_doc:
        st.session_state.data["safe_use_doc"] = safe_use_doc.name
    st.session_state.data["safe_use"] = safe_use
    if st.session_state.data["reach_substances"] and not (safe_use or st.session_state.data["safe_use_doc"]):
        st.warning("Safe use information is required for candidate list substances.")
    update_results_df()

with st.expander("Step 8: Under Construction"):
    st.write("This step is not yet implemented.")

with st.expander("Step 9: Under Construction"):
    st.write("This step is not yet implemented.")

with st.expander("Step 10: Under Construction"):
    st.write("This step is not yet implemented.")
    if st.button("Finish"):
        st.success("Process completed! Data saved in results table.")
        update_results_df()