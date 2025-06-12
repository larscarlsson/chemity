import streamlit as st
import pandas as pd


# Placeholder mappings for demonstration
TARIC_CODES = {
    "battery": "8507 60 00 00",
    "furniture": "9403 10 00 00",
    "phone": "8517 12 00 10",
}
SCIP_SUBSTANCES = {
    "lead": "SVHC: Lead (Example)",
    "mercury": "SVHC: Mercury (Example)",
    "cadmium": "SVHC: Cadmium (Example)",
}

st.title("CHEMITY Data Extraction and Classification Assessment")
st.write("Upload one or more files (PDF, Excel, CSV, or text) for analysis:")

uploaded_files = st.file_uploader("Choose files", type=["pdf", "xlsx", "xls", "csv", "txt"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Please upload at least one file to begin.")
else:
    results = []
    all_file_matches = {}  # store matches per file for feedback
    for file in uploaded_files:
        file_name = file.name
        st.subheader(f"Results for `{file_name}`")
        text = ""
        try:
            if file_name.lower().endswith(".pdf"):
                doc = pymupdf.open(stream=file.read(), filetype="pdf")
                text = "".join(page.get_text() for page in doc)  # extract all pages text
                doc.close()
            elif file_name.lower().endswith((".xls", ".xlsx")):
                file.seek(0)
                df = pd.read_excel(file)
                text = df.astype(str).to_csv(index=False)
            elif file_name.lower().endswith(".csv"):
                file.seek(0)
                df = pd.read_csv(file)
                text = df.astype(str).to_csv(index=False)
            else:
                text = file.getvalue().decode("utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Error processing {file_name}: {e}")
            continue

        # Identify relevant items via keywords
        extracted_items = set()
        matches = []
        text_lower = text.lower()
        for keyword, code in TARIC_CODES.items():
            if keyword in text_lower:
                extracted_items.add(keyword)
                matches.append((keyword, code, "TARIC"))
        for substance, code in SCIP_SUBSTANCES.items():
            if substance in text_lower:
                extracted_items.add(substance)
                matches.append((substance, code, "SCIP"))

        total_items = len(extracted_items)
        matched_items = len(matches)
        coverage = (matched_items / total_items * 100) if total_items > 0 else 0
        results.append({
            "file": file_name,
            "total_items": total_items,
            "matched_items": matched_items,
            "coverage_pct": coverage
        })
        all_file_matches[file_name] = matches

        st.write(f"**Extracted items**: " + (", ".join(extracted_items) if extracted_items else "_None_"))
        if matches:
            match_df = pd.DataFrame(matches, columns=["Item", "Matched Code", "Source"])
            st.write("**Matched Codes:**")
            st.table(match_df)
        else:
            st.write("**Matched Codes:** _No matches found_")
        st.write(f"**Coverage:** {coverage:.1f}%")
        st.progress(int(coverage))
        st.write("---")
    # End for each file

    if len(results) > 1:
        st.subheader("Overall Extraction Coverage")
        summary_df = pd.DataFrame(results).set_index("file")
        st.bar_chart(summary_df["coverage_pct"])
        st.table(summary_df[["total_items", "matched_items", "coverage_pct"]].rename(columns={
            "total_items": "Extracted Items", "matched_items": "Matched Items", "coverage_pct": "Coverage (%)"}))
        st.caption("Coverage = (Matched Items / Extracted Items) * 100")

    # Feedback section
    st.subheader("Feedback")
    st.write("Review the matches and provide feedback:")
    feedback_records = []
    for file_res in results:
        fname = file_res["file"]
        st.write(f"**{fname}**")
        matches = all_file_matches.get(fname, [])
        if matches:
            for (item, code, source) in matches:
                st.write(f"- *{item}* → `{code}` ({source})")
                fb = st.feedback("thumbs", key=f"{fname}_{item}")
                if fb is not None:
                    feedback_records.append({
                        "file": fname, "item": item, "code": code, "source": source,
                        "feedback": "correct" if fb == 1 else "incorrect"
                    })
        else:
            st.write("_No extracted matches to give feedback on._")

    # (Optionally, do something with feedback_records, e.g., log them or display them)
    if feedback_records:
        st.write("**Feedback received:**")
        st.write(pd.DataFrame(feedback_records))
