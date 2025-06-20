import streamlit as st
import os
import PyPDF2
from PyPDF2.errors import PdfReadError
import re
from io import BytesIO
import time

# ReportLab Imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet  # Corrected: singular
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# LangChain imports
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# CORRECTED IMPORT FOR CHROMA
try:
    from langchain.vectorstores import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        st.error(
            "Could not import Chroma from langchain.vectorstores or langchain_community.vectorstores. "
            "Please ensure 'langchain' and 'chromadb' are installed: 'pip install langchain chromadb'"
        )
        st.stop()  # Stop Streamlit app if critical import fails

from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# HTML Escaping for ReportLab
import html as html_escaper


# --- GeminiPDFProcessor Class ---
class GeminiPDFProcessor:
    """
    A class to extract text from PDFs and process it with the Google Gemini API,
    supporting RAG for verification against multiple documents with persistent ChromaDB.
    """

    def __init__(self, api_key=None, model_name='gemini-2.0-flash', embedding_model_name='models/embedding-001',
                 chroma_db_dir="./reach_chroma_db"):
        """
        Initializes the GeminiPDFProcessor.
        """
        if api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Gemini API key not provided. "
                    "Please set the GEMINI_API_KEY environment variable."
                )
        else:
            self.api_key = api_key

        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        self.chroma_db_dir = chroma_db_dir

        # Initialize the generation model (LLM) using LangChain's wrapper
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.0  # Good for fact-checking/summarization for more deterministic output
        )

        # Initialize the embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name, google_api_key=self.api_key)

        # Default safety settings - LangChain's wrapper handles this, but keep for direct genai calls if any
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def _extract_text_from_pdf_stream(self, pdf_file_stream):
        """
        Extracts text from a PDF file stream (from st.file_uploader).
        Returns the full text content of the PDF.
        """
        pages_text = []
        try:
            reader = PyPDF2.PdfReader(pdf_file_stream)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:
                    pages_text.append(text)
        except PdfReadError as e:
            st.error(f"Error reading PDF file (might be corrupted or encrypted): {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred during PDF text extraction: {e}")
            return None
        return "\n".join(pages_text)

    @st.cache_resource  # Cache the vector database creation/loading
    def _create_or_load_vector_db(_self, pdf_paths, collection_name="reach_legislation_db_collection"):
        """
        Creates a ChromaDB vector store from multiple PDF documents, or loads it if it already exists.
        _self is used to access instance attributes in st.cache_resource.
        """
        st.info("Checking for existing REACH knowledge base...")
        db_directory_exists = os.path.exists(_self.chroma_db_dir) and any(
            os.path.isfile(os.path.join(_self.chroma_db_dir, f)) for f in os.listdir(_self.chroma_db_dir))

        if db_directory_exists:
            try:
                db = Chroma(
                    persist_directory=_self.chroma_db_dir,
                    embedding_function=_self.embeddings,
                    collection_name=collection_name
                )

                import chromadb
                client = chromadb.PersistentClient(path=_self.chroma_db_dir)
                loaded_collection = client.get_or_create_collection(name=collection_name)

                if loaded_collection.count() > 0:
                    st.success(f"Loaded existing REACH knowledge base with {loaded_collection.count()} documents.")
                    return db
                else:
                    st.warning(
                        f"Existing DB directory found, but collection '{collection_name}' is empty. Rebuilding...")
            except Exception as e:
                st.error(f"Error loading existing database: {e}. Attempting to rebuild.")

        st.info(f"Building new REACH knowledge base from {len(pdf_paths)} documents...")
        all_texts = []
        for i, pdf_path in enumerate(pdf_paths):
            st.write(f"  Extracting text from '{pdf_path}' (Document {i + 1}/{len(pdf_paths)})")
            doc_text = _self._extract_text_from_pdf_stream(open(pdf_path, 'rb'))
            if doc_text:
                all_texts.append(doc_text)
            else:
                st.warning(f"  Skipping '{pdf_path}' due to extraction failure.")

        if not all_texts:
            st.error("No text extracted from any REACH PDF documents to build the knowledge base.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.create_documents(all_texts)
        st.write(f"  Split documents into {len(chunks)} chunks.")

        with st.spinner("Embedding and storing documents in ChromaDB..."):
            db = Chroma.from_documents(
                chunks,
                _self.embeddings,
                collection_name=collection_name,
                persist_directory=_self.chroma_db_dir
            )
        st.success(f"REACH knowledge base built and persisted successfully.")
        return db

    def _translate_text(self, text, target_language='en'):
        """Translates text using the LLM."""
        if not text:
            return text
        translation_prompt = f"Translate the following term into {target_language}: '{text}'"
        try:
            translation_response = self.llm.invoke(translation_prompt)
            return translation_response.content.strip()
        except Exception as e:
            st.warning(f"Translation failed for '{text}': {e}. Using original name.")
            return text  # Fallback to original if translation fails

    def extract_components(self, product_sheet_pdf_stream):
        """
        Extracts described and probable components from a product sheet PDF,
        and translates them to English.
        """
        st.info("Extracting components from product sheet...")
        product_sheet_text = self._extract_text_from_pdf_stream(product_sheet_pdf_stream)
        if product_sheet_text is None:
            return []

        component_extraction_prompt = (
            "From the following product description, identify and list all described components "
            "and any probable components that would typically be part of such a product. "
            "Present them as a comma-separated list of items. "
            "Example: 'Stainless steel housing, mixing bowl, LCD screen, digital timer, motor, compressor, power cord'."
            "\n\n--- Product Description Start ---\n"
            f"{product_sheet_text}"
            "\n--- Product Description End ---"
        )

        try:
            with st.spinner("Asking Gemini to identify components..."):
                component_response = self.llm.invoke(component_extraction_prompt)
            extracted_components_raw = component_response.content
        except Exception as e:
            st.error(f"Error extracting components: {e}")
            st.warning(f"API Error Details: {e}")
            return []

        if extracted_components_raw is None:
            return []

        components = [
            comp.strip() for comp in extracted_components_raw.split(',') if comp.strip()
        ]

        # Translate components
        translated_components = []
        with st.spinner("Translating components to English..."):
            for comp_original in components:
                comp_english = self._translate_text(comp_original, target_language='English')
                translated_components.append({
                    'original': comp_original,
                    'english': comp_english
                })

        st.success(f"Identified and translated {len(translated_components)} components.")
        return translated_components

    def verify_component_reach(self, component_name_english, reach_retriever):
        """
        Verifies how REACH legislation applies to a single component using RAG.
        Uses the English name for verification.
        """
        reach_verification_prompt_template = PromptTemplate.from_template(
            """You are an expert on EU REACH legislation. Given the following context documents about REACH:
            {context}

            Please explain how the REACH legislation applies to the following component, or if it is not directly addressed,
            discuss general principles of REACH that might be relevant. Be concise and focus on direct applicability.
            If the context does not contain enough information to make a direct statement, indicate that.

            Component: '{input}'
            """
        )

        combine_reach_docs_chain = create_stuff_documents_chain(self.llm, reach_verification_prompt_template)
        reach_retrieval_chain = create_retrieval_chain(reach_retriever, combine_reach_docs_chain)

        try:
            response = reach_retrieval_chain.invoke({"input": component_name_english})
            return response["answer"]
        except Exception as e:
            st.error(f"Error during REACH verification for component '{component_name_english}': {e}")
            return "Failed to determine applicability due to error."


# --- Streamlit App ---

st.set_page_config(
    page_title="REACH Compliance Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 REACH Compliance Assistant")
st.markdown("Upload a product sheet PDF, identify components, and verify their compliance against REACH legislation.")

# --- Initialize Session State ---
if 'extracted_components' not in st.session_state:
    st.session_state.extracted_components = []  # List of {'original': ..., 'english': ...} dicts
if 'reach_verification_results' not in st.session_state:
    st.session_state.reach_verification_results = []  # List of {'component_original': ..., 'component_english': ..., 'applicability': ...}
if 'product_pdf_processed' not in st.session_state:
    st.session_state.product_pdf_processed = False
if 'reach_db_ready' not in st.session_state:
    st.session_state.reach_db_ready = False


# --- Instantiate Processor (once) ---
@st.cache_resource
def get_processor():
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY environment variable not set. Please set it and rerun the app.")
            st.stop()
        return GeminiPDFProcessor(model_name='gemini-2.0-flash')  # Using flash as requested
    except ValueError as e:
        st.error(f"Initialization error: {e}")
        st.stop()


processor = get_processor()

# --- Load REACH Knowledge Base (once) ---
reach_guidance_pdf_path = "REACH guidance document.pdf"
reach_regulation_pdf_path = "CELEX_32006R1907_EN_TXT.pdf"
reach_documents_for_rag = [reach_guidance_pdf_path, reach_regulation_pdf_path]

missing_reach_files = [f for f in reach_documents_for_rag if not os.path.exists(f)]
if missing_reach_files:
    st.error(
        f"Error: The following REACH legislation PDF files are missing from the app directory: {', '.join(missing_reach_files)}")
    st.info("Please ensure these files are present to build the REACH knowledge base.")
    st.stop()

reach_vector_db = processor._create_or_load_vector_db(reach_documents_for_rag)
if reach_vector_db:
    reach_retriever = reach_vector_db.as_retriever(search_kwargs={"k": 5})
    st.session_state.reach_db_ready = True
else:
    st.session_state.reach_db_ready = False
    st.warning("REACH knowledge base could not be built. Verification will not be possible.")

# --- Product PDF Upload Section ---
st.header("1. Upload Product Sheet PDF")
uploaded_product_pdf = st.file_uploader("Choose a product PDF file", type="pdf")

if uploaded_product_pdf and not st.session_state.product_pdf_processed:
    st.session_state.extracted_components = processor.extract_components(uploaded_product_pdf)
    st.session_state.product_pdf_processed = True
    st.session_state.reach_verification_results = []  # Clear previous results

if st.session_state.product_pdf_processed:
    st.header("2. Review & Edit Components")
    st.write("Here's the list of components identified. You can add or remove items.")

    # Display components with remove buttons
    # Iterate over a copy to allow modification during iteration
    current_components_list = st.session_state.extracted_components[:]
    for i, component_data in enumerate(current_components_list):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            display_name = f"{component_data['original']} (English: {component_data['english']})" if component_data[
                                                                                                         'original'] != \
                                                                                                     component_data[
                                                                                                         'english'] else \
            component_data['original']
            st.write(f"- {display_name}")
        with col2:
            if st.button(f"Remove", key=f"remove_comp_{i}"):
                st.session_state.extracted_components.pop(i)  # Remove by index
                st.session_state.reach_verification_results = []  # Clear results if list changes
                st.rerun()  # Rerun to update the displayed list

    # Add new component input
    with st.form("add_component_form"):
        new_component_original_input = st.text_input("Enter new component (original language):",
                                                     key="add_comp_original_input")
        add_button_clicked = st.form_submit_button("Add Component")
        if add_button_clicked:
            if new_component_original_input and new_component_original_input.strip():
                # Check if component (by original name) already exists
                existing_original_names = [c['original'].lower() for c in st.session_state.extracted_components]
                if new_component_original_input.strip().lower() in existing_original_names:
                    st.warning("Component with this original name already in list!")
                else:
                    with st.spinner(f"Adding and translating '{new_component_original_input}'..."):
                        new_component_english = processor._translate_text(new_component_original_input.strip(),
                                                                          target_language='English')
                        st.session_state.extracted_components.append({
                            'original': new_component_original_input.strip(),
                            'english': new_component_english
                        })
                    st.session_state.reach_verification_results = []
                    st.rerun()
            else:
                st.warning("Please enter a component name.")

    # --- REACH Verification Section ---
    st.header("3. Verify REACH Applicability")
    if st.session_state.reach_db_ready and st.session_state.extracted_components:
        if st.button("Start REACH Verification", key="start_reach_verification"):
            st.session_state.reach_verification_results = []  # Reset results before starting
            st.subheader("REACH Verification Details:")

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, component_data in enumerate(st.session_state.extracted_components):
                original_name = component_data['original']
                english_name = component_data['english']

                status_text.text(
                    f"Verifying: {original_name} (English: {english_name}) ({i + 1}/{len(st.session_state.extracted_components)})")
                with st.spinner(f"Processing '{english_name}'..."):
                    # Use the English name for the RAG query
                    result = processor.verify_component_reach(english_name, reach_retriever)

                st.session_state.reach_verification_results.append({
                    "component_original": original_name,
                    "component_english": english_name,
                    "applicability": result
                })
                # Display result dynamically in the UI
                with st.expander(f"**Component: {original_name} (English: {english_name})**"):
                    st.markdown(result)  # Use st.markdown to render text and basic markdown

                progress_bar.progress((i + 1) / len(st.session_state.extracted_components))

            st.success("REACH Verification Complete!")
            progress_bar.empty()  # Remove progress bar after completion
            status_text.empty()  # Remove status text

        # No separate "REACH Verification Summary" section is needed, as it's shown dynamically above.

    elif not st.session_state.extracted_components:
        st.info("Upload a product sheet and identify components to start REACH verification.")
    elif not st.session_state.reach_db_ready:
        st.warning("REACH knowledge base is not ready. Please ensure REACH PDFs are present and try again.")

    # --- Generate PDF Report Section ---
    st.header("4. Generate & Download Report")
    if st.session_state.reach_verification_results:
        # Generate PDF report
        def generate_pdf_report(results_data, extracted_components_list):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            # Custom style for report body text to handle newlines and basic HTML/Markdown conversion
            report_body_style = styles['Normal']
            report_body_style.leading = 14  # Line spacing
            report_body_style.wordWrap = 'CJK'  # Enable word wrap for non-latin characters too (general purpose)

            story = []

            # Title
            story.append(Paragraph("REACH Compliance Report", styles['h1']))
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 0.4 * inch))

            story.append(Paragraph("Extracted Components:", styles['h2']))

            # Display original and translated components in report
            for comp_data in extracted_components_list:
                display_name = f"{comp_data['original']} (English: {comp_data['english']})" if comp_data['original'] != \
                                                                                               comp_data['english'] else \
                comp_data['original']
                story.append(Paragraph(f"• {display_name}", styles['Normal']))
            story.append(Spacer(1, 0.4 * inch))

            story.append(Paragraph("REACH Verification Results:", styles['h2']))
            story.append(Spacer(1, 0.2 * inch))

            for result in results_data:
                display_component_name = f"<b>{result['component_original']}</b> (English: {result['component_english']})" if \
                result['component_original'] != result[
                    'component_english'] else f"<b>{result['component_original']}</b>"
                story.append(Paragraph(display_component_name, styles['h3']))

                # --- Robust Markdown to HTML conversion for ReportLab ---
                formatted_applicability = result['applicability']

                # 1. Escape HTML special characters that might be in the raw text
                formatted_applicability = html_escaper.escape(formatted_applicability)

                # 2. Convert Markdown bold (**text**) to HTML <b>text</b>
                # Use a non-greedy regex to prevent overlapping matches
                formatted_applicability = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_applicability)

                # 3. Convert Markdown italics (*text*) to HTML <i>text</i> (if needed)
                # Use word boundaries (\b) to avoid matching asterisks in the middle of words
                formatted_applicability = re.sub(r'\b\*(.*?)\*\b', r'<i>\1</i>',
                                                 formatted_applicability)  # For *italic*

                # 4. Convert bullet points (- item or * item or • item)
                # This is tricky because ReportLab's Paragraph doesn't inherently render list markup.
                # Simplest is to replace bullet markers with actual bullet character and line breaks.
                formatted_applicability = re.sub(r'^[*-]\s*', r'• ', formatted_applicability,
                                                 flags=re.MULTILINE)  # Handle initial bullets
                formatted_applicability = formatted_applicability.replace('\n\n• ',
                                                                          '<br/><br/>• ')  # Ensure spacing for subsequent list items
                formatted_applicability = formatted_applicability.replace('\n• ',
                                                                          '<br/>• ')  # Ensure spacing for subsequent list items

                # 5. Convert Newlines to HTML line breaks AFTER other formatting
                formatted_applicability = formatted_applicability.replace('\n', '<br/>')

                story.append(Paragraph(formatted_applicability, report_body_style))
                story.append(Spacer(1, 0.2 * inch))

            doc.build(story)
            buffer.seek(0)
            return buffer


        pdf_buffer = generate_pdf_report(st.session_state.reach_verification_results,
                                         st.session_state.extracted_components)
        st.download_button(
            label="Download REACH Report (PDF)",
            data=pdf_buffer,
            file_name="reach_compliance_report.pdf",
            mime="application/pdf"
        )
    else:
        st.info("Complete the REACH verification process to generate a report.")