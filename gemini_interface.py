import google.generativeai as genai
import os
import PyPDF2
from io import BytesIO # Although not strictly used for file paths, good to keep in mind for in-memory PDFs


class GeminiPDFProcessor:
    """
    A class to extract text from PDFs and process it with the Google Gemini API.
    """

    def __init__(self, api_key=None, model_name='gemini-2.0-flash'):
        """
        Initializes the GeminiPDFProcessor.

        Args:
            api_key (str, optional): Your Google Gemini API key.
                                     If None, it will try to get it from the GEMINI_API_KEY environment variable.
            model_name (str): The name of the Gemini model to use (e.g., 'gemini-2.0-flash', 'gemini-2.0-pro').
        """
        if api_key is None:
            self.api_key = os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Gemini API key not provided. "
                    "Please pass it as an argument or set the GEMINI_API_KEY environment variable."
                )
        else:
            self.api_key = api_key

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

        # Default safety settings - adjust as needed
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

    def _extract_text_from_pdf(self, pdf_path):
        """
        Private method to extract text from a PDF file.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            str: The extracted text from the PDF, or None if an error occurs.
        """
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() or ""
        except PyPDF2.utils.PdfReadError as e:
            print(f"Error reading PDF file (might be corrupted or encrypted): {e}")
            return None
        except FileNotFoundError:
            print(f"Error: PDF file not found at {pdf_path}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during PDF text extraction: {e}")
            return None
        return text

    def process_pdf(self, pdf_path, prompt_instruction):
        """
        Extracts text from a PDF, combines it with a prompt, and sends to the Gemini model.

        Args:
            pdf_path (str): The path to the PDF file.
            prompt_instruction (str): The specific instruction for the LLM (e.g., "Summarize this document").

        Returns:
            str: The AI's response text, or None if an error occurs.
        """
        print(f"1. Extracting text from PDF: {pdf_path}")
        pdf_text = self._extract_text_from_pdf(pdf_path)

        if pdf_text is None:
            print("Failed to extract text from PDF. Aborting processing.")
            return None

        if not pdf_text.strip():
            print("Warning: Extracted PDF text is empty or only whitespace. The LLM might not have content to work with.")
            # You might choose to return None here or proceed with an empty document depending on desired behavior

        # Combine the instruction prompt and the PDF content
        full_prompt = f"{prompt_instruction}\n\n--- Document Start ---\n{pdf_text}\n--- Document End ---"

        print(f"\n2. Sending request to Gemini model ({self.model_name})...")
        print(f"   Prompt instruction: '{prompt_instruction}'")
        # print(f"   Full prompt (truncated for display):\n{full_prompt[:500]}...") # Optional: print full prompt truncated

        try:
            response = self.model.generate_content(
                full_prompt,
                safety_settings=self.safety_settings
            )

            print("\n3. Gemini Response:")
            response_text = response.text
            print(response_text)
            return response_text

        except Exception as e:
            print(f"\nError calling Gemini API: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"API Error Details: {e.response.text}")
            return None


# --- Example Usage ---
if __name__ == "__main__":
    pdf_name = "Foilboard-GreenRigid-Planels-DataSheet-2025.pdf"

    if pdf_name:
        pdf_file_to_process = pdf_name
    else:
        pdf_file_to_process = "your_actual_document.pdf" # Fallback if dummy not created
        print(f"\nUsing '{pdf_file_to_process}'. Please ensure this file exists.")


    # --- Instantiate the class ---
    # The API key will be read from the environment variable GEMINI_API_KEY
    # Or you can pass it directly: processor = GeminiPDFProcessor(api_key="YOUR_HARDCODED_API_KEY_HERE")
    try:
        processor = GeminiPDFProcessor(model_name='gemini-2.0-flash') # Or 'gemini-2.0-pro'
    except ValueError as e:
        print(e)
        print("Exiting. Please set your GEMINI_API_KEY environment variable.")
        exit()


    # --- Define your prompt instruction ---
    prompt_for_summary = "Summarize the key concepts of AI and Machine Learning as described in the document, in no more than 4 sentences."
    prompt_for_extraction = "Extract any specific definitions of AI or Machine Learning provided in the document. List them clearly."

    # --- Process the PDF with different prompts ---
    print("\n--- Running Summary Task ---")
    summary_result = processor.process_pdf(pdf_file_to_process, prompt_for_summary)
    if summary_result:
        print(f"\nTask Complete: Summary Generated.")

    print("\n\n--- Running Extraction Task ---")
    extraction_result = processor.process_pdf(pdf_file_to_process, prompt_for_extraction)
    if extraction_result:
        print(f"\nTask Complete: Extraction Done.")