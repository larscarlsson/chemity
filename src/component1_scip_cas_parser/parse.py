import os
import re
import pandas as pd
from PyPDF2 import PdfReader
from pathlib import Path


'''
Notes: wurth har levererat massor av partnumbers med SCIP CAS och TARIC i en xls som är välformaterad.
Detta script letar efter SCIP nummer i PDFer i en angiven mapp.
'''

repo_path = Path(__file__).resolve().parents[2]
folder_path = repo_path / 'source_data/examples_chemity/Underlag'  # Replace with your folder path


def extract_table_data(text, scip_pattern, cas_pattern):
    """Extract potential SCIP and CAS numbers from table-like structures."""
    lines = text.split('\n')
    scip_numbers = []
    cas_numbers = []
    
    # Look for table headers
    for i, line in enumerate(lines):
        if re.search(r'\b[Ss][Cc][Ii][Pp]\b', line) and re.search(r'\b[Cc][Aa][Ss]\b', line):
            # Potential table header found, check subsequent lines
            for j in range(i + 1, min(i + 10, len(lines))):  # Look at next 10 lines
                line = lines[j].strip()
                if line:
                    # Try to find SCIP and CAS in the same line
                    scip_matches = re.findall(scip_pattern, line)
                    cas_matches = re.findall(cas_pattern, line)
                    scip_numbers.extend(scip_matches)
                    cas_numbers.extend(cas_matches)
    
    return scip_numbers, cas_numbers

def find_scip_and_cas_numbers(folder_path):
    # Initialize lists to store results
    filenames = []
    scip_numbers = []
    cas_numbers = []
    
    # Regex patterns
    scip_pattern = r'[Ss][Cc][Ii][Pp]\s*[-]?\s*\d+'
    cas_pattern = r'\b\d{2,7}-\d{2}-\d\b'
    
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Read PDF
                pdf = PdfReader(file_path)
                file_scip_numbers = []
                file_cas_numbers = []
                
                # Search each page
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    
                    # Extract from general text
                    scip_matches = re.findall(scip_pattern, text, re.IGNORECASE)
                    cas_matches = re.findall(cas_pattern, text)
                    file_scip_numbers.extend(scip_matches)
                    file_cas_numbers.extend(cas_matches)
                    
                    # Extract from potential tables
                    table_scip, table_cas = extract_table_data(text, scip_pattern, cas_pattern)
                    file_scip_numbers.extend(table_scip)
                    file_cas_numbers.extend(table_cas)
                
                # Store results
                filenames.append(filename)
                scip_numbers.append(', '.join(set(file_scip_numbers)) if file_scip_numbers else 'None')
                cas_numbers.append(', '.join(set(file_cas_numbers)) if file_cas_numbers else 'None')
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                filenames.append(filename)
                scip_numbers.append('Error')
                cas_numbers.append('Error')
    
    # Create DataFrame
    df = pd.DataFrame({
        'Filename': filenames,
        'SCIP_Number': scip_numbers,
        'CAS_Number': cas_numbers
    })
    
    return df

def main():    
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return
    
    # Get results
    df = find_scip_and_cas_numbers(folder_path)
    
    # Print results
    print("\nResults:")
    print(df)
    
    # Save to CSV
    df.to_csv('scip_and_cas_numbers.csv', index=False)
    print("\nResults saved to scip_and_cas_numbers.csv")

if __name__ == "__main__":
    main()