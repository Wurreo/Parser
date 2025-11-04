import os
import shutil
import asyncio
import pandas as pd
import tkinter as tk
import asyncio
import json
from datetime import datetime
from tkinter import filedialog
from pathlib import Path
from llama_cloud_services import LlamaExtract
from llama_cloud.client import AsyncLlamaCloud
from llama_cloud_services.beta.classifier.client import ClassifyClient
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor


# === Commands ===
# === source venv/bin/activate.fish

def main():
    while True:
        try:
            user_input = input("parse> ").strip().lower()
            if user_input == "exit":
                print("Exiting.")
                break
            elif user_input == "clrmem":
                clear_cache()
            elif user_input == "clrt":
                clear_table()
            elif user_input == "auto":  
                asyncio.run(auto_async())
            elif user_input == "read":
                asyncio.run(read_async())
            elif user_input == "sum":
                print(get_table_summary())
            elif user_input == "tablesource":
                display_table_with_sources()
            elif user_input == "send":
                export() 
            elif user_input == "table":
                show_table()
            elif user_input == "selectfolder":
                asyncio.run(select_folder_and_process())
            elif user_input == "upload":
                if upload_files_to_dat():
                    # Optionally auto-process uploaded files
                    response = input("Process uploaded files now? (y/n): ").strip().lower()
                    if response == 'y' or response == 'yes':
                        asyncio.run(auto_async())
            elif user_input == "help":
                print("Commands:")
                print("  auto - Process all files in dat folder")
                print("  read - Read and display files from dat folder") 
                print("  selectfolder - Choose a folder to scan documents from")
                print("  upload - Upload files to dat folder")
                print("  table - Show student table")
                print("  tablesource - Show table with source files")
                print("  sum - Show table summary")
                print("  send - Export table to Excel")
                print("  clrmem - Clear cache")
                print("  clrt - Clear table")
                print("  exit - Exit program")
            else:
                print("Unknown command. Type 'help' for available commands.")                
        except EOFError:
            print("CTRL+Z Detected Exiting...")
            break


# === Setup ===

input_folder = "dat"
HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(HERE, "dat")

load_dotenv()
api_key = os.getenv('LLAMA_CLOUD_API_KEY')
os.environ['LLAMA_CLOUD_API_KEY'] = api_key # Set it in os.environ too (sometimes SDKs check here directly)
if not api_key:
    print("Error: LLAMA_CLOUD_API_KEY not found in environment variables")
    exit(1)

extractor = LlamaExtract()
agent = extractor.get_agent(name="Docparse")
client = AsyncLlamaCloud(token=api_key)
PERSISTENCE_FILE = os.path.join(HERE, "student_data.json")


# === Mem Config ===

def clear_cache():
    """Clear LlamaParse cache"""
    try:
        # Adjust path based on where your cache is stored
        cache_path = ".cache"  # or wherever LlamaParse stores cache
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print("Cache cleared successfully!")
        else:
            print("No cache found to clear.")
    except Exception as e:
        print(f"Error clearing cache: {e}")


def load_student_table():
    """Load the student table from the persistence file."""
    global student_table
    try:
        if os.path.exists(PERSISTENCE_FILE):
            with open(PERSISTENCE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure we load a dictionary, otherwise keep table empty
                if isinstance(data, dict):
                    student_table = data
                    print(f"✅ Loaded {len(student_table)} student records from file.")
                else:
                    print("⚠️ Persistence file corrupted, starting with empty table.")
    except Exception as e:
        print(f"❌ Error loading student table: {e}. Starting with empty table.")
        student_table = {}

def save_student_table():
    """Save the student table to the persistence file."""
    try:
        with open(PERSISTENCE_FILE, 'w', encoding='utf-8') as f:
            json.dump(student_table, f, indent=4)
        # print("Student table saved to disk.") # Keep this silent for GUI operations
    except Exception as e:
        print(f"Error saving student table: {e}")


# === File Scan Filter ===

def is_valid_input_file(filename):
    lower = filename.lower()
    valid_extensions = (".jpg", ".jpeg", ".png", ".pdf")
    return lower.endswith(valid_extensions)

for fname in os.listdir(input_folder):
    clean_name = fname.strip()
    if clean_name != fname:
        os.rename(os.path.join(input_folder, fname), os.path.join(input_folder, clean_name))
 

# === File Management Functions ===

def select_folder_and_process():
    """Allow user to select a folder containing documents - returns folder info without processing"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    folder_path = filedialog.askdirectory(
        title="Select folder containing documents to scan",
        initialdir=os.path.expanduser("~")
    )
    
    root.destroy()
    
    if not folder_path:
        print("No folder selected.")
        return None
    
    # Validate folder contains valid files
    valid_files = []
    for file_path in Path(folder_path).iterdir():
        if file_path.is_file() and is_valid_input_file(file_path.name):
            valid_files.append(file_path.name)
    
    if not valid_files:
        print(f"No valid document files found in selected folder: {folder_path}")
        print("Supported formats: PDF, JPG, JPEG, PNG")
        return None
    
    print(f"Found {len(valid_files)} valid files in: {folder_path}")
    
    # Return folder info instead of processing immediately
    return {
        "folder_path": folder_path,
        "files": valid_files
    }

async def process_folder_async(folder_info):
    """Process files from selected folder asynchronously - Returns raw extraction data for preview"""
    if not folder_info:
        return []
    
    folder_path = folder_info["folder_path"]
    valid_files = folder_info["files"]
    
    print(f"Processing {len(valid_files)} files from selected folder...")
    
    # Temporarily store original input_folder
    original_input_folder = globals()['input_folder']
    
    try:
        # Temporarily change input_folder to the selected folder
        globals()['input_folder'] = folder_path
        
        # CHANGED: Pass add_to_table_flag=False to prevent automatic addition
        raw_results = await process_files_async(valid_files, display_results=False, add_to_table_flag=False)
        
        print(f"Completed processing files from: {folder_path}")
        
        # CHANGED: Return formatted results for preview
        formatted_results = []
        for result, filename in raw_results:
            if result and not result.get("error"):
                extracted = result.get("extracted_data", {})
                formatted_results.append({
                    "name": extracted.get("name", "N/A"),
                    "dl_number": extracted.get("dl_number", "N/A"),
                    "dob": extracted.get("date_of_birth", "N/A"),
                    "skills_test_date": extracted.get("skills_test_date", "N/A"),
                    "Exam_Result": extracted.get("exam_result", "N/A"),
                    "ITTD_completion_date": extracted.get("ITTD_completion_date", "N/A"),
                    "ITAD_completion_date": extracted.get("ITAD_completion_date", "N/A"),
                    "de_964_number": extracted.get("pt_dee_d_number", "N/A"),
                    "adee_number": extracted.get("ADEE_number", "N/A"),
                    "_raw_data": result,  # Store raw data for later use
                    "_filename": filename  # Store filename for later use
                })
        
        return formatted_results
        
    except Exception as e:
        print(f"Error processing folder files: {e}")
        return []
    finally:
        # Restore original input_folder
        globals()['input_folder'] = original_input_folder

def upload_files_to_dat():
    """Allow user to upload multiple files to the dat folder"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_paths = filedialog.askopenfilenames(
        title="Select documents to upload",
        filetypes=[
            ("All Supported", "*.pdf *.jpg *.jpeg *.png"),
            ("PDF files", "*.pdf"),
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*")
        ],
        initialdir=os.path.expanduser("~")
    )
    
    root.destroy()
    
    if not file_paths:
        print("No files selected.")
        return False
    
    # Validate and copy files to dat folder
    copied_files = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        
        if not is_valid_input_file(filename):
            print(f"Skipping unsupported file: {filename}")
            continue
        
        destination = os.path.join(input_folder, filename)
        
        # Handle duplicate filenames
        counter = 1
        original_destination = destination
        while os.path.exists(destination):
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_{counter}{ext}"
            destination = os.path.join(input_folder, new_filename)
            counter += 1
        
        try:
            shutil.copy2(file_path, destination)
            copied_files.append(os.path.basename(destination))
            print(f"✓ Uploaded: {os.path.basename(destination)}")
        except Exception as e:
            print(f"✗ Failed to upload {filename}: {e}")
    
    if copied_files:
        print(f"\nSuccessfully uploaded {len(copied_files)} files to dat folder.")
        return True
    else:
        print("No files were successfully uploaded.")
        return False

async def process_folder_files(folder_path, valid_files):
    """Process files directly from a selected folder without copying to dat"""
    print(f"Processing {len(valid_files)} files from selected folder...")
    
    # Temporarily store original input_folder
    original_input_folder = globals()['input_folder']
    
    try:
        # Temporarily change input_folder to the selected folder
        globals()['input_folder'] = folder_path
        
        # Process files using existing async functions
        await process_files_async(valid_files, display_results=False)
        
        print(f"Completed processing files from: {folder_path}")
        display_table()
        
        return True
        
    except Exception as e:
        print(f"Error processing folder files: {e}")
        return False
    finally:
        # Restore original input_folder
        globals()['input_folder'] = original_input_folder

def get_folder_file_count(folder_path):
    """Get count of valid files in a folder (utility function for GUI)"""
    if not os.path.exists(folder_path):
        return 0
    
    count = 0
    for file_path in Path(folder_path).iterdir():
        if file_path.is_file() and is_valid_input_file(file_path.name):
            count += 1
    
    return count

def validate_upload_files(file_paths):
    """Validate a list of file paths for upload (utility function for GUI)"""
    valid_files = []
    invalid_files = []
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        if is_valid_input_file(filename):
            valid_files.append(file_path)
        else:
            invalid_files.append(filename)
    
    return valid_files, invalid_files


# === Extraction Logic ===

async def process_files_batch_async(files):
    """Async batch processing for LlamaExtract"""
    try:
        print(f"Processing {len(files)} files in parallel...")
        
        # Create tasks for all files
        tasks = []
        for filename in files:
            file_path = os.path.join(input_folder, filename)
            task = extract_with_llama_async(file_path, filename)
            tasks.append(task)
        
        # Run all extractions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            filename = files[i]
            if isinstance(result, Exception):
                print(f"  ✗ Failed: {filename} - {str(result)}")
                processed_results.append(({"error": f"Processing failed: {str(result)}"}, filename))
            else:
                print(f"  ✓ Completed: {filename}")
                processed_results.append((result, filename))
        
        print(f"Batch processing complete. Processed {len(processed_results)} files.")
        return processed_results
        
    except Exception as e:
        print(f"Batch processing failed: {str(e)}")
        return []
 

async def extract_with_llama_async(file_path, filename):
    """
    Extracts data from the full document using LlamaExtract and handles DL-40 pairing.
    The agent handles multi-page processing internally.
    """
    print(f"Starting extraction for: {file_path}")
    
    try:
        # 1. Send the full file path to agent
        result = await asyncio.to_thread(agent.extract, file_path)
        
        # 2. Get the list of results (
        results = result.data if hasattr(result, 'data') else result 
        
        # 3. Ensure 'results' is always a list for the pairing logic
        if not isinstance(results, list):
            results = [results] if results and not results.get("error") else []
        
        # 4. Run the pairing logic on the full list of extracted pages/documents

        paired_results = pair_dl40_pages(results)
        
        # 5. Return the consolidated list
        return paired_results
        
    except Exception as e:
        print(f"Extraction error: {e}")
        import traceback
        traceback.print_exc()
        # Return a list containing an error dictionary for consistent handling
        return [{"error": f"Extraction failed: {str(e)}"}]

def pair_dl40_pages(results):
    """Pair DL-40-Front and DL-40-Back pages and merge their data"""
    paired = []
    i = 0
    
    while i < len(results):
        current = results[i]
        doc_type = current.get("document_type", "").lower()
        
        # **FIX: Neutralize the examiner name from the DL-40 back page**
        is_dl40_back = "dl-40-back" in doc_type or "dl-40 back" in doc_type
        if is_dl40_back:
             # Ensure the examiner name (if extracted as 'name') is ignored.
             if "extracted_data" in current and "name" in current["extracted_data"]:
                 current["extracted_data"]["name"] = None
        # **END FIX**
        
        # Check if this is a DL-40-Front
        if "dl-40-front" in doc_type or "dl-40 front" in doc_type:
            # Look for the next page to see if it's DL-40-Back
            if i + 1 < len(results):
                next_page = results[i + 1]
                next_doc_type = next_page.get("document_type", "").lower()
                
                if "dl-40-back" in next_doc_type or "dl-40 back" in next_doc_type:
                    
                    front_data = current.get("extracted_data", {})
                    back_data = next_page.get("extracted_data", {})
                    
                    skill_date_front_str = front_data.get("skills_test_date")
                    exam_date_back_str = back_data.get("exam_date")
                    
                    # Initialize final date and warning
                    final_skill_date = skill_date_front_str or exam_date_back_str
                    warning_sign = ""
                    
                    # --- EMBEDDED DATE COMPARISON AND CONSOLIDATION LOGIC ---
                    
                    # List of formats to try for robust parsing
                    date_formats = ["%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%B %d, %Y"]

                    def safe_parse(date_str):
                        if not date_str or date_str.lower() in ["n/a", "not found"]:
                            return None
                        for fmt in date_formats:
                            try:
                                return datetime.strptime(date_str, fmt), date_str
                            except ValueError:
                                continue
                        return None

                    parsed_front = safe_parse(skill_date_front_str)
                    parsed_back = safe_parse(exam_date_back_str)
                    
                    obj1, str1 = parsed_front if parsed_front else (None, None)
                    obj2, str2 = parsed_back if parsed_back else (None, None)

                    # Case 1: Both dates parsed and don't match
                    if obj1 and obj2 and obj1 != obj2:
                        
                        # Select the most recent date object
                        if obj1 > obj2:
                            final_skill_date = str1
                        else:
                            final_skill_date = str2

                        # Set the warning sign and print message
                        warning_sign = " ⚠️" 
                        print(f"DL-40 Date Conflict: Front ({str1}) vs. Back ({str2}). Using most recent: {final_skill_date}")

                    # Case 2: Both dates parsed and they match, or only one exists.
                    # In these cases, final_skill_date is already set correctly.
                    
                    # --- END EMBEDDED LOGIC ---
                    
                    # Merge front and back data
                    merged = {
                        "document_type": "DL-40",
                        "extracted_data": {}
                    }
                    
                    # Get standard data
                    merged["extracted_data"]["name"] = front_data.get("name")
                    merged["extracted_data"]["dl_number"] = front_data.get("dl_number")
                    
                    # Store the final consolidated date with the warning sign (if any)
                    if final_skill_date:
                        merged["extracted_data"]["skills_test_date"] = final_skill_date + warning_sign
                    else:
                        merged["extracted_data"]["skills_test_date"] = "N/A"
                        
                    # Only keep exam_result from the back page
                    merged["extracted_data"]["exam_result"] = back_data.get("exam_result")
                    
                    # Note: 'exam_date' is excluded from the merged record.
                    
                    paired.append(merged)
                    i += 2  # Skip both pages
                    continue
            
            # If no back page found, just add front page
            paired.append(current)
            i += 1
        else:
            # Not a DL-40 front, just add it
            paired.append(current)
            i += 1
    
    return paired

  
async def read_async():
    """Async version of read()"""
    files = [f for f in os.listdir(input_folder) if is_valid_input_file(f)]
    
    if not files:
        print("No valid files found in Dat folder.")
        return
    
    if len(files) == 1:
        await process_single_file_async(files[0], display_results=True)
    else:
        print(f"Processing {len(files)} files...")
        await process_files_async(files, display_results=True)



async def auto_async():
    """
    Async version of auto() - Processes files, runs consolidation, and returns 
    a consolidated list of records for preview WITHOUT adding to persistence.
    """
    global student_table 
    
    files = [f for f in os.listdir(input_folder) if is_valid_input_file(f)]
    
    if not files:
        print("No valid files found in Dat folder.")
        return []
    
    print(f"Processing {len(files)} files for preview...")
    
    # 1. Store the CURRENT state of the persistent table
    original_table_state = student_table.copy()
    
    # 2. Clear the table for temporary, in-memory processing of new files
    student_table = {} 
    
    try:
        # 3. Process files and force them to be ADDED to the now-empty global table.
        # This runs ALL consolidation logic (DL-40 merge, name priority, ITAD merge).
        # add_to_table_flag=True forces the data into the temporary student_table.
        await process_files_async(files, display_results=False, add_to_table_flag=True)

        # 4. Pull the CONSOLIDATED, formatted results from the temporary table state
        consolidated_preview_data = display_table() 
        
        # 5. Return the consolidated list for display.
        return consolidated_preview_data
        
    except Exception as e:
        print(f"Error during preview consolidation: {e}")
        return []
    finally:
        # 6. RESTORE the original persistent table state, discarding the temporary data
        student_table = original_table_state
        # We do NOT call save_student_table() here, preserving the original disk state.
        print("Preview consolidation complete. Restored original table state.")


async def process_files_async(files, display_results=True, add_to_table_flag=True):
    """Main async file processing
    
    Args:
        files: List of filenames to process
        display_results: Whether to print extraction results
        add_to_table_flag: Whether to add results to student_table (False for preview mode)
    
    Returns:
        List of (result, filename) tuples when add_to_table_flag=False, otherwise None
    """
    raw_results = []  # Store results for preview mode
    
    if len(files) >= 2:
        batch_results = await process_files_batch_async(files)
        
        for result, filename in batch_results:
            if result and not result.get("error"):
                if display_results:
                    display_relevant_fields(result)
                    
                # CHANGED: Only add to table if flag is True
                if add_to_table_flag:
                    add_to_table(result, filename)
                else:
                    # Store raw results for preview mode
                    raw_results.append((result, filename))
            else:
                error_msg = result.get('error', 'Unknown error') if result else 'Unknown error'
                if display_results:
                    print(f"Processing failed for {filename}: {error_msg}")
                else:
                    print(f"Skipping {filename} - processing failed: {error_msg}")
    else:
        for filename in files:
            result = await process_single_file_async(filename, display_results, add_to_table_flag)
            if not add_to_table_flag and result:
                raw_results.extend(result)
    
    # Return raw results if in preview mode
    return raw_results if not add_to_table_flag else None

async def process_single_file_async(filename, display_results=True, add_to_table_flag=True):
    """Async single file processing
    
    Args:
        filename: Name of file to process
        display_results: Whether to print extraction results
        add_to_table_flag: Whether to add results to student_table (False for preview mode)
    
    Returns:
        List of (result, filename) tuples when add_to_table_flag=False, otherwise None
    """
    file_path = os.path.join(input_folder, filename)
    raw_results = []
    
    if display_results:
        print(f"Processing: {filename}")
        print("-" * 50)
    
    try:
        results = await extract_with_llama_async(file_path, filename)
        for result in results:
            if display_results:
                display_relevant_fields(result)
            
            # CHANGED: Only add to table if flag is True
            if add_to_table_flag:
                add_to_table(result, filename)
            else:
                raw_results.append((result, filename))
                
    except Exception as e:
        print(f"Error processing {filename}: {e}")
    
    return raw_results if not add_to_table_flag else None




def format_field_name(field_name):
    """Format field names with proper capitalization for abbreviations"""
    # Handle special cases first
    if field_name == "ITAD_completion_date":
        return "ITAD Completion Date"
    elif field_name == "ITTD_completion_date":
        return "ITTD Completion Date"
    elif field_name == "dl_number":
        return "DL Number"
    elif field_name == "pt_dee_d_number":
        return "PT DEE D Number"
    elif field_name == "ADEE_number":
        return "ADEE Number"
    else:
        # For other fields, use standard title case
        return field_name.replace('_', ' ').title()


def display_relevant_fields(data):
    """Show document type and relevant fields, or raw text if extraction fails"""
    if not data:
        print("No data received.")
        return
    
    # If extraction failed, show raw text
    if "error" in data:
        print(f"Extraction failed: {data['error']}")
        return
    
    # Get document type and extracted data
    doc_type = data.get("document_type", "unknown").lower()
    extracted_data = data.get("extracted_data", {})
    
    print(f"Document Type: {doc_type.upper()}")
    
    # Field mappings for each document type
    doc_fields = {
        "driver license": ["name", "dl_number", "date_of_birth"],
        "de-964 certificate": ["name", "pt_dee_d_number"],
        "adee-1317 certificate": ["name", "ADEE_number"],
        "dl-40": ["name", "dl_number", "skills_test_date"],
        "teen impact certificate": ["name", "ITTD_completion_date"],
        "adult impact certificate": ["name", "ITAD_completion_date"]
    }
    
    # Show relevant fields for this document type
    fields_to_show = doc_fields.get(doc_type, [])
    
    if fields_to_show:
        print("Extracted Fields:")
        for field in fields_to_show:
            value = extracted_data.get(field, "Not found")
            display_name = format_field_name(field)
            if value and value != "Not found":
                print(f"  {display_name}: {value}")
            else:
                print(f"  {display_name}: Not found")
    else:
        print("No specific fields defined for this document type.")
        print("Raw extracted data:")
        for key, value in extracted_data.items():
            if value:
                display_name = format_field_name(key)
                print(f"  {display_name}: {value}")


# === Table Engine ===

student_table = {}
def clean_name(name):
    """Clean and normalize names for consistent matching"""
    if not name:
        return ""
    
    # Remove leading/trailing spaces and normalize internal spaces
    cleaned = ' '.join(name.strip().split())
    
    # Convert to uppercase for consistent comparison
    cleaned = cleaned.upper()
    
    return cleaned

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def find_matching_student(new_name):
    """Find existing student that matches the new name using first+last name matching"""
    new_name_clean = clean_name(new_name)
    
    if not new_name_clean:
        return None
    
    # Split the new name into words
    new_words = new_name_clean.split()
    
    if len(new_words) == 0:
        return None
    
    # Extract first and last name from new name
    new_first = new_words[0] if len(new_words) > 0 else ""
    new_last = new_words[-1] if len(new_words) > 1 else ""
    
    best_match = None
    best_score = 0
    
    for existing_name in student_table.keys():
        existing_clean = clean_name(existing_name)
        
        if existing_clean == new_name_clean:
            return existing_name
        
        # Split existing name into words
        existing_words = existing_clean.split()
        
        if len(existing_words) == 0:
            continue
        
        # Extract first and last name from existing name
        existing_first = existing_words[0] if len(existing_words) > 0 else ""
        existing_last = existing_words[-1] if len(existing_words) > 1 else ""
        
        score = 0
        
        # MATCHING STRATEGY:
        # 1. First name + Last name match (case-insensitive) = 100 points
        # 2. First name only match (single word names) = 50 points
        # 3. Partial name subset match = 30 points
        # 4. Fuzzy match with OCR errors = 20 points
        
        # Strategy 1: First + Last name exact match
        if new_first and new_last and existing_first and existing_last:
            if new_first == existing_first and new_last == existing_last:
                score = 100  # Perfect first+last match
        
        # Strategy 2: First name only match (for single word names)
        elif len(new_words) == 1 and len(existing_words) == 1:
            if new_first == existing_first:
                score = 50
        
        # Strategy 3: Check if one name is a subset of another (partial extraction)
        if score == 0:
            new_words_set = set(new_words)
            existing_words_set = set(existing_words)
            
            if new_words_set.issubset(existing_words_set) or existing_words_set.issubset(new_words_set):
                overlap = len(new_words_set.intersection(existing_words_set))
                score = 30 + (overlap * 5)  # Base 30 + bonus for each matching word
        
        # Strategy 4: Fuzzy matching for OCR errors
        if score == 0:
            matched_words = 0
            for new_word in new_words[:2]:  # Check only first 2 words (first+last)
                for existing_word in existing_words[:2]:
                    word_distance = levenshtein_distance(new_word, existing_word)
                    word_max_len = max(len(new_word), len(existing_word))
                    
                    if word_distance <= 2 and word_max_len > 3 and word_distance / word_max_len <= 0.25:
                        matched_words += 1
                        break
            
            if matched_words >= 2:  # Both first and last names fuzzy match
                score = 20
        
        # Update best match if this score is better
        if score > best_score:
            best_match = existing_name
            best_score = score
    
    # Return match only if score is high enough
    # Score 100 = perfect first+last match
    # Score 50 = single name exact match
    # Score 30+ = subset match
    # Score 20 = fuzzy match
    return best_match if best_score >= 20 else None


# --- Document Priority Configuration ---
PRIORITY_SCORES = {
    # High Priority: Primary sources for student ID and name (100)
    "dl-40": 100,
    "dl-40 front": 100,
    "driver license": 100,
    
    # Medium Priority: State-issued certs with reliable name info (80)
    "de-964 certificate": 80,
    "adee-1317 certificate": 80,
    
    # Low Priority: Impact certs often have abbreviated names or less official formatting (50)
    "teen impact certificate": 50,
    "adult impact certificate": 50,
    "unknown": 0,
}

def get_doc_priority(doc_type):
    """Maps document type to a consistent priority score."""
    if not doc_type:
        return 0
    clean_type = doc_type.lower().strip()
    # The 'dl-40' entry covers cases where it is merged or simply identified as 'DL-40'
    return PRIORITY_SCORES.get(clean_type, 0)

def goodest_name(name):
    """
    Calculates a score based on name length and word count (higher is better).
    This function leverages the existing clean_name helper.
    """
    if not name:
        return (0, 0)
    
    # Note: clean_name is defined earlier in parse.py (around line 497)
    clean = clean_name(name) 
    words = clean.split()
    word_count = len(words)
    char_count = len(clean) # Total characters in the cleaned name
    
    # Return as a tuple for lexicographical comparison: (word_count, char_count)
    return (word_count, char_count)




def add_to_table(data, filename):
    """Add extracted document data to the student table"""
    if not data or "error" in data:
        print(f"Skipping table entry for {filename} - extraction failed")
        return
    
    extracted_data = data.get("extracted_data", {})
    raw_name = (extracted_data.get("name") or "").strip()
    doc_type = data.get("document_type", "unknown").lower()
    
    
    if not raw_name or raw_name == "N/A":
        print(f"No student name found in {filename} - skipping table entry")
        return  # Exit function completely
    
    # Find if this student already exists
    matching_student = find_matching_student(raw_name)
    
    # (Existing Impact Certificate handling logic omitted for brevity, but retained in your file)
    
    # Find if this student already exists
    matching_student = find_matching_student(raw_name)
    
    if matching_student:
        # Update existing student record
        student_name = matching_student
        
        # Get the document type of the existing student's most recent entry
        existing_doc_type = None
        if student_table[matching_student]["documents"]:
            last_doc = student_table[matching_student]["documents"][-1]
            if "(" in last_doc and ")" in last_doc:
                existing_doc_type = last_doc.split("(")[-1].replace(")", "")
        
        # Check if we need to update to a better name (prioritizing length/completeness)
        better_name = goodest_name(matching_student, raw_name, existing_doc_type, doc_type)
        if better_name != matching_student:
            # Create new entry with better name and copy all data
            student_table[better_name] = student_table[matching_student].copy()
            student_table[better_name]["name"] = better_name
            # Remove old entry
            del student_table[matching_student]
            student_name = better_name
    else:
        # Create new student record
        student_name = raw_name
        student_table[student_name] = {
            "name": student_name,
            "dl_number": None,
            "date_of_birth": None,
            "skills_test_date": None,
            "exam_result": None,
            "de_964_number": None,
            "adee_number": None,
            "ittd_completion_date": None,
            "itad_completion_date": None,
            "documents": []
        }
    
    # Current document to track (avoid duplicates)
    doc_entry = f"{filename} ({doc_type})"
    if doc_entry not in student_table[student_name]["documents"]:
        student_table[student_name]["documents"].append(doc_entry)
    
    # Update fields based on document type and available data
    student_record = student_table[student_name]
    
    # Map fields from extracted data to table columns (exam_result is omitted here)
    field_mappings = {
        "dl_number": "dl_number",
        "date_of_birth": "date_of_birth", 
        "skills_test_date": "skills_test_date",
        "Exam_Result": "exam_result", 
        "pt_dee_d_number": "de_964_number",
        "ADEE_number": "adee_number",
        "ITTD_COMPLETION_DATE": "ittd_completion_date", 
        "ITAD_COMPLETION_DATE": "itad_completion_date"  
    }
    
    # Update fields if they exist in extracted data and aren't already filled
    for source_field, table_field in field_mappings.items():
        if source_field in extracted_data and extracted_data[source_field]:
            value = extracted_data[source_field]
            
            # Handle None values
            if value is None or value == "Not found":
                continue
            
            # Convert to string and strip
            value = str(value).strip()
            
            if not value:  # Skip empty strings
                continue
            
            # --- START FIX FOR IMPACT DATES / SINGLE-SOURCE FIELDS ---
            # Ensure these fields are prioritized and fill the spot if found.
            if table_field in ["ittd_completion_date", "itad_completion_date", "de_964_number", "adee_number"]:
                current_value = student_record[table_field]
                if not current_value or current_value == "N/A":
                    student_record[table_field] = value
                    continue # Field filled, move to next source field
            # --- END FIX ---
            
            
            # Default logic for shared/multi-source fields (Name, DL#, DOB, Skills Date)
            current_value = student_record[table_field]
            if not current_value or current_value == "N/A":
                student_record[table_field] = value
            elif len(str(value)) > len(str(current_value)):
                student_record[table_field] = value
                
    save_student_table()

def display_table():
    """Return formatted student table data as list of dicts for frontend"""
    if not student_table:
        return []  # Return empty list instead of None
        
    rows = []
    for name in sorted(student_table.keys()):
        record = student_table[name]
        rows.append({
            "name": name,
            "dl_number": record.get("dl_number") or "N/A",
            "dob": record.get("date_of_birth") or "N/A",
            "skills_test_date": record.get("skills_test_date") or "N/A",
            "exam_result": record.get("exam_result") or "N/A",
            "de_964_number": record.get("de_964_number") or "N/A",
            "adee_number": record.get("adee_number") or "N/A",
            "ittd_completion_date": record.get("ittd_completion_date") or "N/A",
            "itad_completion_date": record.get("itad_completion_date") or "N/A",
        })
    return rows


def add_preview_records_to_table(preview_records):
    """
    Add preview records to the main student table.
    This function is called from the UI when user clicks 'Add to Table'.
    
    Args:
        preview_records: List of preview records with _raw_data and _filename fields
    
    Returns:
        Number of records successfully added
    """
    added_count = 0
    
    for record in preview_records:
        # Extract the raw data and filename that were stored during preview
        raw_data = record.get("_raw_data")
        filename = record.get("_filename", "unknown")
        
        if raw_data and not raw_data.get("error"):
            # Add the raw extraction data to the table
            add_to_table(raw_data, filename)
            added_count += 1
        else:
            print(f"Skipping record for {record.get('name', 'unknown')} - no valid data")
    
    return added_count


def display_table_with_sources():
    if not student_table:
        print("No student data in table.")
        return
    
    print("\n" + "="*130)
    print("STUDENT DOCUMENT TABLE (WITH SOURCES)")
    print("="*130)
    
    # Header
    header = f"{'Name':<30} {'DL Number':<15} {'DOB':<12} {'Skills Test Date':<12} {'DE-964#':<12} {'ADEE#':<12} {'ITTD Date':<12} {'ITAD Date':<12}"
    print(header)
    print("-"*130)
    
    # Rows - sort by name for consistent display
    for name in sorted(student_table.keys()):
        record = student_table[name]
        
        # Truncate long names
        display_name = name[:28] + ".." if len(name) > 30 else name
        
        row = f"{display_name:<30} "
        row += f"{record['dl_number'] or 'N/A':<15} "
        row += f"{record['date_of_birth'] or 'N/A':<12} "
        row += f"{record['skills_test_date'] or 'N/A':<12} "
        row += f"{record['de_964_number'] or 'N/A':<12} "
        row += f"{record['adee_number'] or 'N/A':<12} "
        row += f"{record['ITTD_completion_date'] or 'N/A':<12} "
        row += f"{record['ITAD_completion_date'] or 'N/A':<12}"
        
        print(row)
        
        # Show source documents for this student
        docs_str = "Sources: " + ", ".join(record['documents'])
        if len(docs_str) > 125:
            docs_str = docs_str[:122] + "..."
        print(f"   {docs_str}")
        print()
    
    print(f"Total Students: {len(student_table)}")
    print("="*130)


def clear_table():
    """Clear all data from the student table"""
    global student_table
    student_table = {}
    save_student_table()
    print("Student table cleared.")

def get_student_data(name):
    """Get data for a specific student"""
    return student_table.get(name, None)

def get_table_summary():
    """Get summary statistics of the table with age group classification"""
    if not student_table:
        return "Table is empty"
    
    total_students = len(student_table)
    minor_students = 0
    adult_18_24 = 0
    adult_25_plus = 0
    
    # Count age groups
    for record in student_table.values():
        has_de964 = record['de_964_number'] and record['de_964_number'] != 'N/A'
        has_adee = record['adee_number'] and record['adee_number'] != 'N/A'

    if has_de964:
        minor_students += 1
    elif has_adee:
        adult_18_24 += 1
    else:
        adult_25_plus += 1
    
    return f"""
Total Students: {total_students}

Student Classification:
- Minor Students (DE-964): {minor_students}
- Adult Students 18-24 (ADEE): {adult_18_24}
- Adult Students 25+ (No specific cert): {adult_25_plus}
- Total Students: {total_students}
"""


def export():
    """Export the student table to Excel file"""
    if not student_table:
        print("No student data to export.")
        return
    
    # Convert table to list of dictionaries for DataFrame
    export_data = []
    for name in sorted(student_table.keys()):
        record = student_table[name]
        export_data.append({
            'Name': record['name'],
            'DL Number': record['dl_number'] or 'N/A',
            'DOB': record['date_of_birth'] or 'N/A',
            'Skills Test Date': record['skills_test_date'] or 'N/A',
            'DE-964 Number': record['de_964_number'] or 'N/A',
            'ADEE Number': record['adee_number'] or 'N/A',
            'ITTD Date': record['ITTD_completion_date'] or 'N/A',
            'ITAD Date': record['ITAD_completion_date'] or 'N/A'
        })
    
    # Create DataFrame and export to Excel
    df = pd.DataFrame(export_data)
    filename = "student_documents_table.xlsx"
    
    try:
        df.to_excel(filename, index=False)
        print(f"Table exported successfully to {filename}")
    except Exception as e:
        print(f"Error exporting to Excel: {e}")

def show_table():
    """Display the current student table"""
    display_table()


# === Table Storing ===

def group_students_by_month():
    
    if not student_table:
        return []
    
    monthly_groups = {}
    unknown_month = []  # ENHANCED: Separate list for students without valid skills_test_date
    
    for name, record in student_table.items():
        skills_date = record.get("skills_test_date")
        
        # Helper to create a consistent student dict for export
        def create_student_dict(date_value):
            return {
                "name": record["name"],
                "dl_number": record["dl_number"] or "N/A",
                "dob": record["date_of_birth"] or "N/A",
                "skills_test_date": date_value or "N/A",
                "exam_result": record.get("exam_result") or "N/A", # ADDED FOR MONTHLY TABLE
                "ittd_completion_date": record["ittd_completion_date"] or "N/A",
                "itad_completion_date": record["itad_completion_date"] or "N/A",
                "de_964_number": record["de_964_number"] or "N/A",
                "adee_number": record["adee_number"] or "N/A",
            }

        
        # Check if date is missing or invalid
        if not skills_date or skills_date == "N/A" or skills_date == "Not found":
            unknown_month.append(create_student_dict("N/A"))
            continue
        
        try:
            # Try multiple date formats
            date_obj = None
            for fmt in ["%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%B %d, %Y"]:
                try:
                    date_obj = datetime.strptime(skills_date, fmt)
                    break
                except:
                    continue
            
            if not date_obj:
                # Couldn't parse date, add to unknown
                unknown_month.append(create_student_dict(skills_date))
                continue
            
            month_key = f"{date_obj.strftime('%B')}_{date_obj.year}"
            month_name = date_obj.strftime('%B')
            year = date_obj.year
            
            if month_key not in monthly_groups:
                monthly_groups[month_key] = {
                    "month": month_name,
                    "year": year,
                    "students": []
                }
            
            monthly_groups[month_key]["students"].append(create_student_dict(skills_date))
        except Exception as e:
            print(f"Error parsing date for {name}: {e}")
            unknown_month.append(create_student_dict(skills_date))
    
    # Sort by year and month
    sorted_groups = sorted(monthly_groups.values(), 
                          key=lambda x: (x["year"], x["month"]), 
                          reverse=True)
    
    # Add unknown month group if it has entries
    if unknown_month:
        sorted_groups.append({
            "month": "Unknown",
            "year": "",
            "students": unknown_month
        })
    
    return sorted_groups

def delete_student_from_table(student_name):
    """Delete a student from the table"""
    if student_name in student_table:
        del student_table[student_name]
        save_student_table()
        return True
    return False


def export_month_to_excel(month_data):
    """Export a specific month's data to Excel"""
    if not month_data or not month_data.get("students"):
        return None
    
    month = month_data["month"]
    year = month_data["year"]
    students = month_data["students"]
    
    df = pd.DataFrame(students)
    filename = f"{month}_{year}_report.xlsx"
    
    try:
        df.to_excel(filename, index=False)
        return filename
    except Exception as e:
        print(f"Error exporting to Excel: {e}")
        return None

load_student_table()
