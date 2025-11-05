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
            elif user_input == "send":
                export() 
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

async def process_single_file_async(filename):
    """Async single file processing. Extracts and returns document list.
    
    Returns:
        List of document dictionaries (results).
    """
    file_path = os.path.join(input_folder, filename)
    
    print(f"Processing: {filename}")
    print("-" * 50)
    
    try:
        # Returns paired/normalized documents
        results = await extract_with_llama_async(file_path, filename)
        
        # NOTE: No console display or saving occurs here.
        return results
                
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return []

async def process_files_async(files):
    """Main async file processing. Orchestrates extraction and collects results.
    
    Returns:
        List of (result_doc, filename) tuples containing all processed documents.
    """
    all_processed_docs_with_files = [] # This is the flat list returned to the consolidation engine
    
    if len(files) >= 2:
        # Batch processing: CRITICAL FIX for "successfully parsed 0 records"
        batch_results = await process_files_batch_async(files)
        
        for results_list, filename in batch_results: # results_list is List[dict] or Exception
            if isinstance(results_list, Exception):
                print(f"Processing failed for {filename}: {str(results_list)}")
                continue
            
            if not results_list:
                print(f"No documents found in {filename}")
                continue

            # Collect all documents from this file
            for result_doc in results_list: 
                if result_doc and not result_doc.get("error"):
                    # COLLECT: Add the document and its source filename to the flat list
                    all_processed_docs_with_files.append((result_doc, filename))
                else:
                    error_msg = result_doc.get('error', 'Unknown error') if result_doc else 'Unknown error'
                    print(f"  - Sub-document failed for {filename}: {error_msg}")
    else:
        # Single file processing
        for filename in files:
            # Note: process_single_file_async no longer takes display/save flags
            result_docs = await process_single_file_async(filename) 
            if result_docs:
                all_processed_docs_with_files.extend([(doc, filename) for doc in result_docs])

    # Returns the collected documents for the consolidation engine
    return all_processed_docs_with_files

async def auto_async():
    """
    Async version of auto() - Processes files from 'dat' folder, 
    and returns a list of formatted records for preview.
    """
    
    files = [f for f in os.listdir(input_folder) if is_valid_input_file(f)]
    
    if not files:
        print("No valid files found in Dat folder.")
        return []
    
    print(f"Processing {len(files)} files for preview from 'dat' folder...")
    
    try:
        # 1. Orchestrate Extraction and Collect Data
        # Returns List[ (document_data, filename) ]
        all_docs_with_files = await process_files_async(files)

        if not all_docs_with_files:
            print("No data extracted from files.")
            return []
            
        # 2. Run Consolidation Engine
        # assembled_table will be a dictionary: {student_name: row_data}
        assembled_table = consolidate_records(all_docs_with_files)

        # 3. Format the assembled table for the GUI preview table
        formatted_results = []
        for name, record in assembled_table.items():
            
            # NOTE: All keys are guaranteed to be lowercase now
            formatted_results.append({
                "name": record.get("name", name),
                "dl_number": record.get("dl_number", "N/A"),
                "dob": record.get("date_of_birth", "N/A"),
                "skills_test_date": record.get("skills_test_date", "N/A"),
                "exam_result": record.get("exam_result", "N/A"),
                "ittd_completion_date": record.get("ittd_completion_date", "N/A"),
                "itad_completion_date": record.get("itad_completion_date", "N/A"),
                "de_964_number": record.get("de_964_number", "N/A"),
                "adee_number": record.get("adee_number", "N/A"),
                "_raw_data": record,  # Store the assembled row for saving
                "_filename": record.get("documents", []) # Store source list
            })
        
        print(f"Preview generation complete. Found {len(formatted_results)} potential records.")
        return formatted_results
        
    except Exception as e:
        print(f"Error during 'auto' preview generation: {e}")
        return []



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
    """Process files from selected folder asynchronously - Returns consolidated data for preview"""
    if not folder_info:
        return []
    
    folder_path = folder_info["folder_path"]
    valid_files = folder_info["files"]
    
    print(f"Processing {len(valid_files)} files from selected folder...")
    
    # Temporarily store original input_folder
    original_input_folder = globals()['input_folder']
    
    try:
        # Temporarily change input_folder to the selected folder path for correct file reading
        globals()['input_folder'] = folder_path
        
        # 1. Orchestrate Extraction and Collect Data
        # We call process_files_async (our new orchestrator)
        all_docs_with_files = await process_files_async(valid_files)

        if not all_docs_with_files:
            print("No data extracted from files.")
            return []
            
        # 2. Run Consolidation Engine
        assembled_table = consolidate_records(all_docs_with_files)

        # 3. Format the assembled table for the GUI preview table (same as auto_async)
        formatted_results = []
        for name, record in assembled_table.items():
            
            # All keys are guaranteed to be lowercase now
            formatted_results.append({
                "name": record.get("name", name),
                "dl_number": record.get("dl_number", "N/A"),
                "dob": record.get("date_of_birth", "N/A"),
                "skills_test_date": record.get("skills_test_date", "N/A"),
                "exam_result": record.get("exam_result", "N/A"),
                "ittd_completion_date": record.get("ittd_completion_date", "N/A"),
                "itad_completion_date": record.get("itad_completion_date", "N/A"),
                "de_964_number": record.get("de_964_number", "N/A"),
                "adee_number": record.get("adee_number", "N/A"),
                "_raw_data": record,  # Store the assembled row for saving
                "_filename": record.get("documents", []) # Store source list
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

async def process_files_batch_async(files):
    """
    Async batch processing for LlamaExtract.
    Returns: List of (result_list, filename) tuples. 
             'result_list' is a List[dict] on success, or Exception on failure.
    """
    try:
        print(f"Processing {len(files)} files in parallel...")
        
        # Create tasks for all files
        tasks = []
        for filename in files:
            file_path = os.path.join(input_folder, filename)
            # NOTE: We use extract_with_llama_async here, which calls the DL-40 pairing logic
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
                # Return the Exception wrapped in a dictionary for consistent handling in the caller
                processed_results.append((Exception(f"Extraction failed: {str(result)}"), filename)) 
            else:
                print(f"  ✓ Completed: {filename}")
                processed_results.append((result, filename)) # result is List[dict]
        
        print(f"Batch processing complete. Processed {len(processed_results)} files.")
        return processed_results
        
    except Exception as e:
        print(f"Batch processing failed: {str(e)}")
        return []


# === Extraction & Document Processing Logic

def normalize_document_keys(result):
    """
    Converts keys in 'extracted_data' to lowercase with underscores.
    This enforces the 'all lowercase' rule immediately after extraction/pairing.
    """
    if result and "extracted_data" in result and isinstance(result["extracted_data"], dict):
        result["extracted_data"] = {
            k.lower().replace(' ', '_'): v 
            for k, v in result["extracted_data"].items()
        }
    return result

async def extract_with_llama_async(file_path, filename):
    """
    Extracts data from the full document using LlamaExtract and handles DL-40 pairing.
    The agent handles multi-page processing internally.
    """
    print(f"Starting extraction for: {file_path}")
    
    try:
        # 1. Send the full file path to agent
        result = await asyncio.to_thread(agent.extract, file_path)
        
        # 2. Get the list of results
        results = result.data if hasattr(result, 'data') else result 
        
        # 3. Ensure 'results' is always a list
        if not isinstance(results, list):
            results = [results] if results and not results.get("error") else []
        
        # 4. Run the pairing logic on the full list of extracted pages/documents
        # Normalization (to lowercase) happens *inside* pair_dl40_pages now.
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
    """
    Pair DL-40-Front and DL-40-Back pages using non-sequential matching (within the same file).
    This collects all fronts and backs first, then attempts a 1:1 merge.
    All extracted keys are normalized to lowercase before use.
    """
    dl40_fronts = []
    dl40_backs = []
    others = []
    
    # --- Pass 1: Categorize and Normalize ---
    for result in results:
        # Normalize keys before categorization
        normalized_result = normalize_document_keys(result) 
        doc_type = normalized_result.get("document_type", "").lower()
        
        is_dl40_front = "dl-40-front" in doc_type or "dl-40 front" in doc_type
        is_dl40_back = "dl-40-back" in doc_type or "dl-40 back" in doc_type
        
        if is_dl40_front:
            dl40_fronts.append(normalized_result)
        elif is_dl40_back:
            # We keep the back page even without a name, as it holds the result.
            dl40_backs.append(normalized_result)
        else:
            others.append(normalized_result)

    # --- Pass 2: Matching and Merging ---
    
    # Standard Case: 1 Front and 1 Back from the same file
    if len(dl40_fronts) == 1 and len(dl40_backs) == 1:
        
        current = dl40_fronts[0]
        next_page = dl40_backs[0]
        
        # --- PAIRING LOGIC ---
        front_data = current.get("extracted_data", {})
        back_data = next_page.get("extracted_data", {})
        
        skill_date_front_str = front_data.get("skills_test_date")
        exam_date_back_str = back_data.get("exam_date") 
        
        final_skill_date = skill_date_front_str or exam_date_back_str
        warning_sign = ""
        
        # (Date comparison logic remains the same...)
        date_formats = ["%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%B %d, %Y"]
        def safe_parse(date_str):
            if not date_str or date_str.lower() in ["n/a", "not found"]: return None
            for fmt in date_formats:
                try: return datetime.strptime(date_str, fmt), date_str
                except ValueError: continue
            return None

        parsed_front = safe_parse(skill_date_front_str)
        parsed_back = safe_parse(exam_date_back_str)
        obj1, str1 = parsed_front if parsed_front else (None, None)
        obj2, str2 = parsed_back if parsed_back else (None, None)

        if obj1 and obj2 and obj1 != obj2:
            if obj1 > obj2: final_skill_date = str1
            else: final_skill_date = str2
            warning_sign = " ⚠️" 
            print(f"DL-40 Date Conflict: Front ({str1}) vs. Back ({str2}). Using most recent: {final_skill_date}")
        
        # Simplified retrieval: TRUSTING THE USER'S EXPECTED KEY
        exam_result_value = back_data.get("exam_result") or "N/A"
        
        print(f"DEBUG: Successfully paired 1 Front and 1 Back. Exam result: {exam_result_value}")

        # Merge front and back data
        merged = {
            "document_type": "DL-40",
            "extracted_data": {
                # Name MUST come ONLY from front
                "name": front_data.get("name"), 
                "dl_number": front_data.get("dl_number"),
                "date_of_birth": front_data.get("date_of_birth"),
                # Merged/Resolved Fields
                "skills_test_date": (final_skill_date + warning_sign) if final_skill_date else "N/A",
                "exam_result": exam_result_value 
            }
        }
        
        return [merged] + others # Return the single merged DL-40 plus all others

    # Fallback/Error Handling: If no merge or multiple parts found
    
    # If the file contains multiple DL-40 documents or mismatched pairs, we cannot safely merge.
    if len(dl40_fronts) > 1 or len(dl40_backs) > 1:
        print(f"Warning: Skipping DL-40 merge due to multiple parts ({len(dl40_fronts)} Fronts, {len(dl40_backs)} Backs). Returning unmerged documents.")
        # Return all documents un-merged so they can be individually consolidated
        return dl40_fronts + dl40_backs + others 
    
    # If only fronts or only backs exist, the originals are returned un-merged.
    if len(dl40_fronts) == 1 and len(dl40_backs) == 0:
        print("Note: Found lone DL-40 Front, returning unmerged.")
    
    if len(dl40_fronts) == 0 and len(dl40_backs) >= 1:
        print(f"Note: Found {len(dl40_backs)} lone DL-40 Back(s), returning unmerged (contains exam result).")

    return dl40_fronts + dl40_backs + others
  
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

def consolidate_records(all_processed_docs_with_files):
    """
    NEW CONSOLIDATION ENGINE: Takes all documents and merges them into a temporary student table.
    """
    newly_consolidated_table = {}
    
    for result_doc, filename in all_processed_docs_with_files:
        
        extracted_data = result_doc.get("extracted_data", {})
        raw_name = (extracted_data.get("name") or "").strip()
        doc_type = result_doc.get("document_type", "unknown").lower()
        
        if not raw_name or raw_name.lower() == "n/a":
            print(f"Skipping document from {filename} - no valid name found.")
            continue  

        # 1. Find existing student (fuzzy match) - checks the temporary session table
        matching_student_key = find_matching_student(raw_name, source_table=newly_consolidated_table)
        
        # If no match in session, check permanent global table for merging old with new
        if not matching_student_key:
             matching_student_key = find_matching_student(raw_name, source_table=student_table)

        
        if matching_student_key:
            # Update existing student record
            student_name = matching_student_key
            
            # Get existing doc type (for get_best_name priority)
            # Check the global table if it's not in the session table yet
            current_source_table = newly_consolidated_table if matching_student_key in newly_consolidated_table else student_table
            existing_doc_type = current_source_table[matching_student_key].get("documents", [""])[-1].split("(")[-1].replace(")", "")

            # Check if we need to update to a better name (priority-based)
            better_name = get_best_name(matching_student_key, raw_name, existing_doc_type, doc_type)
            
            if better_name != matching_student_key:
                # Use the permanent record if it exists, otherwise use session copy
                record_to_copy = newly_consolidated_table.get(matching_student_key, student_table.get(matching_student_key, {})).copy()

                # Create new entry with better name and copy data
                newly_consolidated_table[better_name] = record_to_copy
                newly_consolidated_table[better_name]["name"] = better_name
                
                # Remove the old key if it was created in this session
                if matching_student_key in newly_consolidated_table:
                    del newly_consolidated_table[matching_student_key]
                
                student_name = better_name
            
            # Ensure the record we are updating is in the session table
            if student_name not in newly_consolidated_table:
                # Copy from global table to session table for merging
                newly_consolidated_table[student_name] = student_table[student_name].copy()
            
        else:
            # Create new student record in the temporary table
            student_name = raw_name
            newly_consolidated_table[student_name] = {
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
        
        # 2. Update fields based on document data (All keys are lowercase)
        student_record = newly_consolidated_table[student_name]
        
        # Current document to track (avoid duplicates)
        doc_entry = f"{filename} ({doc_type})"
        if doc_entry not in student_record["documents"]:
            student_record["documents"].append(doc_entry)
        
        # Map extracted fields (source) to table fields (target)
        field_mappings = {
            "dl_number": "dl_number", "date_of_birth": "date_of_birth", 
            "skills_test_date": "skills_test_date", "exam_result": "exam_result", 
            "pt_dee_d_number": "de_964_number", "adee_number": "adee_number",
            "ittd_completion_date": "ittd_completion_date", "itad_completion_date": "itad_completion_date"  
        }
        
        single_source_fields = ["ittd_completion_date", "itad_completion_date", "de_964_number", "adee_number", "exam_result"]

        for source_field, table_field in field_mappings.items():
            if source_field in extracted_data and extracted_data[source_field]:
                value = str(extracted_data[source_field]).strip()
                
                if not value or value.lower() in ["n/a", "not found"]:
                    continue
                
                current_value = student_record.get(table_field)
                
                # Logic for single-source fields (fill if empty)
                if table_field in single_source_fields:
                    if not current_value or current_value == "N/A":
                        student_record[table_field] = value
                        continue 
                
                # Default logic for shared/multi-source fields
                if not current_value or current_value == "N/A":
                    student_record[table_field] = value
                elif "skills_test_date" in table_field and "⚠️" in str(current_value) and "⚠️" not in str(value):
                    student_record[table_field] = value
                elif "skills_test_date" not in table_field and len(str(value)) > len(str(current_value)):
                    student_record[table_field] = value

    return newly_consolidated_table

def merge_record(data_to_merge):
    """
    PERSISTENCE CORE: Merges one fully assembled record (data_to_merge) into the global student_table and saves.
    """
    global student_table
    
    raw_name = data_to_merge.get("name", "")
    if not raw_name:
        return
        
    # We must use the global table for matching here
    matching_student = find_matching_student(raw_name)
    
    student_name = matching_student if matching_student else raw_name
    
    # Use the existing record if found, otherwise use the newly assembled record
    if student_name not in student_table:
        student_table[student_name] = data_to_merge.copy()
    else:
        # Merge the fields from the new data into the existing data 
        current_record = student_table[student_name]
        
        # Merge field-by-field (All keys are lowercase)
        for key, new_value in data_to_merge.items():
            if key in ["documents", "name"]:
                continue
            
            existing_value = current_record.get(key)
            
            if new_value and new_value != "N/A" and new_value != existing_value:
                # Simple priority: New data overwrites N/A or if it's longer
                if not existing_value or existing_value == "N/A" or len(str(new_value)) > len(str(existing_value)):
                     current_record[key] = new_value

        # Merge documents list (just append the new sources)
        for doc in data_to_merge.get("documents", []):
            if doc not in current_record["documents"]:
                 current_record["documents"].append(doc)
    
    save_student_table()

def persist_table(newly_consolidated_table):

    added_count = 0
    
    # 1. Merge new students into the global student_table
    for name, new_record in newly_consolidated_table.items():
        merge_record(new_record)
        added_count += 1

    return added_count

def save_preview(preview_records):
    """
    GUI HOOK: Collects the raw records from the preview table and sends them 
    to the persistence engine.
    """
    consolidated_table = {}
    
    for record in preview_records:
        raw_data = record.get("_raw_data") 
        if raw_data:
            name = raw_data.get("name")
            if name:
                consolidated_table[name] = raw_data
        else:
            print(f"Skipping record for {record.get('name', 'unknown')} - no valid raw_data for persistence")

    if consolidated_table:
        added_count = persist_table(consolidated_table)
        return added_count
        
    return 0

# === Table Engine Calculators === 

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

def find_matching_student(new_name, source_table=None):
    """
    Find existing student that matches the new name using first+last name matching.
    Can search against a specified source_table (like the session's temp table)
    or against the global student_table if none is provided.
    """
    target_table = source_table if source_table is not None else student_table
    
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
    
    for existing_name in target_table.keys():
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
        
        # MATCHING STRATEGY: (Same as existing logic)
        
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
                score = 30 + (overlap * 5)
        
        # Strategy 4: Fuzzy matching for OCR errors
        if score == 0:
            matched_words = 0
            for new_word in new_words[:2]:
                for existing_word in existing_words[:2]:
                    word_distance = levenshtein_distance(new_word, existing_word)
                    word_max_len = max(len(new_word), len(existing_word))
                    
                    if word_distance <= 2 and word_max_len > 3 and word_distance / word_max_len <= 0.25:
                        matched_words += 1
                        break
            
            if matched_words >= 2:
                score = 20
        
        # Update best match if this score is better
        if score > best_score:
            best_match = existing_name
            best_score = score
    
    return best_match if best_score >= 20 else None

PRIORITY_SCORES = {
    # High Priority: Primary sources for student ID and name (100)
    "dl-40": 100,
    "dl-40 front": 100,
    "driver license": 100,
    
    # Medium Priority: State-issued certs with reliable name info (80)
    "de-964 certificate": 80,
    "adee-1317 certificate": 80, # CONVERTED to lowercase
    
    # Low Priority: Impact certs often have abbreviated names or less official formatting (50)
    "teen impact certificate": 50,
    "adult impact certificate": 50,
    "unknown": 0,
}

def get_doc_priority(doc_type):
    """Maps document type to a consistent priority score."""
    if not doc_type:
        return 0
    # doc_type flowing in is already lowercase from pair_dl40_pages
    clean_type = doc_type.lower().strip()
    return PRIORITY_SCORES.get(clean_type, 0)

def get_name_completeness_score(name):
    """
    Calculates a score based on name length and word count (higher is better).
    """
    if not name:
        return (0, 0)
    
    clean = clean_name(name) 
    words = clean.split()
    word_count = len(words)
    char_count = len(clean) 
    
    # Return as a tuple for lexicographical comparison: (word_count, char_count)
    return (word_count, char_count)

def get_best_name(name1, name2, doc_type1=None, doc_type2=None):
    """
    Return the best name between two options based on a weighted comparison:
    1. Document Priority (Higher score wins).
    2. Name Completeness (Word Count/Length tie-breaker if priorities are equal).
    """
    if not name1:
        return name2
    if not name2:
        return name1
    
    # 1. Determine Document Priority Score
    score1 = get_doc_priority(doc_type1)
    score2 = get_doc_priority(doc_type2)
    
    # If priorities are unequal, the name from the higher-priority document wins immediately.
    if score1 > score2:
        return name1 
    if score2 > score1:
        return name2
        
    # 2. If Document Priority is equal, use Name Completeness as the tie-breaker.
    
    completeness1 = get_name_completeness_score(name1)
    completeness2 = get_name_completeness_score(name2)
    
    # Compare the tuples (word_count, char_count): 
    if completeness1 > completeness2:
        return name1
    if completeness2 > completeness1:
        return name2
    
    # 3. If everything is equal, keep the original name (name1).
    return name1


# === Table Commands ===

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
            'ITTD Date': record['ittd_completion_date'] or 'N/A',
            'ITAD Date': record['itad_completion_date'] or 'N/A'
        })
    
    # Create DataFrame and export to Excel
    df = pd.DataFrame(export_data)
    filename = "student_documents_table.xlsx"
    
    try:
        df.to_excel(filename, index=False)
        print(f"Table exported successfully to {filename}")
    except Exception as e:
        print(f"Error exporting to Excel: {e}")


# === Table Storage ===

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
