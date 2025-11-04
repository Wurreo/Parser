# Document Parser for Driving School Records

This application is a data extraction and management tool designed to process documents from driving schools, specifically focusing on merging data from multiple sources into a single, standardized record per student. The system uses a LlamaExtract AI agent for high-accuracy data extraction and a Flet-based GUI for ease of use.

## Features

* **Multipage Document Processing:** Processes single or multi-page PDF/image documents without requiring manual splitting.
* **Intelligent Consolidation:** Merges student records from different documents (e.g., Driver's License, DL-40, Impact Certificates) based on name-matching and prioritization logic.
* **DL-40 Date Resolution:** Automatically compares conflicting Skills Test Date and Exam Date from DL-40 front and back pages, selecting the most recent date and flagging the entry with a warning if a conflict is detected.
* **Data Persistence:** Student records are automatically saved to disk (`student_data.json`) and persist across program sessions.
* **Monthly Reporting:** Groups finalized student data by the Skills Test Date month for easy export to Excel.
* **Unknown Records Handling:** Automatically isolates records missing a Skills Test Date into a separate 'Unknown' table for manual review.

## Supported Document Types

The program is trained to classify and extract data from the following documents:

* **DL-40 Front**
* **DL-40 Back**
* **Driver License**
* **DE-964 Certificate**
* **ADEE-1317 Certificate**
* **Teen Impact Certificate**
* **Adult Impact Certificate**

## Setup and Installation

## Setup and Installation (Linux)

Getting the Document Parser running is straightforward, assuming you have Python 3 and a `.venv` set up.

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Name]
    ```

2.  **Set up Virtual Environment:**
    If you don't have a virtual environment (`venv`) yet, create one.
    ```bash
    python3 -m venv venv
    ```

3.  **Activate Virtual Environment:**
    You must run the program from within the virtual environment.
    ```bash
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    You will need to install the required Python packages into your environment (e.g., `flet`, `pandas`, `llama-cloud-services`).
    ```bash
    # Assuming you have a requirements.txt file:
    pip install -r requirements.txt 
    ```

5.  **API Key Configuration:**
    Your program requires the Llama Cloud API key to function.

    * Ensure you have a file named **`.env`** in your project's `src` directory (as shown in your file structure).
    * Edit this file to include your API key:

    ```
    LLAMA_CLOUD_API_KEY="[YOUR_API_KEY_HERE]"
    ```

## Usage

1.  **Start the GUI:**
    ```bash
    python main.py
    ```
    The application will launch maximized.

2.  **Processing:** Load documents into the `dat/` folder or select a folder via the GUI, then click "Start Parse".

## Usage

1.  **Start the GUI:**
    ```bash
    python main.py
    ```
2.  **Processing:** Use the "Upload Files" or "Select Folder" buttons on the Home tab to load documents into the system.
3.  **Extraction:** Click "Start Parse" to run the LlamaExtract agent. Results will appear in the preview table.
4.  **Commit Data:** Click "Add to Table" to move the consolidated records from the preview into the persistent monthly tables.
5.  **Review:** Use the "Tables" tab to view, delete, or export records grouped by month.
