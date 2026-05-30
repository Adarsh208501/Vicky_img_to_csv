# Image & PDF to CSV Extraction Pipeline

A robust, AI-powered data extraction pipeline that reliably converts tables and text from images (PNG, JPEG) and PDF files into structured CSV formats.

The project features two distinct extraction engines:
1. **Gemini Extractor (`extractor.py`)**: A high-reliability cloud-based extraction tool using Google's Gemini Vision models (Gemini 3.1 Pro/2.5 Pro/2.0 Flash) with native JSON structuring and automatic model failover.
2. **Hybrid Extractor (`hybrid_extractor.py`)**: A dual-stage pipeline that uses local character recognition (HuggingFace `LightOnOCR-2-1B`) for raw text extraction, falling back to Gemini Vision if needed, and uses Gemini exclusively for structural reasoning and CSV marshalling.

---

## 📋 Prerequisites

- **Python 3.8+** installed on your system.
- Git (to clone the repository).
- (Optional) CUDA-enabled GPU if you wish to run the local HuggingFace OCR model with hardware acceleration.

---

## 🚀 Step-by-Step Installation

### 1. Clone the Repository
Clone the code to your local machine and navigate into the directory:
```bash
git clone https://github.com/Adarsh208501/Vicky_img_to_csv.git
cd Vicky_img_to_csv
```

### 2. Create a Virtual Environment (Recommended)
Isolate your dependencies by creating a Python virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Core Dependencies
Install the required packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

*(Note: If you plan to use the `hybrid_extractor.py` which runs a local AI model for OCR, you must also install PyTorch and Transformers):*
```bash
pip install torch transformers
```

### 4. Environment Setup (.env)
This project relies on the Google Gemini API for structural reasoning and high-fidelity vision processing. 
Create a `.env` file in the root of your project directory:

**`.env` file contents:**
```env
GEMINI_API_KEY=your_actual_google_gemini_api_key_here
```
*(You can get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

---

## 💻 Usage Instructions

### Using the Gemini Extractor (Cloud-based)
Provides highly accurate table extraction using Google's generative models with automatic rate-limit handling and model failovers.

**Command:**
```bash
python extractor.py <path_to_input_file_or_directory> [-o <output_path>] [--model <model_id>]
```

**Examples:**
```bash
# Extract from a single image
python extractor.py 2.2.png

# Extract and save to a specific CSV file
python extractor.py report.pdf -o final_report.csv

# Process an entire folder of images
python extractor.py my_images_folder -o output_csvs_folder
```

### Using the Hybrid Extractor (Local + Cloud)
Runs a two-stage pipeline. Stage 1 attempts local OCR (CPU/GPU) to save API costs, and Stage 2 uses Gemini strictly for logical data structuring.

**Command:**
```bash
python hybrid_extractor.py <path_to_input_file> [-o <output_csv>]
```

**Examples:**
```bash
python hybrid_extractor.py 2.2.png -o hybrid_results.csv
```

---

## 🛠️ Code Structure

- **`extractor.py`**: The primary extraction engine leveraging the `google-genai` SDK. Features dynamic payload handling (using the Files API for PDFs and PIL for images) and robust error handling (managing 429 Quota errors and 503 Server Busy errors).
- **`hybrid_extractor.py`**: The split-architecture engine. It defines a `HuggingFaceOCR` class for perception and a `GeminiRefiner` class for structural intelligence.
- **`requirements.txt`**: Standard Python dependency file.
- **`execution.log` & `hybrid_execution.log`**: Professional telemetry and logging files generated automatically during runtime to monitor extraction steps and model fallbacks.
