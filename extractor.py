import os
import json
import logging
import argparse
import time
import io
import re
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("execution.log")
    ]
)
logger = logging.getLogger(__name__)

class GeminiExtractor:
    def __init__(self, api_key: str, primary_model: str = "gemini-3.1-pro-preview"):
        logger.info(f"PHASE: Initialization - Setting up Multi-Model Engine (Primary: {primary_model})")
        self.client = genai.Client(api_key=api_key)
        self.primary_model = primary_model
        # High-precision models prioritized
        self.fallback_models = [
            "gemini-3.1-pro-preview", 
            "gemini-3-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash"
        ]
        
        if self.primary_model in self.fallback_models:
            self.fallback_models.remove(self.primary_model)
        self.fallback_models.insert(0, self.primary_model)
        
        # Native JSON Mode for absolute structural fidelity
        self.config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            max_output_tokens=8192,
        )

    def _prepare_content(self, file_path: Path) -> Any:
        """Determines the best way to handle the file based on its type."""
        logger.info(f"PHASE: File Analysis - {file_path.name}")
        
        if file_path.suffix.lower() == ".pdf":
            logger.info("  -> PDF detected. Using Google Files API.")
            pdf_file = self.client.files.upload(path=str(file_path))
            return pdf_file
        else:
            logger.info("  -> Image detected. Processing with PIL.")
            return Image.open(file_path)

    def _parse_table_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parses native JSON output (compact or object-based) into structured data."""
        logger.info("PHASE: Table Structural Analysis")
        try:
            raw_data = json.loads(text)
            
            # CASE 1: Nested List (Compact Format: [[Header], [Row1], [Row2], ...])
            if isinstance(raw_data, list) and len(raw_data) > 0:
                if isinstance(raw_data[0], list):
                    headers = [str(h) for h in raw_data[0]]
                    data = []
                    for row_cells in raw_data[1:]:
                        if isinstance(row_cells, list):
                            # Map cells to headers safely
                            row_dict = {headers[i]: row_cells[i] for i in range(min(len(headers), len(row_cells)))}
                            data.append(row_dict)
                    logger.info(f"  -> Extracted {len(data)} rows via Compact JSON.")
                    return data
            
            # CASE 2: List of Objects (Standard Format)
            if isinstance(raw_data, list):
                logger.info(f"  -> Extracted {len(raw_data)} rows via Object JSON.")
                return raw_data
                
            # CASE 3: Object with a data key
            if isinstance(raw_data, dict):
                for key in raw_data:
                    if isinstance(raw_data[key], list):
                        return self._parse_table_from_text(json.dumps(raw_data[key]))
            
            return []
        except Exception as e:
            logger.error(f"  -> JSON Parse Error: {e}")
            return []

    def extract_table(self, file_path: Path) -> List[Dict[str, Any]]:
        content_obj = self._prepare_content(file_path)
        
        prompt = """
        ACT AS A PRECISION DATA EXTRACTION ENGINE.
        Extract the table from this document as a COMPACT NESTED LIST.
        
        FORMAT EXAMPLE:
        [
          ["SI_NO", "Buyer", "Metric - MOU", ...],
          [1, "COMPANY A", 50, ...],
          [2, "COMPANY B", 30, ...]
        ]
        
        REQUIREMENTS:
        1. THE FIRST ELEMENT MUST BE THE HEADER LIST. Merge hierarchical headers (e.g. "Metric - MOU").
        2. EXHAUSTIVE: Capture every single row. Do not skip the Totals at the bottom.
        3. COMPACT: Use the nested list format above to save space and prevent truncation.
        4. ACCURACY: Values must be exactly as shown in the image.
        """

        for model_id in self.fallback_models:
            logger.info(f"PHASE: AI Extraction - Targeting Model: {model_id}")
            max_retries = 3
            
            for attempt in range(max_retries):
                logger.info(f"  -> Attempt {attempt + 1}/{max_retries} for {model_id}")
                try:
                    # Provide feedback on how to skip
                    if attempt > 0 or model_id != self.fallback_models[0]:
                        logger.info("  -> TIP: Press Ctrl+C to skip this model and try a fallback engine.")

                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=[content_obj, prompt],
                        config=self.config
                    )
                    
                    if not response.text:
                        continue

                    data = self._parse_table_from_text(response.text)
                    if data:
                        self._cleanup(content_obj)
                        return data

                except KeyboardInterrupt:
                    logger.warning(f"  -> USER INTERRUPT: Skipping {model_id}...")
                    break # Break out of attempt loop to try next model
                except Exception as e:
                    err_txt = str(e).lower()
                    if "429" in err_txt or "resource_exhausted" in err_txt:
                        # Instant fallback for Pro to keep things "smooth"
                        if "pro" in model_id:
                            logger.warning(f"  -> {model_id} QUOTA HIT. Instant failover to high-availability engine...")
                            break
                        
                        wait = 15 * (attempt + 1)
                        logger.warning(f"  -> QUOTA HIT on {model_id}. Waiting {wait}s...")
                        try:
                            time.sleep(wait)
                        except KeyboardInterrupt:
                            logger.warning(f"  -> USER INTERRUPT: Skipping {model_id}...")
                            break 
                    elif "503" in err_txt or "unavailable" in err_txt:
                        if "pro" in model_id:
                            logger.warning(f"  -> {model_id} BUSY. Instant failover to high-availability engine...")
                            break
                        logger.warning(f"  -> SERVER BUSY ({model_id}). Retrying in 10s...")
                        try:
                            time.sleep(10)
                        except KeyboardInterrupt:
                            logger.warning(f"  -> USER INTERRUPT: Skipping {model_id}...")
                            break
                    else:
                        logger.error(f"  -> Model {model_id} failed: {e}")
                        break # Try next model
            
            logger.info(f"PHASE: Failover - Moving to next engine...")

        self._cleanup(content_obj)
        return []

    def _cleanup(self, content_obj: Any):
        if hasattr(content_obj, 'name') and "files/" in content_obj.name:
            try:
                self.client.files.delete(name=content_obj.name)
            except: pass

class CSVGenerator:
    @staticmethod
    def to_csv(data: List[Dict[str, Any]], output_path: Path):
        if not data:
            logger.warning("PHASE: Output Generation - No data found.")
            return

        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print("\n" + "="*60)
            print(f"SUCCESS: Extraction Complete!")
            print(f"RESULTS SAVED TO: {output_path.absolute()}")
            print("="*60 + "\n")
            logger.info(f"PHASE: Output Generation - SAVED: {output_path.name}")
        except Exception as e:
            logger.error(f"PHASE: Output Generation - FAILED: {e}")

def main():
    parser = argparse.ArgumentParser(description="High-Reliability Gemini Extraction Tool.")
    parser.add_argument("input", help="Path to input image/PDF or directory.")
    parser.add_argument("-o", "--output", help="Output CSV path or directory.", default="results")
    parser.add_argument("--model", help="Primary model ID.", default="gemini-3.1-pro-preview")
    
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.critical("GEMINI_API_KEY not found in .env")
        return

    extractor = GeminiExtractor(api_key, primary_model=args.model)
    generator = CSVGenerator()

    input_path = Path(args.input)
    output_base = Path(args.output)
    
    files = list(input_path.glob("*")) if input_path.is_dir() else [input_path]
    supported = ('.pdf', '.png', '.jpg', '.jpeg')
    files = [f for f in files if f.suffix.lower() in supported]

    logger.info(f"PHASE: Deployment - Processing {len(files)} file(s)")
    
    for file_path in files:
        data = extractor.extract_table(file_path)
        
        if input_path.is_dir():
            output_file = output_base / f"{file_path.stem}.csv"
            output_base.mkdir(parents=True, exist_ok=True)
        else:
            output_file = output_base if output_base.suffix == '.csv' else Path(f"{file_path.stem}.csv")
        
        generator.to_csv(data, output_file)
        time.sleep(1) # Base pacing

if __name__ == "__main__":
    main()
