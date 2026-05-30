import os
import io
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("hybrid_execution.log")]
)
logger = logging.getLogger(__name__)

class HuggingFaceOCR:
    """Stage 1: Specialized visual character recognition via LightOnOCR-2-1B (Local)."""
    
    def __init__(self, model_id: str = "lightonai/LightOnOCR-2-1B"):
        self.model_id = model_id
        self.device = "cpu"
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return True
        
        logger.info(f"PHASE: OCR Initialization - Loading {self.model_id} (Local CPU)")
        try:
            from transformers import AutoProcessor, AutoModel
            import torch
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            ).to(self.device).eval()
            return True
        except Exception as e:
            logger.error(f"  -> Local Model Load Failed: {e}")
            return False

    def process(self, image_path: Path, gemini_client: genai.Client) -> str:
        """Extracts text. Tries LightOnOCR local, then Gemini Vision with retries."""
        logger.info(f"PHASE: OCR Scan - {image_path.name}")
        
        # 1. Local OCR Pathway
        if self._load_model():
            try:
                import torch
                image = Image.open(image_path).convert("RGB")
                # Ensure processor exists before call
                if self.processor:
                    inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                    logger.info("  -> Running Local Character Recognition...")
                    with torch.no_grad():
                        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
                        decoded = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                        if decoded and len(decoded) > 0:
                            logger.info(f"  -> Local OCR Success ({len(decoded[0])} chars).")
                            return decoded[0]
            except Exception as e:
                logger.warning(f"  -> Local OCR Error: {e}")

        # 2. Gemini OCR Pathway (Robust Multi-Model Backup)
        logger.info("  -> Stage 1 Fallback: Initiating Gemini Visual Character Recognition...")
        # Using verified working model names from extractor.py
        ocr_models = ["gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-2.0-flash"]
        for model_id in ocr_models:
            for attempt in range(4):
                logger.info(f"  -> OCR via {model_id} (Attempt {attempt+1})")
                try:
                    img = Image.open(image_path)
                    response = gemini_client.models.generate_content(
                        model=model_id,
                        contents=[img, "Transcribe all text and tables from this image perfectly."]
                    )
                    if response.text and len(response.text) > 20:
                        logger.info(f"  -> Stage 1 Success via {model_id}.")
                        return response.text
                except Exception as e:
                    if "429" in str(e):
                        wait_sec = 20 * (attempt + 1)
                        logger.warning(f"  -> Quota Hit ({model_id}). Waiting {wait_sec}s...")
                        time.sleep(wait_sec)
                    else:
                        break
        return ""

class GeminiRefiner:
    """Stage 2: Structural reasoning for high-fidelity CSV generation."""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.fallback_models = ["gemini-2.0-flash", "gemini-3.1-pro-preview", "gemini-2.5-pro"]
        self.config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )

    def refine(self, ocr_text: str) -> List[Dict[str, Any]]:
        """Takes raw text and structures it into a high-fidelity CSV format."""
        logger.info("PHASE: Stage 2 - Structural Refinement")
        if not ocr_text or len(ocr_text) < 10:
            return []
        
        prompt = f"Structure this OCR text into a COMPACT NESTED LIST: [[headers], [row1], ...]\n\nDATA:\n{ocr_text}"

        for model_id in self.fallback_models:
            for attempt in range(3):
                logger.info(f"  -> Reasoning with {model_id} (Pass {attempt+1})")
                try:
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=[prompt],
                        config=self.config
                    )
                    data = json.loads(response.text)
                    if isinstance(data, list) and len(data) > 1:
                        headers = [str(h) for h in data[0]]
                        refined = []
                        for row in data[1:]:
                            if isinstance(row, list):
                                refined.append({headers[i]: row[i] for i in range(min(len(headers), len(row)))})
                        return refined
                except Exception as e:
                    if "429" in str(e):
                        time.sleep(10)
                    else:
                        break
        return []

class HybridPipeline:
    def __init__(self, gemini_key: str):
        self.ocr_engine = HuggingFaceOCR()
        self.refiner_engine = GeminiRefiner(gemini_key)

    def execute(self, input_file: Path, output_file: Path):
        logger.info(f"=== HYBRID EXTRACTION PIPELINE START: {input_file.name} ===")
        
        # STAGE 1: Visual Perception
        raw_output = self.ocr_engine.process(input_file, self.refiner_engine.client)
        
        # STAGE 2: Structural Intelligence
        structured_data = self.refiner_engine.refine(raw_output)
        
        # STAGE 3: CSV Marshalling
        if structured_data:
            df = pd.DataFrame(structured_data)
            df.to_csv(output_file, index=False)
            logger.info(f"=== EXTRACTION SUCCESS: {output_file} ===")
        else:
            logger.error("=== PIPELINE FAILURE: Extraction stalled at reasoning stage. ===")

def main():
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        logger.error("CRITICAL: GEMINI_API_KEY missing from .env")
        return

    parser = argparse.ArgumentParser(description="Hybrid OCR/LLM Extraction Platform.")
    parser.add_argument("input", help="Image/PDF file path.")
    parser.add_argument("-o", "--output", help="Output CSV name.", default="hybrid_results.csv")
    args = parser.parse_args()

    pipeline = HybridPipeline(gemini_key)
    pipeline.execute(Path(args.input), Path(args.output))

if __name__ == "__main__":
    main()
