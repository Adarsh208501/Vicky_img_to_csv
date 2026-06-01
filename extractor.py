import os
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("execution.log"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  EXTRACTOR
# ─────────────────────────────────────────────
class GeminiExtractor:
    # Real, publicly available Gemini models (best → fastest)
    MODELS = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    PROMPT = """
You are a PRECISION DATA EXTRACTION ENGINE.

TASK: Extract every table from this document into a single JSON nested list.

STRICT RULES:
1. FIRST ROW = exact column headers.
   - Merge hierarchical/multi-level headers with " > " e.g. "Sales > Q1".
   - If two header rows exist, combine them: "Region" + "North" → "Region > North".
2. CAPTURE EVERY ROW — including sub-totals, totals, grand-totals, and footnote rows.
   Count the rows you see, then make sure your output has exactly that many data rows.
3. PRESERVE VALUES EXACTLY:
   - Do NOT round numbers.
   - Keep currency/percent symbols (₹, $, %, etc.) as part of the string.
   - Keep dashes, asterisks, or any special characters.
4. EMPTY CELLS → use null (not "" and not 0).
5. MERGED CELLS that span columns → repeat the value in every column it covers.
6. OUTPUT FORMAT: compact nested JSON list, NOTHING ELSE — no markdown, no explanation.
7. DUPLICATE ROW NAMES ARE INTENTIONAL AND VALID — rows like "E-SALE TOTAL" or 
   "DEALER TOTAL" may appear multiple times under different parent groups. 
   You MUST include ALL of them. NEVER merge, skip, or deduplicate rows 
   that share the same name. Each row in the document = exactly one row in output.

   EXAMPLE (generic -- your actual headers and values will differ):
[
  ["__seq__", "Col1",    "Col2", "Col3"],
  [1,         "Group A", 100,    null  ],
  [2,         "Total",   500,    "10%" ],
  [3,         "Group B", 200,    null  ],
  [4,         "Total",   700,    "15%" ],
  [5,         "Total",   1200,   "25%" ]
]
"""

    def __init__(self, api_key: str, primary_model: str = "gemini-2.5-pro"):
        logger.info(f"Initializing extractor | Primary model: {primary_model}")
        self.client = genai.Client(api_key=api_key)

        # Build ordered fallback list starting from user's choice
        ordered = [primary_model] + [m for m in self.MODELS if m != primary_model]
        self.model_queue = ordered

        self.config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            max_output_tokens=65536,      # ← was 8192; big tables need more room
        )

    # ── File preparation ──────────────────────────────────────────────────────
    def _prepare(self, path: Path) -> Any:
        if path.suffix.lower() == ".pdf":
            logger.info(f"  PDF detected → uploading via Files API: {path.name}")
            return self.client.files.upload(path=str(path))
        else:
            logger.info(f"  Image detected → loading with PIL: {path.name}")
            return Image.open(path)

    def _cleanup(self, obj: Any):
        if hasattr(obj, "name") and "files/" in str(obj.name):
            try:
                self.client.files.delete(name=obj.name)
                logger.info("  Uploaded file deleted from Google servers.")
            except Exception:
                pass

    # ── JSON parsing ──────────────────────────────────────────────────────────
    def _parse(self, raw: str) -> List[Dict[str, Any]]:
        """Convert nested-list JSON from the model into list-of-dicts."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"  JSON decode error: {e}")
            return []

        # Case 1 — nested list  [[headers], [row], ...]
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], list):
            headers = [str(h) for h in parsed[0]]
            rows = []
            for i, cells in enumerate(parsed[1:], start=2):
                if not isinstance(cells, list):
                    continue
                # Pad short rows; warn on long rows
                if len(cells) < len(headers):
                    cells = cells + [None] * (len(headers) - len(cells))
                elif len(cells) > len(headers):
                    logger.warning(
                        f"  Row {i} has {len(cells)} cells but only {len(headers)} "
                        f"headers — extra cells dropped."
                    )
                row_dict = {headers[j]: cells[j] for j in range(len(headers))}
                row_dict["_row_num"] = len(rows) + 1  # unique row identifier
                rows.append(row_dict)
            logger.info(f"  Parsed {len(rows)} rows from compact nested list.")
            return rows

        # Case 2 — list of dicts
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            logger.info(f"  Parsed {len(parsed)} rows from list-of-objects.")
            return parsed

        # Case 3 — dict wrapper  {"data": [...]}
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list) and v:
                    logger.info("  Found list inside dict wrapper — recursing.")
                    return self._parse(json.dumps(v))

        logger.warning("  Could not recognise JSON structure.")
        return []

    # ── Truncation check ──────────────────────────────────────────────────────
    def _was_truncated(self, response) -> bool:
        try:
            return response.candidates[0].finish_reason.name == "MAX_TOKENS"
        except Exception:
            return False

    # ── Validation ────────────────────────────────────────────────────────────
    def _validate(self, data: List[Dict], label: str) -> bool:
        if not data:
            logger.error(f"  [{label}] VALIDATION FAILED — no rows extracted.")
            return False

        col_counts = {len(r) for r in data}
        if len(col_counts) > 1:
            logger.warning(
                f"  [{label}] Inconsistent column counts across rows: {col_counts} "
                f"— possible merged-cell misalignment."
            )

        empty = sum(1 for r in data if all(v is None or v == "" for v in r.values()))
        if empty > len(data) * 0.15:
            logger.warning(
                f"  [{label}] {empty}/{len(data)} rows are nearly empty — "
                f"possible extraction issue."
            )

        logger.info(
            f"  [{label}] Validation OK — {len(data)} rows × {len(data[0])} columns."
        )
        return True

    # ── Two-pass cross-check ──────────────────────────────────────────────────
    def _two_pass(self, content_obj: Any, label: str) -> List[Dict[str, Any]]:
        """Run extraction twice and pick the result with more rows."""
        results = []
        for attempt in range(2):
            logger.info(f"  Two-pass extraction — pass {attempt + 1}/2")
            data = self._single_extract(content_obj, label)
            if data:
                results.append(data)

        if not results:
            return []
        if len(results) == 1:
            return results[0]

        if len(results[0]) != len(results[1]):
            winner = max(results, key=len)
            logger.warning(
                f"  Pass 1 → {len(results[0])} rows | Pass 2 → {len(results[1])} rows. "
                f"Using larger result ({len(winner)} rows)."
            )
            return winner

        logger.info("  Both passes agree on row count — using pass 1.")
        return results[0]

    # ── Single extraction attempt ─────────────────────────────────────────────
    def _single_extract(self, content_obj: Any, label: str) -> List[Dict[str, Any]]:
        for model_id in self.model_queue:
            logger.info(f"  Trying model: {model_id}")
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=[content_obj, self.PROMPT],
                        config=self.config,
                    )

                    if not response.text:
                        logger.warning(f"  Empty response from {model_id} (attempt {attempt+1})")
                        continue

                    if self._was_truncated(response):
                        logger.warning(
                            f"  Response TRUNCATED by token limit on {model_id}. "
                            f"Table may be incomplete."
                        )

                    data = self._parse(response.text)
                    if data:
                        logger.info(f"  ✓ Success with {model_id}")
                        return data
                    else:
                        logger.warning(f"  Parsed 0 rows from {model_id} (attempt {attempt+1})")

                except KeyboardInterrupt:
                    logger.warning(f"  Ctrl+C — skipping {model_id}")
                    break

                except Exception as e:
                    err = str(e).lower()

                    if "429" in err or "resource_exhausted" in err:
                        if "pro" in model_id:
                            logger.warning(f"  {model_id} quota hit — instant failover.")
                            break
                        wait = 15 * (attempt + 1)
                        logger.warning(f"  Quota hit on {model_id}. Waiting {wait}s…")
                        try:
                            time.sleep(wait)
                        except KeyboardInterrupt:
                            break

                    elif "503" in err or "unavailable" in err:
                        if "pro" in model_id:
                            logger.warning(f"  {model_id} busy — instant failover.")
                            break
                        logger.warning(f"  Server busy ({model_id}). Retrying in 10s…")
                        try:
                            time.sleep(10)
                        except KeyboardInterrupt:
                            break

                    else:
                        logger.error(f"  {model_id} error: {e}")
                        break  # move to next model

            logger.info(f"  Moving to next model…")

        return []

    # ── Public entry point ────────────────────────────────────────────────────
    def extract_table(self, file_path: Path, two_pass: bool = False) -> List[Dict[str, Any]]:
        logger.info(f"━━━━ Extracting: {file_path.name} ━━━━")
        content_obj = self._prepare(file_path)

        try:
            if two_pass:
                data = self._two_pass(content_obj, file_path.name)
            else:
                data = self._single_extract(content_obj, file_path.name)

            self._validate(data, file_path.name)
            return data
        finally:
            self._cleanup(content_obj)


# ─────────────────────────────────────────────
#  CSV WRITER
# ─────────────────────────────────────────────
class CSVGenerator:
    @staticmethod
    def save(data: List[Dict[str, Any]], output_path: Path):
        if not data:
            logger.warning("No data to save.")
            return

        try:
            df = pd.DataFrame(data)
            df = df.reset_index(drop=True)  # ensure unique index, never deduplicate
            if "_row_num" in df.columns:
              df = df.drop(columns=["_row_num"])
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print("\n" + "=" * 60)
            print("  SUCCESS — Extraction Complete!")
            print(f"  Rows    : {len(df)}")
            print(f"  Columns : {len(df.columns)}")
            print(f"  Saved   : {output_path.absolute()}")
            print("=" * 60 + "\n")
        except Exception as e:
            logger.error(f"Save failed: {e}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="High-accuracy Gemini table extractor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gemini_extractor.py invoice.pdf
  python gemini_extractor.py scan.jpg -o output/result.csv
  python gemini_extractor.py ./docs/ -o ./results/ --two-pass
  python gemini_extractor.py table.png --model gemini-2.5-flash
        """,
    )
    parser.add_argument("input", help="Path to image/PDF or a directory of them.")
    parser.add_argument(
        "-o", "--output",
        help="Output CSV path (single file) or directory (batch). Default: results/",
        default="results",
    )
    parser.add_argument(
        "--model",
        help="Primary model to use. Default: gemini-2.5-pro",
        default="gemini-2.5-pro",
        choices=["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"],
    )
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Extract twice and pick the result with more rows (slower but more reliable).",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.critical("GEMINI_API_KEY not set. Add it to a .env file.")
        return

    extractor = GeminiExtractor(api_key, primary_model=args.model)
    writer = CSVGenerator()

    input_path = Path(args.input)
    output_base = Path(args.output)
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tiff"}

    # Collect files
    if input_path.is_dir():
        files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in supported)
    elif input_path.suffix.lower() in supported:
        files = [input_path]
    else:
        logger.error(f"Unsupported file type: {input_path.suffix}")
        return

    if not files:
        logger.error("No supported files found.")
        return

    logger.info(f"Processing {len(files)} file(s) | two-pass={args.two_pass}")

    for file_path in files:
        data = extractor.extract_table(file_path, two_pass=args.two_pass)

        if input_path.is_dir():
            out_file = output_base / f"{file_path.stem}.csv"
        else:
            if output_base.suffix == ".csv":
                out_file = output_base
            else:
                out_file = output_base / f"{file_path.stem}.csv"

        writer.save(data, out_file)
        time.sleep(1)  # polite pacing between files


if __name__ == "__main__":
    main()
