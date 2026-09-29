#!/usr/bin/env python3
"""Correct speech recognition errors in the O test set and export to Excel.

Requirements: openai (with Responses API support), pandas, openpyxl
Usage:
    python infer_api_zero-shot.py

Enter your API key and edit the settings below before running.
Output columns match the Ns-LLM output file in preference/:
no / completion / large-v3-turbo / output / inference_time_sec
Results are saved to data_infer_testset/ next to this script. CER is calculated separately.
"""

import json
import re
from time import perf_counter
from pathlib import Path

import pandas as pd
from openai import OpenAI

# =====================================================================
# Manual settings
# =====================================================================
API_KEY = ""  # Enter your OpenAI API key here.
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "asr_test_wav_large-v3-turbo_O_testset.txt"
SR_MODEL_SIZE = "large-v3-turbo"
OUTPUT_DIR = BASE_DIR / "data_infer_testset"
GPT_MODEL = "gpt-5.4"
# Standard: "none" / thinking low: "low" / thinking medium: "medium"
# Use the same gpt-5.4 model and switch reasoning.effort for thinking modes.
REASONING_EFFORT = "none"
OUTPUT_PATH = OUTPUT_DIR / (
    f"infer_O_testset_{GPT_MODEL}_{REASONING_EFFORT}_cer_{SR_MODEL_SIZE}_api.xlsx"
)
MAX_OUTPUT_TOKENS = 8192  # Includes reasoning tokens
SYSTEM_MESSAGE = "You are an experienced nurse in a Japanese hospital who corrects automatic speech recognition (ASR) errors in dictated nursing records."


def build_prompt(o_text: str) -> str:
    """Build the user prompt from an O transcript (zero-shot verbatim correction)."""
    input_text = "The following text is the ASR output of the objective (O) section of a SOAP nursing record dictated by a nurse.\n"
    input_text += "Reconstruct the text that the nurse dictated.\n"
    input_text += "\n"
    input_text += "### Rules\n"
    input_text += "1. Identify words that are likely misrecognized, that is, phonetically similar but clinically implausible, and replace them with the medical or nursing term the nurse most likely dictated, using the clinical context.\n"
    input_text += "2. Do not change anything else. Keep the wording, word order, abbreviations, and level of detail as dictated. Do not expand abbreviations, add units or information that were not dictated, or rephrase the text.\n"
    input_text += "3. Remove fillers and disfluencies.\n"
    input_text += "4. Write terms in the notation used in Japanese nursing records (e.g., SpO2, HR, BP) and insert punctuation as in a written record.\n"
    input_text += "5. Output only the corrected Japanese text enclosed in <outputO> and </outputO>. Do not include explanations, headings, or Markdown code fences.\n"
    input_text += "\n"
    input_text += "### ASR output\n"
    input_text += o_text + "\n"

    return input_text


def main() -> None:
    records = json.loads(INPUT_PATH.read_text(encoding="utf-8-sig"))
    client = OpenAI(api_key=API_KEY)
    results = []

    for index, record in enumerate(records, start=1):
        o_text = record["prompt"].replace("<inputO>", "").replace("</inputO>", "").replace("<outputO>", "").strip()
        started = perf_counter()
        response = client.responses.create(
            model=GPT_MODEL,
            reasoning={"effort": REASONING_EFFORT},
            instructions=SYSTEM_MESSAGE,
            input=build_prompt(o_text),
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )
        inference_time = perf_counter() - started
        if response.status != "completed" or not response.output_text.strip():
            raise RuntimeError(f"no={record['no']}: Failed to get a valid response from the model. Status: {response.status}, Output: {response.output_text}")
        match = re.fullmatch(
            r"\s*<outputO>(.*?)</outputO>\s*", response.output_text, flags=re.DOTALL
        )
        if not match or not match.group(1).strip():
            raise ValueError(f"no={record['no']}: Unexpected output format")
        corrected_text = match.group(1).strip()
        if "<outputO>" in corrected_text or "</outputO>" in corrected_text:
            raise ValueError(f"no={record['no']}: Multiple or nested output tags")

        results.append({
            "no": record["no"],
            "completion": record["completion"].replace("<outputO>", "").replace("</outputO>", "").strip(),
            SR_MODEL_SIZE: o_text,
            "output": corrected_text,
            "inference_time_sec": inference_time,
        })
        print(f"[{index}/{len(records)}] no={record['no']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_excel(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
