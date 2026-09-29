#!/usr/bin/env python3
"""
Pairwise preference evaluation with an LLM-as-a-Judge.

Two result files are selected from the inference outputs, and for each record
the two candidate texts (one from each file) are compared by a judge LLM via
the OpenAI Responses API.

Main features:
- Two-candidate (A/B) comparison only
- Each record is judged in both orders (AB and BA)
- The results of both orders are aggregated to determine final_best
- Ties are accepted liberally
- Resumable from a checkpoint CSV
- The judge model defaults to gpt-5.4
"""

import argparse
import concurrent.futures as cf
import os
import re
import time
from typing import Dict, List, Tuple

import pandas as pd
import requests

from checkpoint_utils import canonical_no, load_checkpoint_data, save_checkpoint_data


# =====================================================================
# Settings
# =====================================================================

AMIVOICE_RESULT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "result_preference_testset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOURCE_FILES = {
    "amivoice_general": "amivoice_results_general.xlsx",
    "amivoice_medical": "amivoice_results_medical.xlsx",
    "gemma-9b_large-v3-turbo": "infer_O_testset_gemma-9b_cer_large-v3-turbo_vllm.xlsx",
}

DEFAULT_SOURCE1 = "gemma-9b_large-v3-turbo"
DEFAULT_SOURCE2 = "amivoice_medical"

DEFAULT_JUDGE_MODEL = "gpt-5.4"
DEFAULT_API_BASE = "https://api.openai.com/v1/responses"
DEFAULT_API_KEY = "YOUR_OPENAI_API_KEY"
DEFAULT_MAX_OUTPUT_TOKENS = 2500
DEFAULT_CHECKPOINT_INTERVAL = 50
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_TIMEOUT_SEC = 180
DEFAULT_MAX_RETRIES = 5
MAX_SAMPLES = None # 10


# =====================================================================
# Utilities
# =====================================================================

def is_amivoice(source_name: str) -> bool:
    return source_name.startswith("amivoice_")


def get_result_dir(source_name: str) -> str:
    return AMIVOICE_RESULT_DIR if is_amivoice(source_name) else RESULT_DIR


def normalize_testset_no(value) -> str:
    text = str(value).strip()
    match = re.fullmatch(r"[Tt](\d+)", text)
    if match:
        return str(int(match.group(1)))
    return text


def sort_key(value) -> tuple[int, str]:
    normalized = normalize_testset_no(value)
    return (0, int(normalized)) if normalized.isdigit() else (1, normalized)


def detect_whisper_column(df: pd.DataFrame) -> str:
    reserved = {"no", "completion", "output", "inference_time_sec"}
    whisper_cols = [col for col in df.columns if col not in reserved]
    if len(whisper_cols) != 1:
        raise ValueError(f"Unable to detect whisper column uniquely: {whisper_cols}")
    return whisper_cols[0]


def load_source(source_name: str) -> dict:
    filename = SOURCE_FILES[source_name]
    path = os.path.join(get_result_dir(source_name), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_excel(path)

    if is_amivoice(source_name):
        required = {"filename", "text"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        df = df.copy()
        df["no"] = df["filename"].map(normalize_testset_no)
        df = df.set_index("no")
        return {
            "name": source_name,
            "filename": filename,
            "df": df,
            "text_column": "text",
            "completion_column": None,
        }

    required = {"no", "completion", "output"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")

    _ = detect_whisper_column(df)
    df = df.copy()
    df["no"] = df["no"].map(normalize_testset_no)
    df = df.set_index("no")
    return {
        "name": source_name,
        "filename": filename,
        "df": df,
        "text_column": "output",
        "completion_column": "completion",
    }


def get_reference_completion(source1: dict, source2: dict, no: str) -> str:
    row1 = source1["df"].loc[no]
    row2 = source2["df"].loc[no]

    if source1["completion_column"] is not None:
        return str(row1[source1["completion_column"]])
    if source2["completion_column"] is not None:
        return str(row2[source2["completion_column"]])

    raise ValueError("Neither source has a completion column. At least one non-amivoice source is required.")


# =====================================================================
# Evaluation prompt
# =====================================================================

EVAL_PROMPT_TEMPLATE = """\
You are a highly experienced expert Japanese nurse.

Compare the Reference Text with two candidate texts and determine which candidate is better as a nursing record.

Task context:
Each candidate is either (a) a direct speech-to-text transcription of the same spoken content as the Reference Text, or (b) an LLM-corrected version of such transcription.
Judge only by the final text quality and fidelity to the Reference Text. Do not infer quality from the presumed source.

Evaluation policy:
- First, evaluate Candidate A independently against the Reference Text.
- Second, evaluate Candidate B independently against the Reference Text.
- Then compare them.
- If there is no meaningful quality difference, output a tie.

Priority order of criteria (higher priority first):
1. Semantic agreement with the Reference Text
2. No omission, contradiction, or modification of factual details
3. Accuracy of medical and professional terminology
4. Appropriateness as a nursing record
5. Textual coherence and readability
6. Fewer surface errors such as typos, mis-transcriptions, or unintelligible words

Reference Text (correct nursing record):
{completion}

Candidate A:
{text_a}

Candidate B:
{text_b}

Instructions:
- Prefer fidelity over fluency.
- A more fluent sentence is NOT better if it changes facts.
- Evaluate Candidate A and Candidate B separately before comparing them.
- If both candidates are essentially equally good, choose tie.
- Keep your explanation brief and focused on the key differences.
- The final line of your response must be exactly one of the following:
best: [A]
best: [B]
best: [A,B]
best: [tie]
"""


def build_eval_prompt(completion: str, text_a: str, text_b: str) -> str:
    return EVAL_PROMPT_TEMPLATE.format(
        completion=completion,
        text_a=text_a,
        text_b=text_b,
    )


def parse_judge_output(text: str) -> str:
    if not text:
        return ""

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    tail_candidates = list(reversed(lines[-5:] if len(lines) >= 5 else lines))

    patterns = [
        r"best\s*[:：]\s*\[\s*(A\s*,\s*B|B\s*,\s*A|A|B|tie)\s*\]",
        r"best\s*[:：]\s*(A\s*,\s*B|B\s*,\s*A|A|B|tie)",
    ]

    for line in tail_candidates:
        for pat in patterns:
            match = re.search(pat, line, flags=re.IGNORECASE)
            if match:
                val = match.group(1).strip().upper().replace(" ", "")
                if val in {"A", "B", "TIE"}:
                    return val.lower() if val == "TIE" else val
                if val in {"A,B", "B,A"}:
                    return "A,B"

    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            val = matches[-1].strip().upper().replace(" ", "")
            if val in {"A", "B", "TIE"}:
                return val.lower() if val == "TIE" else val
            if val in {"A,B", "B,A"}:
                return "A,B"

    return ""


def map_choice_to_source(choice: str, label_a: str, label_b: str) -> str:
    if choice == "A":
        return label_a
    if choice == "B":
        return label_b
    if choice == "A,B":
        return ",".join(sorted({label_a, label_b}))
    if choice == "tie":
        return "tie"
    return ""


def aggregate_pairwise_results(
    ab_choice: str,
    ba_choice: str,
    source1_name: str,
    source2_name: str,
) -> Tuple[str, str, str]:
    def to_source_set(choice: str, a_source: str, b_source: str):
        if choice == "A":
            return {a_source}
        if choice == "B":
            return {b_source}
        if choice == "A,B":
            return {a_source, b_source}
        if choice == "tie":
            return {"tie"}
        return set()

    ab_sources = to_source_set(ab_choice, source1_name, source2_name)
    ba_sources = to_source_set(ba_choice, source2_name, source1_name)

    if not ab_choice and not ba_choice:
        return "", "parse_error", "both_empty"
    if not ab_choice or not ba_choice:
        only = ab_sources if ab_choice else ba_sources
        if only == {source1_name}:
            return source1_name, "one_sided_parse_error", "single_valid_source1"
        if only == {source2_name}:
            return source2_name, "one_sided_parse_error", "single_valid_source2"
        if only in ({source1_name, source2_name}, {"tie"}):
            return "tie", "one_sided_parse_error", "single_valid_tie_like"
        return "", "one_sided_parse_error", "single_valid_unknown"

    if ab_sources in ({"tie"}, {source1_name, source2_name}) and ba_sources in ({"tie"}, {source1_name, source2_name}):
        return "tie", "consistent", "both_tie_like"

    if ab_sources == ba_sources:
        only = next(iter(ab_sources)) if len(ab_sources) == 1 else None
        if only == source1_name:
            return source1_name, "consistent", "both_choose_source1"
        if only == source2_name:
            return source2_name, "consistent", "both_choose_source2"
        return "tie", "consistent", "same_multivalue"

    tie_like_sets = ({"tie"}, {source1_name, source2_name})
    if ab_sources in tie_like_sets and ba_sources == {source1_name}:
        return "tie", "conflict", "ab_tie_ba_source1"
    if ab_sources in tie_like_sets and ba_sources == {source2_name}:
        return "tie", "conflict", "ab_tie_ba_source2"
    if ba_sources in tie_like_sets and ab_sources == {source1_name}:
        return "tie", "conflict", "ba_tie_ab_source1"
    if ba_sources in tie_like_sets and ab_sources == {source2_name}:
        return "tie", "conflict", "ba_tie_ab_source2"

    if ab_sources == {source1_name} and ba_sources == {source2_name}:
        return "tie", "conflict", "source1_vs_source2"
    if ab_sources == {source2_name} and ba_sources == {source1_name}:
        return "tie", "conflict", "source2_vs_source1"

    return "tie", "conflict", "fallback_tie"


def extract_response_text(response_json: dict) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    chunks: List[str] = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text", "")
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def call_openai_responses_api(
    *,
    prompt_text: str,
    api_key: str,
    model: str,
    api_base: str,
    max_output_tokens: int,
    timeout_sec: int,
    max_retries: int,
    metadata: Dict[str, str] | None = None,
) -> Tuple[str, str, float]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    }
                ],
            }
        ],
        "reasoning": {"effort": "medium"},
        "text": {"format": {"type": "text"}},
        #"temperature": 0.0,
        "max_output_tokens": max_output_tokens,
        "store": False,
    }
    if metadata:
        payload["metadata"] = metadata

    started_at = time.time()
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                api_base,
                headers=headers,
                json=payload,
                timeout=timeout_sec,
            )

            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if attempt < max_retries:
                    time.sleep(min(2 ** (attempt - 1), 30))
                    continue

            response.raise_for_status()
            body = response.json()
            raw_text = extract_response_text(body)
            elapsed = time.time() - started_at
            return raw_text, str(body.get("id", "")), elapsed

        except requests.RequestException as exc:
            last_error = str(exc)
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 30))
                continue

    raise RuntimeError(f"OpenAI API request failed after {max_retries} attempts: {last_error}")


# =====================================================================
# Main evaluation
# =====================================================================

def run_evaluation(
    *,
    source1_name: str,
    source2_name: str,
    api_key: str,
    model: str,
    api_base: str,
    max_output_tokens: int,
    checkpoint_interval: int,
    max_concurrency: int,
    timeout_sec: int,
    max_retries: int,
) -> None:
    start_time = time.time()

    if source1_name == source2_name:
        raise ValueError("--source1 and --source2 must be different.")
    if is_amivoice(source1_name) and is_amivoice(source2_name):
        raise ValueError("Two AmiVoice sources cannot be compared because the reference completion is unavailable.")
    if not api_key:
        raise ValueError("OpenAI API key is empty. Set it via --api-key or the environment variable specified by --api-key-env.")

    source1 = load_source(source1_name)
    source2 = load_source(source2_name)

    common_nos_sorted = sorted(
        set(source1["df"].index) & set(source2["df"].index),
        key=sort_key,
    )
    if MAX_SAMPLES is not None:
        common_nos_sorted = common_nos_sorted[:MAX_SAMPLES]

    print(f"[INFO] FILE1 : {source1['filename']} ({source1_name})")
    print(f"[INFO] FILE2 : {source2['filename']} ({source2_name})")
    print(f"[INFO] Number of records to evaluate: {len(common_nos_sorted)}")
    if MAX_SAMPLES is not None:
        print(f"[INFO] Trial run on the first {MAX_SAMPLES} records only")
    print(f"[INFO] Judge model: {model}")
    print(f"[INFO] API endpoint: {api_base}")

    output_suffix = f"_first{MAX_SAMPLES}" if MAX_SAMPLES is not None else ""
    output_xlsx = os.path.join(
        OUTPUT_DIR,
        f"preference_pairwise_testset_api_{source1_name}_vs_{source2_name}{output_suffix}.xlsx",
    )
    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        f"checkpoint_preference_pairwise_testset_api_{source1_name}_vs_{source2_name}{output_suffix}.csv",
    )

    checkpoint_columns = [
        "no",
        "completion",
        source1_name,
        source2_name,
        "ab_label_A",
        "ab_label_B",
        "ba_label_A",
        "ba_label_B",
        "ab_best_candidate",
        "ba_best_candidate",
        "ab_best_source",
        "ba_best_source",
        "final_best_source",
        "final_status",
        "decision_basis",
        "ab_raw_judge_output",
        "ba_raw_judge_output",
        "ab_response_id",
        "ba_response_id",
        "judge_inference_time_sec_per_prompt",
    ]

    results, processed_no_keys = load_checkpoint_data(checkpoint_path, checkpoint_columns)

    total = len(common_nos_sorted)
    pending_nos = [no for no in common_nos_sorted if canonical_no(no) not in processed_no_keys]

    processed_count = total - len(pending_nos)
    if processed_count > 0:
        print(f"[INFO] Resumed from checkpoint: {processed_count} records skipped")

    for batch_start in range(0, len(pending_nos), checkpoint_interval):
        batch_nos = pending_nos[batch_start: batch_start + checkpoint_interval]
        batch_end_global = processed_count + batch_start + len(batch_nos)

        print(f"\n[Batch] Evaluating records {processed_count + batch_start + 1}-{batch_end_global} / {total} via the API...")

        batch_jobs: List[Dict] = []
        for no in batch_nos:
            row1 = source1["df"].loc[no]
            row2 = source2["df"].loc[no]

            completion = get_reference_completion(source1, source2, no)
            source1_text = str(row1[source1["text_column"]])
            source2_text = str(row2[source2["text_column"]])

            batch_jobs.append({
                "no": no,
                "completion": completion,
                source1_name: source1_text,
                source2_name: source2_text,
                "run_type": "AB",
                "label_A": source1_name,
                "label_B": source2_name,
                "prompt_text": build_eval_prompt(
                    completion=completion,
                    text_a=source1_text,
                    text_b=source2_text,
                ),
            })
            batch_jobs.append({
                "no": no,
                "completion": completion,
                source1_name: source1_text,
                source2_name: source2_text,
                "run_type": "BA",
                "label_A": source2_name,
                "label_B": source1_name,
                "prompt_text": build_eval_prompt(
                    completion=completion,
                    text_a=source2_text,
                    text_b=source1_text,
                ),
            })

        batch_started_at = time.time()
        completed_jobs: List[Tuple[Dict, str, str, float]] = []

        with cf.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_map = {
                executor.submit(
                    call_openai_responses_api,
                    prompt_text=job["prompt_text"],
                    api_key=api_key,
                    model=model,
                    api_base=api_base,
                    max_output_tokens=max_output_tokens,
                    timeout_sec=timeout_sec,
                    max_retries=max_retries,
                    metadata={
                        "script": "evaluate_preference_testset_batch_pairwise_api.py",
                        "sample_no": str(job["no"]),
                        "run_type": job["run_type"],
                    },
                ): job
                for job in batch_jobs
            }

            for future in cf.as_completed(future_map):
                job = future_map[future]
                raw_output, response_id, elapsed = future.result()
                completed_jobs.append((job, raw_output, response_id, elapsed))

        batch_elapsed = time.time() - batch_started_at
        per_prompt_time = sum(item[3] for item in completed_jobs) / max(len(completed_jobs), 1)

        grouped: Dict[str, Dict] = {}

        for meta, raw_output, response_id, _elapsed in completed_jobs:
            best = parse_judge_output(raw_output)
            best_source = map_choice_to_source(best, meta["label_A"], meta["label_B"])

            no = meta["no"]
            if no not in grouped:
                grouped[no] = {
                    "no": no,
                    "completion": meta["completion"],
                    source1_name: meta[source1_name],
                    source2_name: meta[source2_name],
                    "ab_label_A": source1_name,
                    "ab_label_B": source2_name,
                    "ba_label_A": source2_name,
                    "ba_label_B": source1_name,
                    "ab_best_candidate": "",
                    "ba_best_candidate": "",
                    "ab_best_source": "",
                    "ba_best_source": "",
                    "ab_raw_judge_output": "",
                    "ba_raw_judge_output": "",
                    "ab_response_id": "",
                    "ba_response_id": "",
                }

            if meta["run_type"] == "AB":
                grouped[no]["ab_best_candidate"] = best
                grouped[no]["ab_best_source"] = best_source
                grouped[no]["ab_raw_judge_output"] = raw_output
                grouped[no]["ab_response_id"] = response_id
            else:
                grouped[no]["ba_best_candidate"] = best
                grouped[no]["ba_best_source"] = best_source
                grouped[no]["ba_raw_judge_output"] = raw_output
                grouped[no]["ba_response_id"] = response_id

        for no in batch_nos:
            row = grouped[no]
            final_best_source, final_status, decision_basis = aggregate_pairwise_results(
                ab_choice=row["ab_best_candidate"],
                ba_choice=row["ba_best_candidate"],
                source1_name=source1_name,
                source2_name=source2_name,
            )

            global_idx = processed_count + batch_start + batch_nos.index(no) + 1
            print(
                f"  [{global_idx}/{total}] No={no}  "
                f"AB={row['ab_best_candidate']} ({row['ab_best_source']}) / "
                f"BA={row['ba_best_candidate']} ({row['ba_best_source']}) "
                f"→ final={final_best_source} [{final_status}]"
            )

            results.append({
                **row,
                "final_best_source": final_best_source,
                "final_status": final_status,
                "decision_basis": decision_basis,
                "judge_inference_time_sec_per_prompt": round(per_prompt_time, 2),
            })
            processed_no_keys.add(canonical_no(no))

        save_checkpoint_data(results, checkpoint_path, checkpoint_columns)
        print(f"  Batch finished: {batch_elapsed:.1f}s ({per_prompt_time:.1f}s/prompt, concurrency={max_concurrency})")

    df_results = pd.DataFrame(results, columns=checkpoint_columns)
    df_results.to_excel(output_xlsx, index=False)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    total_sec = time.time() - start_time
    print(f"\n[INFO] Evaluation finished: {output_xlsx}  (elapsed: {total_sec:.1f} s)")


# =====================================================================
# Entry point
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise preference evaluation of two outputs on the synthetic test records via the OpenAI API.")
    parser.add_argument("--source1", default=DEFAULT_SOURCE1, choices=sorted(SOURCE_FILES))
    parser.add_argument("--source2", default=DEFAULT_SOURCE2, choices=sorted(SOURCE_FILES))
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or (
        args.api_key_env if args.api_key_env.startswith("sk-") else os.getenv(args.api_key_env, "")
    )
    run_evaluation(
        source1_name=args.source1,
        source2_name=args.source2,
        api_key=api_key,
        model=args.model,
        api_base=args.api_base,
        max_output_tokens=args.max_output_tokens,
        checkpoint_interval=args.checkpoint_interval,
        max_concurrency=args.max_concurrency,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
