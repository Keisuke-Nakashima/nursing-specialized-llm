import gc
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import faulthandler
import numpy as np
import soundfile as sf
import torch
import whisper
from scipy import signal as scipy_signal
from tqdm import tqdm

from generate_voice_text_common import (
    PROBLEMATIC_TEXT_LOG,
    add_hospital_noise,
    add_iphone_white_noise,
    apply_random_volume_fluctuation,
    clear_cuda_memory as shared_clear_cuda_memory,
    enhance_audio,
    find_openjtalk_paths,
    is_package_available,
    log_problematic_text,
    preprocess_text,
    process_tags,
    prompt_yes_no,
    read_jsonl_file,
    transcribe_speech as shared_transcribe_speech,
)


if os.environ.get("VOICE_TEXT_CUDA5_WORKER") != "1":
    max_segfault_restarts = int(os.environ.get("MAX_SEGFAULT_RESTARTS", "20"))
    restart_count = 0
    script_path = os.path.abspath(__file__)
    while True:
        child_env = os.environ.copy()
        child_env["VOICE_TEXT_CUDA5_WORKER"] = "1"
        child_env["AUTO_RESUME_ON_SEGFAULT"] = "1"
        if restart_count > 0:
            child_env["SKIP_CONFIG_PROMPT"] = "1"

        result = subprocess.run([sys.executable, script_path], env=child_env)
        return_code = result.returncode
        is_segfault = return_code in (139, -signal.SIGSEGV)

        if return_code == 0:
            sys.exit(0)
        if is_segfault and (max_segfault_restarts < 0 or restart_count < max_segfault_restarts):
            restart_count += 1
            print(
                f"[auto-resume] Segmentation fault detected (exit={return_code}). "
                f"Restarting worker {restart_count}/{max_segfault_restarts} in 5s...",
                flush=True,
            )
            time.sleep(5)
            continue
        sys.exit(return_code)


faulthandler.enable()
warnings.filterwarnings("ignore", message=".*fp16.*")
warnings.filterwarnings("ignore", message=".*FP16.*")
warnings.filterwarnings("ignore", category=UserWarning)


verbose_mode = False
DEFAULT_MODEL_SIZE = "large-v3-turbo" # "large-v3-turbo" "large" "medium"
RUNTIME_CONFIG = {"model_size": DEFAULT_MODEL_SIZE}
CHECKPOINT_INTERVAL = 100
CHECKPOINT_FILE_TEMPLATE = "checkpoint_{model_size}.jsonl"
BASE_OUTPUT_FILE_TEMPLATE = "processed_data_{model_size}.jsonl"
COMPLETE_OUTPUT_FILE_TEMPLATE = "processed_data_complete_{model_size}.jsonl"
CHECKPOINT_APPEND_STATE = {}
CHECKPOINT_EXISTING_COUNTS = {}

ENABLE_SILENCE_TRIM = True
LEADING_KEEP_SEC = 0.5
TRAILING_KEEP_SEC = 0.1
INTERNAL_MIN_SILENCE_SEC = 2.0
INTERNAL_SPEECH_PAD_SEC = 0.3
MIN_AUDIO_MS_AFTER_TRIM = 300.0
VAD_SAMPLE_RATE = 16000

USER_DICT_CSV = Path(os.environ.get("OPENJTALK_USER_DICT_CSV", "openjtalk_user_dict.csv"))
USER_DICT_BIN = Path(os.environ.get("OPENJTALK_USER_DICT_BIN", "openjtalk_user_dict.dic"))
USER_DICT_DIR = Path(os.environ.get("OPENJTALK_USER_DICT_DIR", "dict"))
MERGED_USER_DICT_CSV_NAME = os.environ.get("OPENJTALK_MERGED_USER_DICT_CSV", "openjtalk_merged_user_dict.csv")
_USER_DICT_LOADED = False
DICT_PRIORITY_KEYWORDS = ("unit")


def find_existing_path(preferred: Path) -> Path:
    if preferred.is_absolute():
        return preferred

    base_dir = Path(__file__).resolve().parent
    candidates = [
        preferred,
        base_dir / preferred,
        Path.cwd() / preferred,
    ]
    for p in candidates:
        if p.exists():
            return p
    return base_dir / preferred


def collect_openjtalk_user_dict_csvs():
    dict_dir = find_existing_path(USER_DICT_DIR)
    csv_paths = []
    merged_csv_name = Path(MERGED_USER_DICT_CSV_NAME).name

    if dict_dir.exists() and dict_dir.is_dir():
        discovered_paths = [
            p for p in dict_dir.glob("*.csv")
            if p.is_file() and p.name != merged_csv_name
        ]

        def sort_key(path: Path):
            name = path.stem.lower()
            for index, keyword in enumerate(DICT_PRIORITY_KEYWORDS):
                if keyword in name:
                    return (index, path.name)
            return (len(DICT_PRIORITY_KEYWORDS), path.name)

        csv_paths.extend(sorted(discovered_paths, key=sort_key))

    if not csv_paths:
        legacy_csv = find_existing_path(USER_DICT_CSV)
        if legacy_csv.exists():
            csv_paths.append(legacy_csv)

    return csv_paths, dict_dir


def build_combined_openjtalk_user_dict(csv_paths, output_path: Path):
    merged_lines = []
    seen_surfaces = set()
    for csv_path in csv_paths:
        text = csv_path.read_text(encoding="utf-8-sig")
        for line in text.splitlines():
            stripped_line = line.strip()
            if not stripped_line:
                continue

            surface = stripped_line.split(",", 1)[0].strip()
            if surface in seen_surfaces:
                continue

            seen_surfaces.add(surface)
            merged_lines.append(stripped_line)

    merged_content = "\n".join(merged_lines)
    if merged_content:
        merged_content += "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_content = output_path.read_text(encoding="utf-8") if output_path.exists() else None
    if current_content != merged_content:
        output_path.write_text(merged_content, encoding="utf-8")

    return output_path


def ensure_openjtalk_user_dict():
    """Load the user dictionary for pyopenjtalk / pyopenjtalk-plus only once."""
    global _USER_DICT_LOADED

    if _USER_DICT_LOADED:
        return

    if not is_package_available("pyopenjtalk"):
        verbose_print("No pyopenjtalk package found, so the user dictionary will not be loaded.")
        return

    csv_paths, dict_dir = collect_openjtalk_user_dict_csvs()
    if not csv_paths:
        verbose_print(
            f"Open JTalk user dictionary CSV not found, skipping: "
            f"{find_existing_path(USER_DICT_DIR)}/*.csv or {USER_DICT_CSV}"
        )
        return

    merged_csv_path = build_combined_openjtalk_user_dict(
        csv_paths,
        dict_dir / MERGED_USER_DICT_CSV_NAME if dict_dir.exists() and dict_dir.is_dir() else find_existing_path(Path(MERGED_USER_DICT_CSV_NAME)),
    )
    dict_path = find_existing_path(USER_DICT_BIN)
    if dict_path.name == USER_DICT_BIN.name and dict_dir.exists() and dict_dir.is_dir():
        dict_path = dict_dir / USER_DICT_BIN.name

    try:
        import pyopenjtalk

        if (not dict_path.exists()) or (merged_csv_path.stat().st_mtime > dict_path.stat().st_mtime):
            verbose_print(f"Building the Open JTalk user dictionary: {merged_csv_path} -> {dict_path}")
            verbose_print(f"Dictionary sources: {', '.join(str(path) for path in csv_paths)}")
            pyopenjtalk.mecab_dict_index(str(merged_csv_path), str(dict_path))

        pyopenjtalk.update_global_jtalk_with_user_dict(str(dict_path))
        _USER_DICT_LOADED = True
        verbose_print(f"Open JTalk user dictionary loaded: {dict_path}")
    except Exception as e:
        print(f"Failed to load the Open JTalk user dictionary: {e}")


_vad_model, _vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
_get_speech_timestamps = _vad_utils[0]


def verbose_print(*args, **kwargs):
    if verbose_mode:
        print(*args, **kwargs)


def checkpoint_file_for_model(model_size="large"):
    return CHECKPOINT_FILE_TEMPLATE.format(model_size=model_size)


def output_file_for_model(model_size="large"):
    return BASE_OUTPUT_FILE_TEMPLATE.format(model_size=model_size)


def complete_output_file_for_model(model_size="large"):
    return COMPLETE_OUTPUT_FILE_TEMPLATE.format(model_size=model_size)


def infer_model_size_from_checkpoint_name(filename, default="large"):
    basename = os.path.basename(filename)
    matched = re.match(r"^checkpoint_(?:\d+_)?(.+)\.jsonl$", basename)
    if matched:
        return matched.group(1)
    return default


def count_jsonl_lines(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def find_latest_checkpoint_file(model_size="large", checkpoint_dir="."):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file_for_model(model_size))
    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        return checkpoint_path
    return None


def save_checkpoint(data, filename):
    model_size = RUNTIME_CONFIG["model_size"] or infer_model_size_from_checkpoint_name(filename)
    checkpoint_path = checkpoint_file_for_model(model_size)
    session_saved_count = CHECKPOINT_APPEND_STATE.get(checkpoint_path, 0)
    existing_count = CHECKPOINT_EXISTING_COUNTS.get(checkpoint_path)
    if existing_count is None:
        existing_count = count_jsonl_lines(checkpoint_path)
        CHECKPOINT_EXISTING_COUNTS[checkpoint_path] = existing_count

    new_items = data[session_saved_count:]
    if not new_items:
        return

    try:
        with open(checkpoint_path, "a", encoding="utf-8") as f:
            for item in new_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        CHECKPOINT_APPEND_STATE[checkpoint_path] = session_saved_count + len(new_items)
        verbose_print(
            f"Checkpoint appended: {checkpoint_path} "
            f"(added {len(new_items)} / total {existing_count + CHECKPOINT_APPEND_STATE[checkpoint_path]})"
        )
    except Exception as e:
        print(f"Checkpoint save error: {e}")


def resume_from_checkpoint(checkpoint_file, jsonl_file, limit=0, interactive=True):
    processed_data = []

    try:
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "index" not in item:
                    item["index"] = item.get("_line_idx", len(processed_data))
                item.pop("_line_idx", None)
                processed_data.append(item)

        print(f"Loaded {len(processed_data)} records from checkpoint {checkpoint_file}")

        start_idx = max((item["index"] for item in processed_data), default=-1) + 1
        print(f"Estimated start index: {start_idx}")
        if interactive:
            user_idx = input(f"Enter a different start index to resume from (default: {start_idx}): ").strip()
            if user_idx:
                try:
                    start_idx = int(user_idx)
                    print(f"Start index set to {start_idx}")
                except ValueError:
                    print(f"Not a valid number, using the default {start_idx}")
        else:
            print(f"[auto-resume] Non-interactive mode, continuing from start index {start_idx}.")

        remaining_data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx < start_idx:
                    continue
                if limit > 0 and len(remaining_data) >= limit:
                    break
                item = json.loads(line)
                item["index"] = idx
                remaining_data.append(item)

        CHECKPOINT_APPEND_STATE[checkpoint_file] = 0
        CHECKPOINT_EXISTING_COUNTS[checkpoint_file] = len(processed_data)
        print(f"Remaining records to process: {len(remaining_data)}")
        return processed_data, remaining_data, start_idx
    except Exception as e:
        print(f"Error while restoring from the checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return [], [], 0


def clear_cuda_memory():
    shared_clear_cuda_memory(verbose_print)


def with_index(item, index_value):
    updated = item.copy()
    updated["index"] = index_value
    updated.pop("_line_idx", None)
    return updated


def generate_speech(
    text,
    output_file=None,
    speech_rate=1.0,
    add_noise=True,
    noise_level=0.05,
    loud_mode=False,
    add_iphone_noise=False,
    iphone_noise_level=0.02,
    apply_volume_fluctuation=False,
    max_volume_reduction=0.4,
    save_audio=True,
):
    processed_text = preprocess_text(text, verbose_print=verbose_print)
    if processed_text is None or len(processed_text) < 3:
        verbose_print(f"Text preprocessing replaced the text with dummy text: '{text}'")
        processed_text = "こんにちは"

    try:
        if is_package_available("pyopenjtalk"):
            try:
                import pyopenjtalk
                ensure_openjtalk_user_dict()

                try:
                    x, sr = pyopenjtalk.tts(processed_text, speed=speech_rate)
                except (AttributeError, TypeError):
                    x = pyopenjtalk.synthesize(processed_text, speed=speech_rate)
                    sr = 48000

                x = enhance_audio(x, sr)
                if apply_volume_fluctuation:
                    x = apply_random_volume_fluctuation(x, max_volume_reduction)
                if add_iphone_noise:
                    x = add_iphone_white_noise(x, sr, iphone_noise_level)
                if add_noise:
                    x = add_hospital_noise(x, sr, noise_level, loud_mode)
                if save_audio and output_file:
                    sf.write(output_file, x, sr, subtype="PCM_24")
                return x, sr
            except Exception as e:
                print(f"PyOpenJTalk / PyOpenJTalk-Plus error: {e}")

        verbose_print("Using the command-line Open JTalk...")
        dict_path, voice_path = find_openjtalk_paths()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as f:
            f.write(processed_text)
            temp_txt = f.name

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_wav = temp_file.name

            cmd = [
                "open_jtalk",
                "-x", dict_path,
                "-m", voice_path,
                "-s", "48000",
                "-p", "240",
                "-a", "0.55",
                "-b", "0.0",
                "-r", str(speech_rate),
                "-fm", "0.0",
                "-u", "0.5",
                "-jm", "1.0",
                "-jf", "1.0",
                "-z", "1024",
                "-ow", temp_wav,
                temp_txt,
            ]
            result = subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            stderr_output = result.stderr.decode("utf-8", errors="ignore")
            if "No phoneme" in stderr_output or "not appropriate POS" in stderr_output:
                dummy_audio = np.zeros(16000)
                if save_audio and output_file:
                    sf.write(output_file, dummy_audio, 16000, subtype="PCM_16")
                return dummy_audio, 16000

            data, sr = sf.read(temp_wav)
            enhanced_data = enhance_audio(data, sr)
            if apply_volume_fluctuation:
                enhanced_data = apply_random_volume_fluctuation(enhanced_data, max_volume_reduction)
            if add_iphone_noise:
                enhanced_data = add_iphone_white_noise(enhanced_data, sr, iphone_noise_level)
            if add_noise:
                enhanced_data = add_hospital_noise(enhanced_data, sr, noise_level, loud_mode)
            if save_audio and output_file:
                sf.write(output_file, enhanced_data, sr, subtype="PCM_24")
            if os.path.exists(temp_wav):
                os.unlink(temp_wav)
            return enhanced_data, sr
        except Exception as e:
            verbose_print(f"Speech synthesis with Open JTalk failed: {e}")
            dummy_audio = np.zeros(16000)
            if save_audio and output_file:
                sf.write(output_file, dummy_audio, 16000, subtype="PCM_16")
            return dummy_audio, 16000
        finally:
            if os.path.exists(temp_txt):
                os.unlink(temp_txt)
    except Exception as e:
        verbose_print(f"Speech synthesis error: {e}")
        dummy_audio = np.zeros(16000)
        if save_audio and output_file:
            sf.write(output_file, dummy_audio, 16000, subtype="PCM_16")
        return dummy_audio, 16000


def to_mono_float32(audio_data):
    audio = np.asarray(audio_data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def resample_audio(audio_data, src_sr, dst_sr):
    if src_sr == dst_sr:
        return np.asarray(audio_data, dtype=np.float32)
    gcd = math.gcd(int(src_sr), int(dst_sr))
    up = dst_sr // gcd
    down = src_sr // gcd
    return scipy_signal.resample_poly(audio_data, up, down).astype(np.float32)


def trim_and_remove_silence(audio, sr):
    audio = to_mono_float32(audio)
    total_samples = len(audio)
    if not ENABLE_SILENCE_TRIM or total_samples == 0:
        return audio

    vad_audio = resample_audio(audio, sr, VAD_SAMPLE_RATE)
    if len(vad_audio) == 0:
        return audio

    speech_ts = _get_speech_timestamps(
        torch.from_numpy(vad_audio).float(),
        _vad_model,
        sampling_rate=VAD_SAMPLE_RATE,
    )
    if not speech_ts:
        return audio

    total_vad_samples = len(vad_audio)
    first_speech_sec = speech_ts[0]["start"] / VAD_SAMPLE_RATE
    last_speech_sec = speech_ts[-1]["end"] / VAD_SAMPLE_RATE
    leading_trim_sec = max(0.0, first_speech_sec - LEADING_KEEP_SEC)
    trailing_trim_sec = max(0.0, (total_vad_samples / VAD_SAMPLE_RATE) - last_speech_sec - TRAILING_KEEP_SEC)
    start_vad_sample = int(leading_trim_sec * VAD_SAMPLE_RATE)
    end_vad_sample = total_vad_samples - int(trailing_trim_sec * VAD_SAMPLE_RATE)

    if end_vad_sample - start_vad_sample < int(MIN_AUDIO_MS_AFTER_TRIM / 1000.0 * VAD_SAMPLE_RATE):
        return audio

    pad_samples = int(INTERNAL_SPEECH_PAD_SEC * VAD_SAMPLE_RATE)
    min_silence_samples = int(INTERNAL_MIN_SILENCE_SEC * VAD_SAMPLE_RATE)
    segments = []
    for ts in speech_ts:
        seg_start = max(ts["start"], start_vad_sample)
        seg_end = min(ts["end"], end_vad_sample)
        if seg_start < seg_end:
            segments.append((seg_start, seg_end))
    if not segments:
        return audio

    extract_ranges = []
    for i, (seg_start, seg_end) in enumerate(segments):
        padded_start = max(start_vad_sample, seg_start - pad_samples)
        padded_end = min(end_vad_sample, seg_end + pad_samples)
        if i == 0:
            extract_ranges.append((start_vad_sample, padded_end))
            continue
        gap = seg_start - segments[i - 1][1]
        if gap >= min_silence_samples:
            extract_ranges.append((padded_start, padded_end))
        else:
            last_start, _ = extract_ranges[-1]
            extract_ranges[-1] = (last_start, padded_end)

    last_start, _ = extract_ranges[-1]
    extract_ranges[-1] = (last_start, end_vad_sample)

    trimmed_chunks = []
    ratio = sr / VAD_SAMPLE_RATE
    for vad_start, vad_end in extract_ranges:
        orig_start = max(0, min(total_samples, int(round(vad_start * ratio))))
        orig_end = max(orig_start, min(total_samples, int(round(vad_end * ratio))))
        if orig_end > orig_start:
            trimmed_chunks.append(audio[orig_start:orig_end])

    if not trimmed_chunks:
        return audio

    trimmed_audio = np.concatenate(trimmed_chunks).astype(np.float32, copy=False)
    if len(trimmed_audio) < int(MIN_AUDIO_MS_AFTER_TRIM / 1000.0 * sr):
        return audio
    verbose_print(f"Silence trimming applied: {len(audio) / sr:.2f} s -> {len(trimmed_audio) / sr:.2f} s")
    return trimmed_audio


def generate_speech_with_trim(
    text,
    output_file=None,
    speech_rate=1.0,
    add_noise=True,
    noise_level=0.05,
    loud_mode=False,
    add_iphone_noise=False,
    iphone_noise_level=0.02,
    apply_volume_fluctuation=False,
    max_volume_reduction=0.4,
    save_audio=True,
):
    audio_data, sr = generate_speech(
        text,
        output_file=None,
        speech_rate=speech_rate,
        add_noise=add_noise,
        noise_level=noise_level,
        loud_mode=loud_mode,
        add_iphone_noise=add_iphone_noise,
        iphone_noise_level=iphone_noise_level,
        apply_volume_fluctuation=apply_volume_fluctuation,
        max_volume_reduction=max_volume_reduction,
        save_audio=False,
    )
    if audio_data is None or sr is None:
        return audio_data, sr
    trimmed_audio = trim_and_remove_silence(audio_data, sr)
    if save_audio and output_file:
        subtype = "PCM_24" if sr > 16000 else "PCM_16"
        sf.write(output_file, trimmed_audio, sr, subtype=subtype)
    return trimmed_audio, sr


def transcribe_speech(audio_data, sr, model, fp16_mode):
    try:
        return shared_transcribe_speech(
            audio_data,
            sr,
            model,
            fp16_mode,
            clear_cuda_memory_fn=clear_cuda_memory,
        )
    except RuntimeError as e:
        print(str(e))
        return ""


def process_jsonl(
    jsonl_file,
    limit=5,
    speech_rate=1.0,
    add_noise=True,
    noise_level=0.05,
    loud_mode=False,
    add_iphone_noise=False,
    iphone_noise_level=0.02,
    apply_volume_fluctuation=False,
    max_volume_reduction=0.4,
    save_audio=True,
    verbose=True,
    start_from_idx=0,
    data_to_process=None,
):
    global verbose_mode
    verbose_mode = verbose

    if data_to_process is not None:
        data = data_to_process
        verbose_print(f"Using the provided data: {len(data)} records")
    else:
        try:
            data = read_jsonl_file(jsonl_file, limit, start_idx=start_from_idx)
        except RuntimeError as e:
            print(str(e))
            sys.exit(1)
        verbose_print(f"Loaded {len(data)} records from the JSONL file")

    audio_dir = "generated_audio"
    if save_audio and not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
        verbose_print(f"Created the directory for audio files: {audio_dir}")

    processed_data = []
    successful_count = 0
    error_count = 0

    with open(PROBLEMATIC_TEXT_LOG, "a", encoding="utf-8") as f:
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Start index: {start_from_idx}\n\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16_mode = torch.cuda.is_available()

    try:
        whisper_model = whisper.load_model(RUNTIME_CONFIG["model_size"], device=device)
        verbose_print(
            f"Whisper model loaded "
            f"(model: {RUNTIME_CONFIG['model_size']}, device: {device}, fp16: {fp16_mode})"
        )
        verbose_print(f"torch.cuda.is_available() -> {torch.cuda.is_available()}")
    except Exception as e:
        print(f"Failed to load the Whisper model: {e}")
        print("Retrying on CPU with fp16_mode set to False.")
        fp16_mode = False
        try:
            whisper_model = whisper.load_model(RUNTIME_CONFIG["model_size"], device="cpu")
        except Exception as e2:
            print(f"Retrying to load the Whisper model also failed: {e2}")
            sys.exit(1)

    progress_total = max(start_from_idx + len(data), start_from_idx)
    for idx, item in tqdm(enumerate(data), total=progress_total, initial=start_from_idx, desc="Synthesizing and transcribing"):
        try:
            actual_idx = item["index"] if "index" in item else item.get("_line_idx", start_from_idx + idx)

            if idx > 0 and idx % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(processed_data, checkpoint_file_for_model(RUNTIME_CONFIG["model_size"]))

            completion = item.get("completion", "")
            if not completion:
                verbose_print(f"Skipping record {actual_idx}: the 'completion' field is empty")
                processed_data.append(with_index(item, actual_idx))
                continue

            cleaned_completion, tag_info = process_tags(completion, verbose_print=verbose_print)
            audio_file = os.path.join(audio_dir, f"item_{actual_idx}.wav") if save_audio else None
            audio_data, sr = generate_speech_with_trim(
                cleaned_completion,
                audio_file,
                speech_rate=speech_rate,
                add_noise=add_noise,
                noise_level=noise_level,
                loud_mode=loud_mode,
                add_iphone_noise=add_iphone_noise,
                iphone_noise_level=iphone_noise_level,
                apply_volume_fluctuation=apply_volume_fluctuation,
                max_volume_reduction=max_volume_reduction,
                save_audio=save_audio,
            )
            if audio_data is None or sr is None:
                verbose_print(f"Skipping record {actual_idx}: excluded by text preprocessing")
                processed_data.append(with_index(item, actual_idx))
                continue

            transcription = transcribe_speech(audio_data, sr, whisper_model, fp16_mode)
            processed_item = with_index(item, actual_idx)
            if tag_info and "input_tag" in tag_info:
                processed_item["prompt"] = (
                    f"{tag_info['input_tag']}{transcription}"
                    f"{tag_info['end_input_tag']}{tag_info['start_tag']}"
                )
            else:
                processed_item["prompt"] = transcription

            processed_data.append(processed_item)
            successful_count += 1
            clear_cuda_memory()
        except Exception as e:
            error_message = f"Error while processing record {actual_idx}: {e}"
            verbose_print(error_message)
            log_problematic_text(actual_idx, completion if "completion" in locals() else "", error_message)
            processed_data.append(with_index(item, actual_idx))
            error_count += 1
            clear_cuda_memory()

    print(f"\nProcessing finished. Succeeded: {successful_count}, errors: {error_count}")
    print(f"Number of processed records: {len(processed_data)}")

    if not processed_data:
        print("Warning: no processed data. Nothing will be saved.")
        return processed_data

    output_file = output_file_for_model(RUNTIME_CONFIG["model_size"])
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            saved_count = 0
            for item in processed_data:
                try:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    saved_count += 1
                except Exception as e:
                    print(f"Error converting an item to JSON: {e}")
        print(f"Results saved: {output_file} ({saved_count} records)")
    except Exception as e:
        print(f"Error while saving the file: {e}")

    return processed_data


if __name__ == "__main__":
    auto_resume_mode = os.environ.get("AUTO_RESUME_ON_SEGFAULT") == "1"
    skip_config_prompt = os.environ.get("SKIP_CONFIG_PROMPT") == "1"
    non_interactive_mode = os.environ.get("NON_INTERACTIVE_MODE") == "1" or skip_config_prompt

    default_model_size = DEFAULT_MODEL_SIZE
    latest_checkpoint = find_latest_checkpoint_file(model_size=default_model_size)
    config = {
        "model_size": default_model_size,
        "verbose_mode": False,
        "speech_rate": 1.2,
        "add_noise": True,
        "loud_mode": True,
        "noise_level": 0.04,
        "add_iphone_noise": True,
        "iphone_noise_level": 0.02,
        "apply_volume_fluctuation": True,
        "max_volume_reduction": 0.7,
        "save_audio": False,
        "jsonl_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "sample.jsonl"), # change this to your actual JSONL file path
        "limit": 0,
        "resume_from_checkpoint": latest_checkpoint is not None,
        "checkpoint_file": latest_checkpoint or checkpoint_file_for_model(default_model_size),
    }

    if auto_resume_mode and latest_checkpoint:
        print(f"[auto-resume] Latest checkpoint detected: {latest_checkpoint}")
    elif auto_resume_mode:
        print("[auto-resume] No checkpoint available, starting a new run.")

    print("Starting JSONL processing and speech synthesis...")
    print("\n=== Current settings ===")
    for key, value in config.items():
        print(f"- {key}: {value}")

    if skip_config_prompt:
        print("\n[auto-resume] Restarted, skipping the settings prompt.")
        change_config = False
    else:
        change_config = prompt_yes_no("\nChange the settings?", default=False)

    if change_config:
        model_size_input = input(f"Enter the Whisper model size (default: {config['model_size']}): ").strip()
        if model_size_input:
            config["model_size"] = model_size_input
        config["verbose_mode"] = prompt_yes_no("Show detailed processing information?", default=config["verbose_mode"])

        speech_rate_valid = False
        while not speech_rate_valid:
            try:
                user_rate = input(f"Enter the speaking rate (0.5-2.5, default: {config['speech_rate']}): ").strip()
                if not user_rate:
                    speech_rate_valid = True
                else:
                    rate = float(user_rate)
                    if 0.5 <= rate <= 2.5:
                        config["speech_rate"] = rate
                        speech_rate_valid = True
                    else:
                        print("Enter a speaking rate between 0.5 and 2.5.")
            except ValueError:
                print("Enter a valid number.")

        jsonl_file = input(f"Enter the path to the JSONL file to process (default: {config['jsonl_file']}): ").strip()
        if jsonl_file:
            config["jsonl_file"] = jsonl_file

        try:
            limit_input = input(f"Enter the number of records to process (0 = all, default: {config['limit']}): ").strip()
            if limit_input:
                config["limit"] = int(limit_input)
        except ValueError:
            print(f"Invalid input. Using the default {config['limit']}")

        config["resume_from_checkpoint"] = prompt_yes_no(
            "Resume from a checkpoint?",
            default=config["resume_from_checkpoint"],
        )
        if config["resume_from_checkpoint"]:
            checkpoint_input = input(
                f"Enter the checkpoint file to resume from (default: {config['checkpoint_file']}): "
            ).strip()
            if checkpoint_input:
                config["checkpoint_file"] = checkpoint_input

    previous_checkpoint_file = config["checkpoint_file"]
    RUNTIME_CONFIG["model_size"] = config["model_size"]
    latest_checkpoint = find_latest_checkpoint_file(model_size=RUNTIME_CONFIG["model_size"])
    default_checkpoint_file = checkpoint_file_for_model(RUNTIME_CONFIG["model_size"])
    if previous_checkpoint_file in {None, checkpoint_file_for_model(default_model_size)}:
        config["checkpoint_file"] = latest_checkpoint or default_checkpoint_file

    verbose_mode = config["verbose_mode"]

    if not os.path.exists(config["jsonl_file"]):
        print(f"Error: JSONL file '{config['jsonl_file']}' not found.")
        alternative_file = input("Enter the path to the JSONL file: ")
        if alternative_file and os.path.exists(alternative_file):
            config["jsonl_file"] = alternative_file
        else:
            print("No valid JSONL file provided. Exiting.")
            sys.exit(1)

    if config["resume_from_checkpoint"]:
        checkpoint_path = config.get("checkpoint_file")
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"Error: checkpoint file '{config['checkpoint_file']}' not found.")
            latest_checkpoint = find_latest_checkpoint_file(model_size=RUNTIME_CONFIG["model_size"])
            if latest_checkpoint:
                config["checkpoint_file"] = latest_checkpoint
                print(f"Using the latest checkpoint instead: {latest_checkpoint}")
            elif non_interactive_mode:
                print("Non-interactive mode, starting in normal mode.")
                config["resume_from_checkpoint"] = False
            else:
                alternative_checkpoint = input("Enter the path to the checkpoint file (leave empty to skip): ").strip()
                if alternative_checkpoint and os.path.exists(alternative_checkpoint):
                    config["checkpoint_file"] = alternative_checkpoint
                else:
                    print("No valid checkpoint file provided. Starting in normal mode.")
                    config["resume_from_checkpoint"] = False

        if config["resume_from_checkpoint"]:
            processed_data, remaining_data, start_idx = resume_from_checkpoint(
                config["checkpoint_file"],
                config["jsonl_file"],
                config["limit"],
                interactive=not non_interactive_mode,
            )
            if processed_data and remaining_data:
                print(f"Resuming from the checkpoint. Start index: {start_idx}")
                try:
                    start_time = time.time()
                    newly_processed = process_jsonl(
                        config["jsonl_file"],
                        limit=config["limit"],
                        speech_rate=config["speech_rate"],
                        add_noise=config["add_noise"],
                        noise_level=config["noise_level"],
                        loud_mode=config["loud_mode"],
                        add_iphone_noise=config["add_iphone_noise"],
                        iphone_noise_level=config["iphone_noise_level"],
                        apply_volume_fluctuation=config["apply_volume_fluctuation"],
                        max_volume_reduction=config["max_volume_reduction"],
                        save_audio=config["save_audio"],
                        verbose=config["verbose_mode"],
                        start_from_idx=start_idx,
                        data_to_process=remaining_data,
                    )
                    save_checkpoint(newly_processed, checkpoint_file_for_model(RUNTIME_CONFIG["model_size"]))

                    all_processed = processed_data + newly_processed
                    output_file = complete_output_file_for_model(RUNTIME_CONFIG["model_size"])
                    with open(output_file, "w", encoding="utf-8") as f:
                        for item in all_processed:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")

                    print(f"All data processed. Total: {len(all_processed)} records")
                    elapsed_time = time.time() - start_time
                    total_minutes = elapsed_time / 60
                    total_hours = total_minutes / 60
                    if total_hours >= 1:
                        print(f"Resumed processing time: {total_hours:.2f} h ({elapsed_time:.2f} s)")
                    elif total_minutes >= 1:
                        print(f"Resumed processing time: {total_minutes:.2f} min ({elapsed_time:.2f} s)")
                    else:
                        print(f"Resumed processing time: {elapsed_time:.2f} s")
                except Exception as e:
                    print(f"Error while resuming: {e}")
                    import traceback
                    traceback.print_exc()
                sys.exit(0)
            else:
                print("Failed to restore data from the checkpoint. Starting in normal mode.")

    try:
        start_time = time.time()
        processed_data = process_jsonl(
            config["jsonl_file"],
            limit=config["limit"],
            speech_rate=config["speech_rate"],
            add_noise=config["add_noise"],
            noise_level=config["noise_level"],
            loud_mode=config["loud_mode"],
            add_iphone_noise=config["add_iphone_noise"],
            iphone_noise_level=config["iphone_noise_level"],
            apply_volume_fluctuation=config["apply_volume_fluctuation"],
            max_volume_reduction=config["max_volume_reduction"],
            save_audio=config["save_audio"],
            verbose=config["verbose_mode"],
        )
        save_checkpoint(processed_data, checkpoint_file_for_model(RUNTIME_CONFIG["model_size"]))

        print("\nJSONL processing and speech synthesis completed successfully.")
        print("\nApplied processing options:")
        for key, value in config.items():
            if key not in {"jsonl_file", "limit"}:
                print(f"- {key}: {value}")

        elapsed_time = time.time() - start_time
        total_minutes = elapsed_time / 60
        total_hours = total_minutes / 60
        if total_hours >= 1:
            print(f"Total processing time: {total_hours:.2f} h ({elapsed_time:.2f} s)")
        elif total_minutes >= 1:
            print(f"Total processing time: {total_minutes:.2f} min ({elapsed_time:.2f} s)")
        else:
            print(f"Total processing time: {elapsed_time:.2f} s")

        processed_count = len(processed_data)
        if processed_count > 0:
            avg_time = elapsed_time / processed_count
            print(f"Average processing time: {avg_time:.2f} s/item")
            try:
                with open(config["jsonl_file"], "r", encoding="utf-8") as f:
                    total_items = sum(1 for _ in f)
                if config["limit"] < total_items and config["limit"] > 0:
                    remaining_items = total_items - processed_count
                    if remaining_items > 0:
                        estimated_time = avg_time * remaining_items
                        estimated_hours = estimated_time / 3600
                        estimated_minutes = estimated_time / 60
                        print(f"\nEstimated processing time for the remaining {remaining_items} items:")
                        if estimated_hours >= 1:
                            print(f"about {estimated_hours:.2f} h")
                        elif estimated_minutes >= 1:
                            print(f"about {estimated_minutes:.2f} min")
                        else:
                            print(f"about {estimated_time:.2f} s")
            except Exception as e:
                verbose_print(f"Error while estimating the remaining time: {e}")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
