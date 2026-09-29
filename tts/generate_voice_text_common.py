import gc
import importlib.util
import json
import os
import re
import tempfile

import numpy as np
import soundfile as sf
import torch
from scipy import signal as scipy_signal


RE_XML_TAGS = re.compile(r"<[^>]+>")
END_OUTPUT_O_PATTERN = r"</outputO>"
END_OUTPUT_S_PATTERN = r"</outputS>"
TAG_MAPPING = {
    END_OUTPUT_O_PATTERN: {"start_tag": "<outputO>", "input_tag": "<inputO>", "end_input_tag": "</inputO>"},
    END_OUTPUT_S_PATTERN: {"start_tag": "<outputS>", "input_tag": "<inputS>", "end_input_tag": "</inputS>"},
}

PROBLEMATIC_TEXT_LOG = "problematic_texts.log"
OPENJTALK_DICT_PATH = None
OPENJTALK_VOICE_PATH = None
_TTS_LINE_END_RE = re.compile(r"[。．！？!?]$")
_TTS_DUPLICATE_PUNCT_RE = re.compile(r"([、。！？])(?:[、。！？])+")


def prompt_yes_no(prompt_text, default=False):
    default_char = "y" if default else "n"
    answer = input(f"{prompt_text} (y/n, default: {default_char}): ").strip().lower()
    if not answer:
        return default
    return answer == "y"


def clear_cuda_memory(verbose_print=None):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        if verbose_print is not None:
            verbose_print("CUDA memory cleared")


def is_package_available(package_name):
    return importlib.util.find_spec(package_name) is not None


def find_openjtalk_paths():
    global OPENJTALK_DICT_PATH, OPENJTALK_VOICE_PATH

    if OPENJTALK_DICT_PATH is not None and OPENJTALK_VOICE_PATH is not None:
        return OPENJTALK_DICT_PATH, OPENJTALK_VOICE_PATH

    possible_dict_paths = [
        "/usr/local/dic",
        "/usr/share/open_jtalk/dic",
        "/var/lib/mecab/dic/open-jtalk/naist-jdic",
        "./open_jtalk_dic",
        os.path.expanduser("~/open_jtalk_dic"),
    ]
    possible_voice_paths = [
        "/usr/local/voice/mei/mei_normal.htsvoice",
        "/usr/share/open_jtalk/voice/mei/mei_normal.htsvoice",
        "/usr/share/hts-voice/nitech-jp-atr503-m001/nitech_jp_atr503_m001.htsvoice",
        "./mei_normal.htsvoice",
        os.path.expanduser("~/mei_normal.htsvoice"),
    ]

    OPENJTALK_DICT_PATH = next((path for path in possible_dict_paths if os.path.exists(path)), None)
    OPENJTALK_VOICE_PATH = next((path for path in possible_voice_paths if os.path.exists(path)), None)

    if OPENJTALK_DICT_PATH is None or OPENJTALK_VOICE_PATH is None:
        print("The Open JTalk dictionary or voice model path could not be found automatically.")
        OPENJTALK_DICT_PATH = input("Enter the path to the Open JTalk dictionary: ") or "/usr/local/dic"
        OPENJTALK_VOICE_PATH = input("Enter the path to the Open JTalk voice model: ") or "/usr/local/voice/mei/mei_normal.htsvoice"

    return OPENJTALK_DICT_PATH, OPENJTALK_VOICE_PATH


def read_jsonl_file(file_path, limit=5, start_idx=0):
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                if limit > 0 and len(data) >= limit:
                    break
                item = json.loads(line)
                item["index"] = i
                data.append(item)
        return data
    except Exception as e:
        raise RuntimeError(f"Error reading the JSONL file: {e}") from e


def log_problematic_text(idx, text, error_message=None, log_file=PROBLEMATIC_TEXT_LOG):
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Index: {idx}\n")
            if error_message:
                f.write(f"Error: {error_message}\n")
            f.write(f"Text: {text[:500]}...\n\n")
        return True
    except Exception:
        return False


def insert_tts_pause_punctuation(text, verbose_print=None):
    if text is None:
        return None

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    original = normalized
    processed_lines = []

    for raw_line in normalized.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue

        line = line.replace(" ", "、")
        if not _TTS_LINE_END_RE.search(line):
            line += "。"
        processed_lines.append(line)

    if not processed_lines:
        return ""

    transformed = "".join(processed_lines)
    transformed = _TTS_DUPLICATE_PUNCT_RE.sub(r"\1", transformed)
    if verbose_print is not None and transformed != original:
        preview_before = original[:120].replace("\n", "\\n")
        preview_after = transformed[:120]
        verbose_print(f"Punctuation added for TTS: '{preview_before}' -> '{preview_after}'")
    return transformed


def preprocess_text(text, verbose_print=None, insert_pause_punctuation=False):
    if not text:
        return None

    if insert_pause_punctuation:
        text = insert_tts_pause_punctuation(text, verbose_print=verbose_print)

    text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F-\x9F]", "", text)
    if len(text.strip()) < 3:
        if verbose_print is not None:
            verbose_print(f"Text is extremely short and will be replaced: '{text}'")
        return "こんにちは"  # dummy Japanese text ("hello") used when the input is unusable

    if re.match(r"^[･,.・。、…]+$", text.strip()):
        return "こんにちは"

    if re.match(r"^[＾~∼〜！？＋◯●★☆♪♡♥♠♣♦♤♧\^]+$", text.strip()):
        if verbose_print is not None:
            verbose_print(f"Text consists only of special characters and will be replaced: '{text}'")
        return "こんにちは"

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([!?､｡，．、。!?]+)([!?､｡，．、。!?]+)", r"\1", text)
    return text or None


def _safe_normalize(audio_data):
    max_abs = np.max(np.abs(audio_data))
    if max_abs > 0:
        return audio_data / max_abs
    return audio_data


def enhance_audio(audio_data, sr):
    audio_data = _safe_normalize(audio_data)
    sos = scipy_signal.butter(10, 100, "hp", fs=sr, output="sos")
    audio_data = scipy_signal.sosfilt(sos, audio_data)
    sos = scipy_signal.butter(10, 7000, "lp", fs=sr, output="sos")
    audio_data = scipy_signal.sosfilt(sos, audio_data)
    audio_data = audio_data * 0.9
    return _safe_normalize(audio_data)


def add_iphone_white_noise(audio_data, sr, noise_level=0.02):
    white_noise = np.random.normal(0, noise_level, len(audio_data))
    sos_hp = scipy_signal.butter(2, 80, "hp", fs=sr, output="sos")
    filtered_noise = scipy_signal.sosfilt(sos_hp, white_noise)
    sos_boost = scipy_signal.butter(2, 6000, "hp", fs=sr, output="sos")
    high_freq_boost = scipy_signal.sosfilt(sos_boost, filtered_noise) * 1.5
    iphone_noise = filtered_noise + (high_freq_boost * 0.3)
    comp_threshold = 0.3
    comp_ratio = 0.6
    mask = np.abs(iphone_noise) > comp_threshold
    iphone_noise[mask] = (
        comp_threshold + (np.abs(iphone_noise[mask]) - comp_threshold) * comp_ratio
    ) * np.sign(iphone_noise[mask])
    return _safe_normalize(audio_data + iphone_noise)


def apply_random_volume_fluctuation(audio_data, max_reduction=0.4):
    audio_length = len(audio_data)
    num_control_points = 10
    control_points = np.random.uniform(1.0 - max_reduction, 1.0, num_control_points)
    indices = np.linspace(0, audio_length - 1, num_control_points)
    envelope = np.interp(np.arange(audio_length), indices, control_points)
    return audio_data * envelope


def add_hospital_noise(audio_data, sr, noise_level=0.05, loud_mode=False):
    audio_length = len(audio_data)
    hospital_sounds = np.zeros(audio_length)
    volume_multiplier = 2.0 if loud_mode else 1.0

    def apply_envelope(chunk):
        env_len = len(chunk)
        if env_len <= 0:
            return chunk
        attack = int(env_len * 0.1)
        sustain = int(env_len * 0.8)
        release = env_len - attack - sustain
        envelope = np.concatenate(
            [
                np.linspace(0, 1, max(attack, 1), endpoint=False),
                np.ones(max(sustain, 1)),
                np.linspace(1, 0, max(release, 1)),
            ]
        )[:env_len]
        return chunk * envelope

    def create_medical_alarm(duration_sec):
        alarm_length = int(sr * duration_sec)
        t = np.arange(alarm_length) / sr
        alarm = np.zeros(alarm_length)
        for i in range(int(duration_sec / 0.4)):
            start_idx = int(i * 0.4 * sr)
            end_idx = min(start_idx + int(0.2 * sr), alarm_length)
            freq = 1800 if i % 2 == 0 else 2200
            alarm[start_idx:end_idx] = apply_envelope(np.sin(2 * np.pi * freq * t[start_idx:end_idx]))
        return alarm

    def create_nurse_call(duration_sec):
        call_length = int(sr * duration_sec)
        t = np.arange(call_length) / sr
        nurse_call = np.zeros(call_length)
        freq = 1000
        pattern = [(0.0, 0.15), (0.35, 0.15), (0.7, 0.15), (1.05, 0.4)]
        for offset_sec, beep_sec in pattern:
            start_idx = int(offset_sec * sr)
            end_idx = min(start_idx + int(beep_sec * sr), call_length)
            nurse_call[start_idx:end_idx] = apply_envelope(np.sin(2 * np.pi * freq * t[start_idx:end_idx]))
        return nurse_call

    num_alarms = max(1, int(audio_length / (sr * (10 if loud_mode else 15))))
    for _ in range(num_alarms):
        alarm_start = np.random.randint(0, max(1, audio_length - int(sr * 2)))
        alarm = create_medical_alarm(np.random.uniform(1.5, 2.5))
        end_idx = min(alarm_start + len(alarm), audio_length)
        hospital_sounds[alarm_start:end_idx] += alarm[: end_idx - alarm_start] * noise_level * 0.8 * volume_multiplier

    num_calls = max(1, int(audio_length / (sr * (15 if loud_mode else 25))))
    for _ in range(num_calls):
        call_start = np.random.randint(0, max(1, audio_length - int(sr * 2)))
        nurse_call = create_nurse_call(2.0)
        end_idx = min(call_start + len(nurse_call), audio_length)
        hospital_sounds[call_start:end_idx] += nurse_call[: end_idx - call_start] * noise_level * 0.7 * volume_multiplier

    if loud_mode and audio_length > sr * 5:
        emergency_start = np.random.randint(0, max(1, audio_length - int(sr * 4)))
        emergency_length = int(sr * np.random.uniform(3.0, 4.0))
        t = np.arange(emergency_length) / sr
        emergency = np.zeros(emergency_length)
        for i in range(int((emergency_length / sr) * 8)):
            start_idx = int(i * sr / 8)
            end_idx = min(start_idx + int(sr / 16), emergency_length)
            emergency[start_idx:end_idx] = np.sin(2 * np.pi * 2500 * t[start_idx:end_idx])
        end_idx = min(emergency_start + emergency_length, audio_length)
        hospital_sounds[emergency_start:end_idx] += emergency[: end_idx - emergency_start] * noise_level * volume_multiplier

    reverb_delay = int(sr * 0.05)
    if reverb_delay > 0 and reverb_delay < audio_length:
        reverb_sounds = np.zeros(audio_length)
        reverb_sounds[reverb_delay:] = hospital_sounds[:-reverb_delay] * 0.3
        hospital_sounds += reverb_sounds

    if loud_mode:
        audio_data = audio_data * 0.8

    return _safe_normalize(audio_data + hospital_sounds) * 0.95


def process_tags(text, verbose_print=None):
    detected_tags = {}
    for end_pattern, tag_info in TAG_MAPPING.items():
        if re.search(end_pattern, text):
            detected_tags = {
                "end_tag": end_pattern,
                "start_tag": tag_info["start_tag"],
                "input_tag": tag_info["input_tag"],
                "end_input_tag": tag_info["end_input_tag"],
            }
            break

    text_without_tags = RE_XML_TAGS.sub("", text)
    if text_without_tags and len(text_without_tags.strip()) < 2:
        if verbose_print is not None:
            verbose_print(f"Text after tag removal is extremely short and will be replaced: '{text_without_tags}'")
        text_without_tags = "こんにちは"
    return text_without_tags.strip(), detected_tags


def transcribe_speech(audio_data, sr, model, fp16_mode, clear_cuda_memory_fn=None):
    temp_wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_wav = temp_file.name
            sf.write(temp_wav, audio_data, sr, subtype="PCM_16")

        if clear_cuda_memory_fn is not None:
            clear_cuda_memory_fn()

        result = model.transcribe(temp_wav, language="ja", fp16=fp16_mode)
        return result["text"]
    except Exception as e:
        if clear_cuda_memory_fn is not None:
            clear_cuda_memory_fn()
        raise RuntimeError(f"Whisper transcription error: {e}") from e
    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)
