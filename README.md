# Synthetic Test Records, Evaluation Code, and Supplementary Materials

This repository provides the supplementary materials for the paper:

> K. Nakashima, Y. Inoue, and T. Kudo, "A Nursing-Specialized LLM for ASR Error Correction in On-Premises Voice Input of Nursing Records," submitted to *IEEE Journal of Biomedical and Health Informatics*.

**Note:** The real nursing records used for training and validation are **not** included because they contain patient information. The synthetic test records in this repository were generated with an LLM and contain no real patient information.

## Repository structure

```
tts/         Reference implementation of training data construction
testset/     Synthetic test records (text and audio)
infer_zero-shot/  Zero-shot correction baselines via the OpenAI API
preference/  Preference evaluation with an LLM-as-a-Judge
supplementary/  Supplementary tables and figures referenced in the paper
```

### `tts/`

Reference implementation of the training dataset construction pipeline. Nursing-record texts are synthesized into speech with Open JTalk, mixed with noise simulating clinical environments, and transcribed with Whisper to obtain realistic recognition errors.

- `generate_voice_text.py` — main pipeline (TTS synthesis, noise addition, VAD-based silence trimming, Whisper transcription, checkpointing).
- `generate_voice_text_common.py` — shared audio processing (Butterworth filtering, noise simulating clinical environments, volume fluctuation) and text preprocessing.
- `dict/openjtalk_unit_dict.csv` — pronunciation dictionary entries for unit expressions.
- `pyproject.toml` — `uv` dependency specification of the TTS/ASR environment.
- `dataset/sample.jsonl` — sample input with six fictitious records (three S and three O records), used as the default input of the script. Each line has an empty `prompt` field and a `completion` field containing a record text followed by `</outputS>` or `</outputO>`. The script fills the `prompt` field with the Whisper transcription wrapped in the corresponding input tags.

This code is provided for reference. Running it on the actual training data requires the input nursing records (not included, as described above). Of the nursing dictionary, only the 27 unit-expression entries (`dict/openjtalk_unit_dict.csv`) are included. The entries derived from the Manbyo dictionary and the other nursing-term entries are not included.

### `testset/`

- `testset.xlsx` — 100 synthetic Japanese objective (O) records in SOAP format.
  Columns: `No` (record ID, corresponding to the audio file names), `Department` (target clinical department), `O` (record text).
  Of the 100 records, 95 were generated with GPT-5.2 Thinking in ChatGPT (5 for each of 19 clinical departments), and 5 were written by the authors (`Department`: `Human-written`). All 100 records were reviewed by nurse managers, who corrected expressions not used in clinical practice.
- `m4a/T{No}.m4a` — original recordings (iPhone SE 3rd generation, Voice Memos).
- `wav/T{No}.wav` — the same recordings converted to WAV with FFmpeg using its default settings (`ffmpeg -i T{No}.m4a T{No}.wav`), which decode the AAC audio to 16-bit PCM without resampling (48 kHz, mono).

All records were read aloud by a single male speaker (one of the authors). To better approximate actual dictation, the speaker deliberately inserted fillers and varied the readings of some terms (e.g., reading "Hb" either letter by letter or as "hemoglobin").

### `infer_zero-shot/`

Zero-shot correction baselines used in the external evaluation (Section IV-B of the paper). The Whisper transcription of each test record is corrected by GPT models via the OpenAI Responses API with a prompt that instructs verbatim correction.

- `infer_api_zero-shot.py` — inference script. The prompt is embedded in the script (`build_prompt()` and `SYSTEM_MESSAGE`). Before running, enter your OpenAI API key in `API_KEY` and set `GPT_MODEL` and `REASONING_EFFORT` at the top of the script. The paper evaluates `gpt-5.4` with `none`, `low`, and `medium`, and `gpt-5.4-mini` and `gpt-5.4-nano` with `none`. The API was accessed on September 16–17, 2026.
- `asr_test_wav_large-v3-turbo_O_testset.txt` — Whisper large-v3-turbo transcriptions of the test audio (after the VAD preprocessing described in the paper), in the prompt–completion JSON format. This file is the input to both the Ns-LLM and the zero-shot baselines.

Requirements: Python 3.10+, `openai`, `pandas`, `openpyxl`.

The results are saved to `data_infer_testset/` in the script's directory.

### `preference/`

Pairwise preference evaluation of two outputs with an LLM-as-a-Judge (OpenAI Responses API). The judge prompt, including the prioritized judgment criteria described in the paper, is embedded in the script (`EVAL_PROMPT_TEMPLATE`).

- `evaluate_preference_llm_judge.py` — evaluation script.
  Each record is judged twice with the order of Candidates A and B swapped. A preference is counted only when the judge selects the same output in both runs, and all other cases are counted as ties. Supports checkpointing and resuming.
- `checkpoint_utils.py` — checkpoint I/O helpers used by the script.
- `pyproject.toml` — `uv` dependency specification of the inference environment used to produce the outputs of the proposed method.
- `infer_O_testset_gemma-9b_cer_large-v3-turbo_vllm.xlsx` — outputs of the proposed method with the 1x model (Whisper large-v3-turbo + Ns-LLM based on Gemma-2-Llama Swallow 9B (Gemma2-S-9B) trained on the Sx1, Ox1 dataset) on the test records.
- `amivoice_results_medical.xlsx`, `amivoice_results_general.xlsx` — transcriptions of the test audio by the AmiVoice API medical and general-purpose engines (general-purpose engine accessed 21:47–21:52 JST on April 18, 2026; medical engine accessed 20:23–20:27 JST on April 19, 2026).
- `preference_pairwise_testset_api_gemma-9b_large-v3-turbo_vs_amivoice_medical.xlsx` — judgment results reported in the paper (judge: GPT-5.4, reasoning effort: medium; run on May 5, 2026).

Requirements: Python 3.10+, `pandas`, `openpyxl`, `requests`.

Before running, replace `YOUR_OPENAI_API_KEY` in `DEFAULT_API_KEY` at the top of the script with your OpenAI API key.

By default, the script compares the proposed method (`--source1 gemma-9b_large-v3-turbo`) with AmiVoice-Medical (`--source2 amivoice_medical`) using `gpt-5.4` as the judge model. The results are saved to `result_preference_testset/` in the script's directory.

### `supplementary/`

Supplementary material referenced in the paper.

- `dataset_details.md` — construction procedure, rules for unifying notational variants, and four tables: filtering results for each Whisper model (Table S1), filtering results for each subset (Table S2), the composition of each training dataset variant (Table S3), and the sizes of the merged datasets (Table S4).
- `environment.md` — software versions, TTS parameters, VAD preprocessing parameters, and vLLM decoding settings (Sections III-A and IV-A.1 of the paper).
- `evaluation_dataset_composition.md` — supplementary results on the effect of training dataset size and composition (Figures S1 to S3, with the corresponding PNG files in `fig/`), extending Sections IV-A.6 and IV-B of the paper.

## License

- The code in `preference/`, `tts/`, and `infer_zero-shot/` is licensed under the MIT License (see [LICENSE](LICENSE)).
- The synthetic test records and recorded audio in `testset/`, and the documents, tables, and figures in `supplementary/`, are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0; see [testset/LICENSE](testset/LICENSE)).

## Contact

Keisuke Nakashima (k-nakashima@hp-nurse.med.osaka-u.ac.jp)
