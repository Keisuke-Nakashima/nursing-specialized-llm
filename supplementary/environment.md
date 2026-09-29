# Supplementary: Software Environment and Processing Parameters

This file gives the software versions and the parameters summarized in Sections III-A (Training Dataset Construction) and IV-A.1 (Environment) of the paper. The fine-tuning environment is documented below only, because the fine-tuning code is not included in this repository.

## Hardware

- GPU: NVIDIA GeForce RTX 5090 (32 GB VRAM), CUDA 12.8

## Software versions

### ASR / TTS environment (`tts/pyproject.toml`)

| Package | Version |
|---|---|
| Python | 3.12 |
| openai-whisper | 20250625 |
| pyopenjtalk-plus | 0.4.1.post8 (bundles Open JTalk 1.11) |
| PyTorch | 2.8.0 |
| TorchVision | 0.23.0 |
| TorchAudio | 2.8.0 |
| Silero VAD | loaded via `torch.hub.load("snakers4/silero-vad", "silero_vad")` |

### Fine-tuning environment

| Package | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.9.1+cu128 |
| Transformers | 4.57.3 |
| TRL | 0.24.0 |
| Unsloth | 2025.12.10 |
| Unsloth-Zoo | 2025.12.8 |
| BitsAndBytes | 0.49.0 |
| PEFT | 0.18.0 |
| Accelerate | 1.12.0 |
| Datasets | 4.3.0 |

### Inference environment (`preference/pyproject.toml`)

| Package | Version |
|---|---|
| Python | 3.12 |
| vLLM | 0.14.1 |
| PyTorch | 2.9.1+cu128 |
| Transformers | 4.57.3 |
| BitsAndBytes | 0.49.0 |
| PEFT | 0.18.0 |

### CER computation

| Package | Version |
|---|---|
| JiWER | 4.0.0 |

## Synthetic-speech generation (TTS pipeline)

Parameters of the five-step procedure in Section III-A of the paper.

1. Speech is synthesized with Open JTalk at a speaking rate of 1.2.
2. Two 10th-order Butterworth filters (SciPy) are applied: a high-pass filter with a 100 Hz cutoff and a low-pass filter with a 7,000 Hz cutoff.
3. Random amplitude modulation simulates transient decreases in volume, with attenuation of up to 70%.
4. Zero-mean Gaussian noise with a standard deviation of 0.02 relative to the normalized waveform amplitude is spectrally shaped and amplitude-compressed, then added to approximate smartphone-recording noise.
5. Simulated nurse-call and clinical-alarm tones are added at random positions. The clinical alarm alternates 1,800 Hz and 2,200 Hz sine waves at 0.4 s intervals (each tone 0.2 s). The nurse call consists of three short 1,000 Hz tones of 0.15 s followed by one long tone of 0.4 s. The resulting signal is peak-normalized to a maximum amplitude of 0.95.

See `tts/generate_voice_text.py` and `tts/generate_voice_text_common.py` for the implementation.

## Whisper transcription

- Language: Japanese (`language="ja"`), fp16 on CUDA.

## VAD-based preprocessing (Silero VAD)

Applied to both the synthesized speech (training data) and the recorded audio before Whisper transcription. Splitting of long audio applies to recorded audio only.

- Leading and trailing silence trimmed, retaining 0.5 s before the first detected speech segment and 0.1 s after the last segment.
- Internal silent intervals of at least 2.0 s removed, with 0.3 s speech padding.
- Audio longer than 50.0 s split at VAD-based speech boundaries into chunks of up to 25.0 s. Chunks shorter than 5.0 s merged into the preceding chunk.
- For practical deployments, a similar pipeline (VAD-based preprocessing followed by Whisper transcription) can be realized more simply with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), which provides built-in Silero VAD filtering. The evaluation in the paper used openai-whisper with the separate Silero VAD preprocessing described above.

## vLLM inference settings

- LoRA adapters merged into the base model on CPU in fp32. Merged model loaded with BitsAndBytes 4-bit quantization.
- dtype: bfloat16.
- `max_model_len`: 2048, `max_num_seqs`: 1, `gpu_memory_utilization`: 0.7.
- Decoding: greedy (`temperature=0.0`, `top_p=1.0`), `max_tokens`: 1024, stop strings: `</outputO>` and `</outputS>`.
