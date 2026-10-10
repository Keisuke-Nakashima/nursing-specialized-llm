# Supplementary Results: Preference Judgments Involving AmiVoice-Medical-Input

These results extend Section IV-B.3 of the paper. In the paper, the preference judgments by the LLM-as-a-Judge and by nurses compare the Ns-LLM trained on the Sx1, Ox1 dataset (1x model) with AmiVoice-Medical, whose average CERs are the closest among the Ns-LLM models and the AmiVoice engines. This file reports the judgments by the same LLM-as-a-Judge for the two pairs involving AmiVoice-Medical-Input, the AmiVoice engine for dictation, together with the numbers of punctuation marks in the outputs. The judgments by nurses were performed only for the pair reported in the paper.

## Table S5: Preference judgments by the LLM-as-a-Judge for each pair

The judge (GPT-5.4 with medium reasoning effort), the prompt, and the two-run procedure with swapped candidate order are the same as in the paper. W/T/L: numbers of records for which the Ns-LLM output was preferred, a tie was judged, and the AmiVoice output was preferred. The sign test, Cohen's $g$, and the 95% CI are computed as described in Appendix V of the paper. The last column gives the average CERs of the two outputs on the 100 synthetic test records (Table II of the paper).

| Ns-LLM | AmiVoice engine | W/T/L | Cohen's $g$ | 95% CI | $p$ | Average CER (Ns-LLM / AmiVoice) |
|---|---|---|---|---|---|---|
| 1x | AmiVoice-Medical | 59/11/30 | 0.16 | [0.06, 0.26] | $2.8 \times 10^{-3}$ | 7.91% / 8.35% |
| 1x | AmiVoice-Medical-Input | 60/6/34 | 0.14 | [0.03, 0.24] | $9.6 \times 10^{-3}$ | 7.91% / 5.58% |
| 8x | AmiVoice-Medical-Input | 63/11/26 | 0.21 | [0.10, 0.30] | $1.1 \times 10^{-4}$ | 6.43% / 5.58% |

The Ns-LLM output was preferred in all three pairs. For the 8x model, whose CER is comparable to that of AmiVoice-Medical-Input, the preference was stronger than in the paper. Even for the 1x model, whose CER is significantly higher than that of AmiVoice-Medical-Input (Table II of the paper), the Ns-LLM output was still preferred.

As in the paper, the ties resulted from disagreement between the two runs: opposite preferences for 5 records and a tie in one run for 1 record (1x model vs. AmiVoice-Medical-Input), and opposite preferences for 5 records and a tie in one run for 6 records (8x model vs. AmiVoice-Medical-Input).

The relation between the preference and the per-record CER is also similar to that in the paper. Of the records with the same CER for both outputs, most yielded two outputs that were identical after the normalization used for the CER computation, and for these records the Ns-LLM output was preferred in almost all cases: 20 preferred, 2 ties, and none for AmiVoice among 22 such records (1x model vs. AmiVoice-Medical-Input), and 23 preferred, 2 ties, and none for AmiVoice among 25 such records (8x model vs. AmiVoice-Medical-Input).

## Table S6: Numbers of punctuation marks in the outputs

Numbers of the Japanese full stop (。) and comma (、) in the 100 reference records and in the outputs of each method on the synthetic test records, counted before the normalization used for the CER computation. The last column gives the number of records containing at least one of the two marks.

| Output | 。 | 、 | Records with punctuation |
|---|---|---|---|
| Reference | 336 | 177 | 100 |
| Ns-LLM (1x) | 283 | 172 | 90 |
| Ns-LLM (8x) | 201 | 138 | 71 |
| AmiVoice-Medical | 12 | 68 | 48 |
| AmiVoice-Medical-Input | 11 | 14 | 22 |

The outputs of AmiVoice-Medical-Input contain even fewer punctuation marks than those of AmiVoice-Medical, whereas the Ns-LLM outputs contain punctuation marks at a rate closer to that of the reference records. This is consistent with the interpretation in the paper that formatting and stylistic factors not captured by the CER, such as punctuation, affect the preference judgments.

The counts can be reproduced from the output files in `preference/` (`infer_O_testset_gemma-9b_cer_large-v3-turbo_vllm.xlsx`, `infer_O_testset_gemma-9b_S8O8_cer_large-v3-turbo_vllm.xlsx`, `amivoice_results_medical.xlsx`, and `amivoice_results_medical-input.xlsx`) and the reference records in `testset/testset.xlsx`.
