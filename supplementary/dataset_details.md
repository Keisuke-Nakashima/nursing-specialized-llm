# Supplementary: Training Dataset Details

This file gives the details of the training datasets used in the internal evaluation of the paper: the filtering results for each Whisper model (Section IV-A.2, Table S1) and the datasets used in the dictionary ablation (Section IV-A.5) and the dataset-size experiments (Section IV-A.6, Tables S2–S4).

## Construction

The S and O records were partitioned into mutually disjoint subsets $`\mathcal{S}_{1}, \ldots, \mathcal{S}_{8}`$ and $`\mathcal{O}_{1}, \ldots, \mathcal{O}_{16}`$ as described in Section IV-A.2 of the paper. $`\mathcal{O}_{9}`$–$`\mathcal{O}_{12}`$ and $`\mathcal{O}_{13}`$–$`\mathcal{O}_{16}`$ were partitioned as two grouped subsets. Each subset was then processed with the pipeline described in Section III-A of the paper with Whisper large-v3-turbo.

The record screening described in Section IV-A.2 was applied to each constructed dataset after transcription, although its criteria depend only on the original record texts. Because the screening was carried out separately for each construction, its counts in Table S2 differ slightly among the constructions.

Although the subsets are disjoint at the record level, samples with identical prompt–completion pairs can appear in different subsets because the Duplicate-Data filter is applied within each subset. Such duplicate samples are removed when the subsets are merged.

## Unification of notational variants

The rules for unifying notational variants before TTS synthesis (Section III-A of the paper) include the following:

- conversion of selected Kanji expressions to Hiragana;
- normalization of half-width Katakana to full-width Katakana;
- normalization of capitalization for clinical terms, such as "spo2" and "SPO2" to "SpO2";
- correction of common Kanji misuse, such as the homophonous misspelling "褥創" to the correct "褥瘡" ("pressure ulcer");
- replacement of special unit symbols with standard notation, such as the Japanese liter symbol to "L"; and
- conversion of selected English expressions to Japanese Katakana.

This unification is distinct from the text normalization used for CER computation (Appendix III of the paper).

## Table S1: Filtering results for each Whisper model

Numbers of records removed by each filter (Appendix I of the paper) from the screened records (S: 37,820; O: 39,223), and the final numbers of records, for the three Whisper models used to generate the training data (Section IV-A.2 of the paper). The Length-Ratio filter is applied only to O records.

| Whisper model | Record type | Duplicate-Data | Length-Ratio | Special-Character | Final |
|---|---|---|---|---|---|
| medium | S | -1,347 | - | -7 | 36,466 |
| | O | -18 | -81 | -2 | 39,122 |
| large | S | -1,453 | - | -1 | 36,366 |
| | O | -41 | -139 | -4 | 39,039 |
| large-v3-turbo | S | -1,377 | - | -3 | 36,440 |
| | O | -40 | -53 | -36 | 39,094 |

## Table S2: Filtering results for each subset (large-v3-turbo)

| Dataset | Initial | Screening | Duplicate-Data | Length-Ratio | Special-Character | Final |
|---|---|---|---|---|---|---|
| $`\mathcal{S}_{1}`$ | 40,350 | -2,530 | -1,377 | - | -3 | 36,440 |
| $`\mathcal{S}_{2}`$ | 40,349 | -2,586 | -1,352 | - | -5 | 36,406 |
| $`\mathcal{S}_{3}`$ | 40,350 | -2,471 | -1,306 | - | -4 | 36,569 |
| $`\mathcal{S}_{4}`$ | 40,349 | -2,403 | -1,397 | - | -6 | 36,543 |
| $`\mathcal{S}_{5}`$ | 40,349 | -2,566 | -1,310 | - | -6 | 36,467 |
| $`\mathcal{S}_{6}`$ | 40,350 | -2,609 | -1,331 | - | -5 | 36,405 |
| $`\mathcal{S}_{7}`$ | 40,348 | -2,552 | -1,332 | - | -6 | 36,458 |
| $`\mathcal{S}_{8}`$ | 40,349 | -2,480 | -1,337 | - | -5 | 36,527 |
| $`\mathcal{O}_{1}`$ | 40,051 (40,020+31) | -828 | -40 | -53 | -36 | 39,094 |
| $`\mathcal{O}_{2}`$ | 40,019 | -855 | -56 | -53 | -34 | 39,021 |
| $`\mathcal{O}_{3}`$ | 40,020 | -768 | -44 | -62 | -26 | 39,120 |
| $`\mathcal{O}_{4}`$ | 40,020 | -810 | -34 | -55 | -23 | 39,098 |
| $`\mathcal{O}_{5}`$ | 40,020 | -802 | -51 | -58 | -28 | 39,081 |
| $`\mathcal{O}_{6}`$ | 40,020 | -810 | -43 | -57 | -24 | 39,086 |
| $`\mathcal{O}_{7}`$ | 40,020 | -881 | -57 | -56 | -23 | 39,003 |
| $`\mathcal{O}_{8}`$ | 40,020 | -846 | -35 | -59 | -32 | 39,048 |
| $`\mathcal{O}_{9} \cup \cdots \cup \mathcal{O}_{12}`$ | 160,079 | -3,250 | -441 | -214 | -111 | 156,063 |
| $`\mathcal{O}_{13} \cup \cdots \cup \mathcal{O}_{16}`$ | 160,080 | -3,213 | -438 | -219 | -115 | 156,095 |
| $`\mathcal{S}_{1}^{(1)}`$ | 40,350 | -2,530 | -1,358 | - | -5 | 36,457 |
| $`\mathcal{O}_{1}^{(1)}`$ | 40,052 (40,020+32) | -827 | -29 | -64 | -31 | 39,101 |
| $`\mathcal{S}_{1}^{(2)}`$ | 40,350 | -2,530 | -1,364 | - | -2 | 36,454 |
| $`\mathcal{O}_{1}^{(2)}`$ | 40,051 (40,020+31) | -828 | -31 | -51 | -26 | 39,115 |
| $`\mathcal{S}_{1}^{(3)}`$ | 40,350 | -2,530 | - | - | - | 37,820 |
| $`\mathcal{O}_{1}^{(3)}`$ | 40,051 (40,020+31) | -828 | - | - | - | 39,223 |
| $`\mathcal{S}_{1}^{(4)}`$ | 40,350 | - | - | - | - | 40,350 |
| $`\mathcal{O}_{1}^{(4)}`$ | 40,020 | - | - | - | - | 40,020 |

Initial: number of records at extraction. Screening: the record screening described in Section IV-A.2. Final: final number of records after filtering. Of the 2,530 records excluded from $`\mathcal{S}_{1}`$ by the record screening, 31 were confirmed to be O records that had been registered as S by mistake and were added to $`\mathcal{O}_{1}`$, hence its initial size 40,020+31. This transfer was performed only for $`\mathcal{S}_{1}`$ and $`\mathcal{O}_{1}`$ (including their reconstructions for Comparisons 1–3). In the other subsets, such records were excluded by the screening but were not added to the corresponding O datasets. $`\mathcal{S}_{1}^{(c)}`$ and $`\mathcal{O}_{1}^{(c)}`$ denote the training datasets of Comparison $`c`$ in Section IV-A.5: $`\mathcal{S}_{1}^{(1)}`$ and $`\mathcal{O}_{1}^{(1)}`$ were constructed without any dictionary update, $`\mathcal{S}_{1}^{(2)}`$ and $`\mathcal{O}_{1}^{(2)}`$ with the Manbyo dictionary only, $`\mathcal{S}_{1}^{(3)}`$ and $`\mathcal{O}_{1}^{(3)}`$ are $`\mathcal{S}_{1}`$ and $`\mathcal{O}_{1}`$ without the deletions by the three filters, and $`\mathcal{S}_{1}^{(4)}`$ and $`\mathcal{O}_{1}^{(4)}`$ are the extracted records without any of the preprocessing steps.

## Table S3: Training dataset variants

| Label | Composition | Total Number of Records |
|---|---|---|
| Sx0.5, Ox0.5 | half of $`\mathcal{S}_{1}`$ $`\cup`$ half of $`\mathcal{O}_{1}`$ | 37,767 |
| Ox2 | $`\mathcal{O}_{1} \cup \mathcal{O}_{2}`$ | 78,059 |
| $`\mathrm{Sx}k`$, $`\mathrm{Ox}k`$ | $`(\mathcal{S}_{1} \cup \cdots \cup \mathcal{S}_{k}) \cup (\mathcal{O}_{1} \cup \cdots \cup \mathcal{O}_{k})`$ | see below |
| Sx1, $`\mathrm{Ox}k`$ | $`\mathcal{S}_{1} \cup \mathcal{O}_{1} \cup \cdots \cup \mathcal{O}_{k}`$ | see below |
| Sx2, $`\mathrm{Ox}k`$ | $`\mathcal{S}_{1} \cup \mathcal{S}_{2} \cup \mathcal{O}_{1} \cup \cdots \cup \mathcal{O}_{k}`$ | see below |
| $`\mathrm{Sx}k`$, Ox8 | $`(\mathcal{S}_{1} \cup \cdots \cup \mathcal{S}_{k}) \cup (\mathcal{O}_{1} \cup \cdots \cup \mathcal{O}_{8})`$ | see below |
| $`\mathrm{Sx}k`$, Ox16 | $`(\mathcal{S}_{1} \cup \cdots \cup \mathcal{S}_{k}) \cup (\mathcal{O}_{1} \cup \cdots \cup \mathcal{O}_{16})`$ | see below |

## Table S4: Training dataset sizes for large-v3-turbo

| $`k`$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| Duplicates removed (S) | 0 | 504 | 699 | 896 | 990 | 1,112 | 1,172 | 1,286 | - | - |
| $`\lvert \mathcal{S}_{1} \cup \cdots \cup \mathcal{S}_{k} \rvert`$ | 36,440 | 72,342 | 108,212 | 143,859 | 179,336 | 214,629 | 249,915 | 285,156 | - | - |
| Duplicates removed (O) | 0 | 56 | 72 | 100 | 131 | 163 | 159 | 193 | 565 | 677 |
| $`\lvert \mathcal{O}_{1} \cup \cdots \cup \mathcal{O}_{k} \rvert`$ | 39,094 | 78,059 | 117,107 | 156,105 | 195,055 | 233,978 | 272,822 | 311,677 | 467,175 | 622,593 |
| Total: $`\mathrm{Sx}k`$, $`\mathrm{Ox}k`$ | 75,534 | 150,401 | 225,319 | 299,964 | 374,391 | 448,607 | 522,737 | 596,833 | - | - |
| Total: Sx1, $`\mathrm{Ox}k`$ | 75,534 | 114,499 | 153,547 | 192,545 | 231,495 | 270,418 | 309,262 | 348,117 | 503,615 | 659,033 |
| Total: Sx2, $`\mathrm{Ox}k`$ | - | 150,401 | 189,449 | 228,447 | 267,397 | 306,320 | 345,164 | 384,019 | 539,517 | 694,935 |
| Total: $`\mathrm{Sx}k`$, Ox8 | 348,117 | 384,019 | 419,889 | 455,536 | 491,013 | 526,306 | 561,592 | 596,833 | - | - |
| Total: $`\mathrm{Sx}k`$, Ox16 | 659,033 | 694,935 | 730,805 | 766,452 | 801,929 | 837,222 | 872,508 | 907,749 | - | - |

Duplicate records arising when the constituent datasets are merged are removed. "Duplicates removed (S)" and "(O)" give the number of records removed when $`\mathcal{S}_{k}`$ (resp. $`\mathcal{O}_{k}`$) is merged into $`\mathcal{S}_{1} \cup \cdots \cup \mathcal{S}_{k-1}`$ (resp. $`\mathcal{O}_{1} \cup \cdots \cup \mathcal{O}_{k-1}`$). For $`k = 12`$ and $`16`$, the grouped datasets $`\mathcal{O}_{9} \cup \cdots \cup \mathcal{O}_{12}`$ and $`\mathcal{O}_{13} \cup \cdots \cup \mathcal{O}_{16}`$ were merged, respectively. Because no duplicate records exist between the S and O datasets, the total number of records of $`\mathrm{Sx}j`$, $`\mathrm{Ox}k`$ is $`\lvert \mathcal{S}_{1} \cup \cdots \cup \mathcal{S}_{j} \rvert + \lvert \mathcal{O}_{1} \cup \cdots \cup \mathcal{O}_{k} \rvert`$. The S datasets exist only up to $`k = 8`$, and the corresponding cells for $`k = 12`$ and $`16`$ are marked "-".
