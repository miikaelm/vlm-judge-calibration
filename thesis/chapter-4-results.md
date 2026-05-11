# Chapter 4: Results

This chapter presents the empirical findings of the study in seven sections:

- [4.1 Data Quality and Corpus Overview](4.1-data-quality.md) — Parse success rates, stimulus coverage, and the distribution of records across models and experiments.

- [4.2 Control Condition Results](4.2-control-conditions.md) — False positive rates from the identity (noop) control, sensitivity floor from the source-target control, and rubric validity from the instruction-not-followed control. These baselines are prerequisite for interpreting all main results.

- [4.3 Experiment 1: Perceptual Sensitivity](4.3-exp1-perceptual-sensitivity.md) — Per-model overall detection rates, per-dimension detection rates with blind/sensitive classifications, and Spearman correlations between magnitude and similarity score. Addresses RQ1 and RQ1a.

- [4.4 Detection Thresholds](4.4-detection-thresholds.md) — Tier-level detection rates per dimension and the discrete JND estimates (threshold tier at 50% and 75% detection) in both tier labels and physical units. Addresses RQ1b.

- [4.5 Experiment 2: Edit Judgment Sensitivity](4.5-exp2-edit-judgment.md) — Per-dimension overall_quality scores, rubric dimension descriptives, internal consistency, and Spearman correlations between magnitude and scores. Addresses RQ1 and RQ1a under the actual benchmark task.

- [4.6 The Perception-to-Judgment Gap](4.6-perception-judgment-gap.md) — The score gap between Experiment 2 overall_quality and Experiment 1 similarity_score on matched stimuli, global cross-experiment Spearman correlations, and the rate at which detected differences are also penalised in Experiment 2. Addresses RQ1b.

- [4.7 Cross-Model Agreement on Blind Spots](4.7-cross-model-blind-spots.md) — The blind/sensitive classification table across all (dimension × model) cells, universally and model-specifically blind dimensions, and Fleiss' kappa as a summary of inter-model agreement. Addresses RQ1a.

---

## Chapter Overview

The study evaluates three VLMs — Qwen2.5-VL, Gemini 3.1 Pro Preview, and GPT-4o as judges of text-in-image editing quality, using 623 stimuli that span 13 degradation dimensions and 8 edit types. Each stimulus was evaluated under two experimental conditions: Experiment 1 (perceptual sensitivity, comparing ground truth to degraded image) and Experiment 2 (edit judgment, comparing source to degraded image given an instruction). Together, these yield 3,729 model responses.

The results are organised around the three research questions:

**RQ1** (*Overall reliability*) is addressed primarily in Sections 4.3 and 4.5, which characterise each model's aggregate detection rate and score sensitivity across the full degraded stimulus set, and in Section 4.2, which establishes the false positive baseline against which detection rates are interpreted.

**RQ1a** (*Error-type variation*) is addressed in Sections 4.3, 4.5, and 4.7, which break down detection rates, score sensitivity, and blind/sensitive classifications by degradation dimension and model, and summarise inter-model agreement on which dimensions constitute blind spots.

**RQ1b** (*Magnitude transitions*) is addressed in Sections 4.4 and 4.6, which map the tier-level detection curves and estimate the degradation magnitudes at which each model transitions from unreliable to reliable detection, and measure the additional cost of the instruction-reasoning step.

The central finding is that VLM judge reliability is both dimension-dependent and model-dependent to a degree that calls the general use of these models as benchmark judges into question. All models show some statistical sensitivity above their near-zero FPR baselines on almost all dimensions, but detection rates range from 0% (Qwen on font style and weight) to 95% (GPT-4o on blur), and several practically important dimensions are detected at very low absolute rates. The thresholds at which reliable detection begins are, for many dimensions, far above the error magnitudes that would constitute a perceptible flaw to a human observer. These findings are interpreted further in Chapter 5.
