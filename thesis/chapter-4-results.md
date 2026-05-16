# Chapter 4: Results

This chapter presents the empirical findings of the study. The primary analysis concerns Experiment 2, the benchmark judgment task, which is the only experiment capable of answering the research questions. Experiment 1 (perceptual discrimination) is reported in Section 4.6 as a supplementary sidecar that contextualises Experiment 2 patterns. The chapter is organised as follows:

- [4.1 Data Quality and Corpus Overview](4.1-data-quality.md) — Parse success rates, stimulus coverage, and the distribution of records across models and experiments.

- [4.2 Experiment 2 Control Conditions](4.2-control-conditions.md) — Rubric validity from the instruction-not-followed control and specificity floor from the correct-edit control. These baselines are prerequisite for interpreting all Experiment 2 results.

- [4.3 Main Experiment 2 Results](4.5-exp2-edit-judgment.md) — Per-model score descriptives, per-dimension magnitude sensitivity, and below-baseline quality-discrimination thresholds under the actual benchmark task. Addresses RQ1, RQ1a, and RQ1b.

- [4.4 Research Question Answers](4.4-rq-answers.md) — Direct answers to RQ1, RQ1a, and RQ1b, drawn from the Experiment 2 evidence in Section 4.3.

- [4.5 Supporting Analyses](4.5-supporting-analyses.md) — Rubric internal consistency (Cronbach's alpha, intercorrelations) and inter-model agreement on Experiment 2 classifications (WIP).

- [4.6 Phase 1: Perceptual Discrimination Sidecar](4.6-perception-judgment-gap.md) — Supplementary perceptual context from Experiment 1: controls, per-dimension detection rates, detection thresholds, and the correlation between Phase 1 and Phase 2 patterns. Experiment 1 documents the visual sensitivity available to each model independently of the benchmark task.

---

## Chapter Overview

The study evaluates three VLMs, Qwen2.5-VL, Gemini 3.1 Pro Preview, and GPT-4o, as judges of text-in-image editing quality, using 623 stimuli that span 13 degradation dimensions and 8 edit types. Each stimulus was evaluated under two experimental conditions: Experiment 1 (perceptual sensitivity, comparing ground truth to degraded image) and Experiment 2 (edit judgment, comparing source to degraded image given an instruction). Together, these yield 3,729 model responses.

**Experiment 2 is the benchmark-facing task and is the primary analysis.** It replicates the conditions under which VLM judges are deployed in text-editing benchmarks. The research questions are answered from Experiment 2 evidence. Experiment 1 is reported as a supplementary sidecar (Section 4.6) because, while it documents raw perceptual discrimination capacity, it cannot answer questions about benchmark judgment reliability.

The results are organised around the three research questions, all answered from Experiment 2:

**RQ1** (*Overall reliability*) is addressed in Section 4.3.2, which characterises each model's aggregate score sensitivity across the full degraded stimulus set, and in Section 4.2, which establishes the control baselines against which those scores are interpreted.

**RQ1a** (*Error-type variation*) is addressed in Section 4.3.3, which breaks down magnitude sensitivity by degradation dimension and model.

**RQ1b** (*Magnitude transitions*) is addressed in Section 4.3.4, which estimates the degradation magnitudes at which each model's quality scores first fall below the correct-edit baseline.

The central finding is that VLM judge reliability is both dimension-dependent and model-dependent to a degree that calls the general use of these models as benchmark judges into question. Models show meaningful score sensitivity on image-level degradations (blur, Gaussian noise) but fail to penalise colour, font, and spatial errors until magnitudes far above what constitutes a perceptible flaw. Section 4.6 shows that this pattern is not merely an artefact of the benchmark-task framing: models that cannot see an error in the perceptual task (Experiment 1) do not catch it in the judgment task either. These findings are interpreted further in Chapter 5.
