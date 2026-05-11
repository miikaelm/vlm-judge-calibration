# Chapter 3: Methodology

This chapter describes the research design, stimulus generation pipeline, VLM evaluation protocol, and analysis methods used in the study. It is divided into four sections:

- [3.1 Design Rationale](3.1-design-rationale.md) — Why this experimental paradigm? Motivation for controlled degradations, the two-experiment structure, and positioning relative to prior benchmark evaluation approaches.

- [3.2 Stimulus Generation](3.2-stimulus-generation.md) — The rendering pipeline (Playwright/Chromium), the layout system, the twelve HTML-layer and three image-layer degradation dimensions, magnitude parameterisation with continuous jitter, seed management, and sample size.

- [3.3 Evaluation Protocol](3.3-evaluation-protocol.md) — Models evaluated, prompt design for Experiment 1 (perceptual sensitivity) and Experiment 2 (edit judgment), response parsing, noop controls, (note, remove: and API cost management).

- [3.4 Analysis Methods](3.4-analysis-methods.md) — Formal metric definitions (detection rate, false positive rate, blind-spot rate, score gap, JND estimate), statistical tests (Spearman correlation, point-biserial, Cronbach's alpha), and planned analyses.

---

## Overview

The study measures how reliably Vision-Language Models can detect controlled, quantified errors in text-in-image editing results. The core design is a sensitivity analysis of VLM judges via controlled degradation stimuli: we generate image pairs where the "edited" result contains a precisely parameterised degradation from the correct output, evaluate multiple VLMs on these pairs using the same prompts as existing text-editing benchmarks, and measure how judge scores change as the size of the error grows, separately for each error type.

The stimulus generation pipeline produces triples of images (source, ground truth, degraded) rendered from HTML/CSS templates via a headless Chromium browser. This rendering-based approach gives pixel-accurate ground truth for each of twelve quantitative degradation dimensions (colour offset in CIEDE2000 ΔE, spatial displacement in pixels, scaling error as percentage, rotation in degrees, font weight as CSS numeric value, font style, letter spacing in pixels, opacity reduction, font family substitution) plus three image-level degradations (Gaussian noise, JPEG compression, Gaussian blur).

Degradation magnitudes are sampled continuously from per-tier windows rather than fixed points, producing a more faithful characterisation of the sensitivity curve. All stimuli are fully reproducible from their metadata via a seeded random number generator. The resulting corpus of 703 stimuli spans eight edit types, twelve-plus degradation dimensions, and four layouts.

Each stimulus is evaluated by each VLM under two experimental conditions. Experiment 1 (perceptual sensitivity) presents the model with the ground truth and degraded images and asks only whether they differ, isolating raw visual discrimination. Experiment 2 (edit judgment) presents the source, edit instruction, and degraded result, the actual task benchmarks require VLMs to perform. The gap between Experiment 1 and Experiment 2 scores on the same stimulus tests whether raw sensitivity to a degradation translates into correctly judging edit quality, and quantifies any reasoning overhead cost if one exists.

The analysis maps per-dimension sensitivity curves, estimates just-noticeable-difference thresholds, computes blind-spot rates, and ranks dimensions by model sensitivity. Statistical inference is based on Spearman rank correlations between numeric magnitude and score, with Benjamini-Hochberg correction for multiple comparisons.
