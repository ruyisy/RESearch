# RESearch: Reflection-Enhanced Search Agent for Risk Control

This repository contains the code for the paper **"Controlling Risk in Search Agents: Knowing When They Do Not Know"**.

## 📌 Overview

**Risk Control for Search Agents (RC-Search)** is a critical task for deploying reliable search agents. As shown below, search agents often hallucinate answers when retrieval is noisy or insufficient.

<p align="center">
  <img src="figs/intro.png" width="50%">
</p>

To address this, we propose **RESearch**, a framework that introduces a **Reflection Phase** after the standard execution phase. By modeling the full **execution-reflection trajectory** and optimizing it via PPO with a risk-aware reward, RESearch enables agents to:
1.  **Maintain confidence** when the execution is correct.
2.  **Discard incorrect answers** (via reflection) when risks are detected.
3.  **Safely fail** (passive failure) when information is insufficient.

<p align="center">
  <img src="figs/Method.png" width="100%">
</p>


## 🚀 Training

To train the RESearch agent using PPO with the reflection mechanism enabled:

```bash
# Set your data and model paths in the script first
bash train_ppo.sh
```

Key configuration in `train_ppo.sh`:
- `+enable_reflection=true`: Enables the reflection phase generation and dual-stage reward calculation.

## 🙏 Acknowledgements

This work is implemented based on [Search-R1](https://github.com/HKUST-KnowComp/Search-R1), [StepSearch](https://github.com/Zillwang/StepSearch), and [veRL](https://github.com/volcengine/verl). We sincerely thank the authors of these projects for their valuable contributions to the open-source community.

## ⚠️ Note

This repository is a preliminary version provided for **peer review purposes only**. The full codebase, including data processing scripts and comprehensive evaluation tools, will be released upon acceptance.
