# LatentMem: Customizing Latent Memory for Multi-Agent Systems

## 👋 Introduction
This repo is the official implementation of ***LatentMem: Customizing Latent Memory for Multi-Agent Systems***.

**LatentMem** is a learnable multi-agent memory framework that generates role-aware, token-efficient latent memories for LLM-powered multi-agent systems (MAS). It combines a lightweight experience bank to store raw interaction trajectories with a memory composer that distills compact, agent-specific latent memories conditioned on agent profiles. Using Latent Memory Policy Optimization (LMPO), LatentMem encourages the memory composer to produce high-utility, transferable representations. Experiments show LatentMem improves performance, reduces token usage, and generalizes well across diverse MAS tasks and benchmarks.

## 🌎 Setup
```
conda env create -f environment.yml
conda activate latentmem
```

## 🚀 Quick Start
### 🔧 Installation: Set Up Search Environment
Please follow the instructions in the [Search-R1](https://github.com/PeterGriffinJin/Search-R1?tab=readme-ov-file#retriever-environment-optional) to configure the retriever environment (optional).

### ▶️ How to Run
Before starting LMPO training, a data collection phase must be conducted first.

#### Step 1: Data Collection
```bash
bash data.sh
```

#### Step 2: LMPO Train
```bash
bash lmpo_train.sh
```
