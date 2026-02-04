#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MAS=autogen      # autogen, macnet
MAS_RAG=latentmem  # metagpt, generative, voyager, gmemory, oagent, latentmem
MAS_LLM=Qwen/Qwen3-4B-Instruct-2507  # Qwen/Qwen3-4B-Instruct-2507, meta-llama/Llama-3.1-8B-Instruct

DATASET=kodcode  # in-domain datasets: kodcode, triviaqa, popqa

python main.py \
    --cfg-path configs/latentmem/${DATASET}.yaml \
    --options \
    model.mas.structure ${MAS} \
    model.mas.llm_name_or_path ${MAS_LLM} \
    model.memory.llm_name_or_path ${MAS_LLM} \
    model.memory.rag.mode ${MAS_RAG} \
    model.memory.use_weaver False \
    run.mode data \