#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MAS=macnet      # autogen, macnet
MAS_RAG=generative  # metagpt, generative, voyager, gmemory, oagent, master
MAS_LLM=Qwen/Qwen3-4B-Instruct-2507  # Qwen/Qwen3-4B-Instruct-2507, meta-llama/Llama-3.1-8B-Instruct

DATASET=kodcode  # kodcode, triviaqa, popqa

python main.py \
    --cfg-path configs/memmaster/${DATASET}.yaml \
    --options \
    model.mas.structure ${MAS} \
    model.mas.llm_name_or_path ${MAS_LLM} \
    model.memory.llm_name_or_path ${MAS_LLM} \
    model.memory.rag.mode ${MAS_RAG} \
    model.memory.use_weaver False \
    run.mode data \


if [ $? -eq 0 ]; then
    echo "✅ Memory Master finished successfully."
else
    echo "❌ Memory Master failed with an error."
fi