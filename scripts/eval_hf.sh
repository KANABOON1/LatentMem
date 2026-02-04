#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MAS=autogen  # autogen, macnet, camel
MAS_LLM=Qwen/Qwen3-4B-Instruct-2507
LLM_SUFFIX="${MAS_LLM##*/}"
USE_WEAVER=True

MAS_RAG=latentmem

DATASET=kodcode  # kodcode, triviaqa, popqa, pddl
DATABASE_DIR="results/LatentMem-Qwen3-4B/data/rag_0"

LOAD_MODEL_PATH="results/LatentMem-Qwen3-4B/model/model.safetensors"

python main.py \
    --cfg-path configs/latentmem/${DATASET}.yaml \
    --options \
    model.mas.structure ${MAS} \
    model.mas.llm_name_or_path ${MAS_LLM} \
    model.load_model_path ${LOAD_MODEL_PATH} \
    model.memory.llm_name_or_path ${MAS_LLM} \
    model.memory.rag.mode ${MAS_RAG} \
    model.memory.rag.database_dir ${DATABASE_DIR} \
    model.memory.use_weaver ${USE_WEAVER} \
    run.mode eval \
