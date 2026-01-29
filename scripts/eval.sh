#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MAS=autogen  # autogen, macnet, camel
MAS_LLM=Qwen/Qwen3-4B-Instruct-2507 # Qwen/Qwen3-4B-Instruct-2507, meta-llama/Llama-3.1-8B-Instruct
LLM_SUFFIX="${MAS_LLM##*/}"
USE_WEAVER=True  # True or False

MAS_RAG=master  # metagpt, generative, voyager, gmemory, oagent, master

DATASET=kodcode  # kodcode, triviaqa, popqa, pddl
DATABASE_DIR="<Memory Repo Path Here> (e.g., rag_0/)"

LOAD_MODEL_PATH="<Trained Model Path Here>"

python main.py \
    --cfg-path configs/memmaster/${DATASET}.yaml \
    --options \
    model.mas.structure ${MAS} \
    model.mas.llm_name_or_path ${MAS_LLM} \
    model.load_model_path ${LOAD_MODEL_PATH} \
    model.memory.llm_name_or_path ${MAS_LLM} \
    model.memory.rag.mode ${MAS_RAG} \
    model.memory.rag.database_dir ${DATABASE_DIR} \
    model.memory.use_weaver ${USE_WEAVER} \
    model.memory.rag.pos_shots_num 1 \
    run.mode eval \
