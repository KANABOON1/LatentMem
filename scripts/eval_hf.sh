#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# options: autogen, macnet, camel
MAS=autogen  

# options: True for latentmem, False for baselines
USE_WEAVER=True  

# options: metagpt, generative, voyager, gmemory, oagent, latentmem
MAS_RAG=latentmem  

# options: kodcode, triviaqa, popqa, pddl
DATASET=kodcode  

# options: 
# baselines: results/LatentMem-Qwen3-4B-Trajectory/{baseline}/rag_0
# latentmem: results/LatentMem-Qwen3-4B-Trajectory/latentmem/rag_0 or results/LatentMem-Qwen3-4B/data/rag_0
DATABASE_DIR="results/LatentMem-Qwen3-4B/data/rag_0"

MAS_LLM=Qwen/Qwen3-4B-Instruct-2507
LLM_SUFFIX="${MAS_LLM##*/}"
LOAD_MODEL_PATH="results/LatentMem-Qwen3-4B/model/model.safetensors"

python main.py \
    --cfg-path "configs/latentmem/${DATASET}.yaml" \
    --options \
    model.mas.structure ${MAS} \
    model.mas.llm_name_or_path "${MAS_LLM}" \
    model.load_model_path "${LOAD_MODEL_PATH}" \
    model.memory.llm_name_or_path "${MAS_LLM}" \
    model.memory.rag.mode ${MAS_RAG} \
    model.memory.rag.database_dir "${DATABASE_DIR}" \
    model.memory.use_weaver ${USE_WEAVER} \
    run.mode eval \
