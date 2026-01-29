#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MAS=autogen  # autogen, macnet
MAS_RAG=master
MAS_LLM=Qwen/Qwen3-4B-Instruct-2507  # Qwen/Qwen3-4B-Instruct-2507, meta-llama/Llama-3.1-8B-Instruct
LLM_SUFFIX="${MAS_LLM##*/}"

DATASET=kodcode
DATA_PATH="<Collected Data Path Here> (e.g., /data.json)"

accelerate launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/memmaster/${DATASET}.yaml \
    --options \
    model.mas.structure ${MAS} \
    model.mas.llm_name_or_path ${MAS_LLM} \
    model.memory.llm_name_or_path ${MAS_LLM} \
    model.memory.rag.mode ${MAS_RAG} \
    model.memory.use_weaver True \
    model.memory.weaver.latents_len 8 \
    dataset.bootstrapped_data_path ${DATA_PATH} \
    run.mode sft \
    run.sft.per_device_train_batch_size 2 \
    run.sft.per_device_eval_batch_size 2 \
    run.sft.gradient_accumulation_steps 1 \

    
