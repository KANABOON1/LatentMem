#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MAS=autogen  # autogen, macnet
MAS_LLM=Qwen/Qwen3-4B-Instruct-2507  # Qwen/Qwen3-4B-Instruct-2507, meta-llama/Llama-3.1-8B-Instruct
LLM_SUFFIX="${MAS_LLM##*/}"

MAS_RAG=master  

DATASET=kodcode  # kodcode, triviaqa, popqa, pddl
DATABASE_DIR="<Memory Repo Path Here> (e.g., /rag_0/)"

ROLLOUT_BATCHSIZE=8

LOSS_FUNC=bnpo

accelerate launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/memmaster/${DATASET}.yaml \
    --options \
    model.mas.structure ${MAS} \
    model.mas.llm_name_or_path ${MAS_LLM} \
    model.memory.llm_name_or_path ${MAS_LLM} \
    model.memory.use_weaver True \
    model.memory.rag.mode ${MAS_RAG} \
    model.memory.rag.database_dir ${DATABASE_DIR} \
    model.memory.weaver.latents_len 8 \
    dataset.mode grpo \
    run.mode grpo \
    run.grpo.loss_type ${LOSS_FUNC} \
    run.grpo.per_device_train_batch_size ${ROLLOUT_BATCHSIZE} \
    run.grpo.per_device_eval_batch_size ${ROLLOUT_BATCHSIZE} \
    run.grpo.num_generations ${ROLLOUT_BATCHSIZE} \
    run.grpo.gradient_accumulation_steps 1 \