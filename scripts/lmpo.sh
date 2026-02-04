#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

MAS=autogen  # autogen, macnet
MAS_LLM=Qwen/Qwen3-4B-Instruct-2507  # Qwen/Qwen3-4B-Instruct-2507, meta-llama/Llama-3.1-8B-Instruct
LLM_SUFFIX="${MAS_LLM##*/}"

MAS_RAG=latentmem
LOAD_MODEL_PATH="/mnt/shared-storage-user/fumuxin/MemMaster/results/sft/Qwen3-4B-Instruct-2507/in_domain/all_mas/mem=master_ll=8_lora=True/model.safetensors"  

DATASET=kodcode  # kodcode, triviaqa, popqa, pddl
DATABASE_DIR="/mnt/shared-storage-user/fumuxin/MemMaster/results/data/Qwen3-4B-Instruct-2507/in_domain/all_mas/master/rag_0"

ROLLOUT_BATCHSIZE=8

LOSS_FUNC=bnpo

accelerate launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latentmem/${DATASET}.yaml \
    --options \
    model.load_model_path ${LOAD_MODEL_PATH} \
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