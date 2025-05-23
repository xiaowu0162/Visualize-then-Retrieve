#!/usr/bin/env bash

HOME_DIR=`realpath .`
export PYTHONPATH=${PYTHONPATH}:${HOME_DIR}
export PYTHONIOENCODING=utf-8
mkdir -p ${HOME_DIR}/logs/retrieval/t2i_generations/

GPU=$1
export CUDA_VISIBLE_DEVICES=${GPU}
QUERY_TYPE=$2                                   # original_query, llm_extracted_text, t2i_llm
EMBEDDING_MODEL=$3                              # clip, e5-v
QUERY_REPHRASE_MODEL_ALIAS=${4:-"gpt-4o"}       # gpt-4o, llama-3.1-8b-instruct, llama-3.3-70b-instruct
RUN_ORIG_QUERY=${5:-"false"}                    # true, false  
T2I_MODEL=${6:-"image-1-high"}                  # image-1-high, image-1-medium, image-1-low, dalle, sd3, sd3-5
T2I_N_IMAGES=${7:-"1"}
T2I_RRF_K=${8:-"1"}                             # 1, 3, 5

if [ "$RUN_ORIG_QUERY" == "true" ]; then
    RUN_ORIG_QUERY_FLAG="--run_orig_query"
else
    RUN_ORIG_QUERY_FLAG=""
fi

if [ "$T2I_N_IMAGES" == "1" ]; then
    T2I_N_IMAGES_FLAG="--n_images ${T2I_N_IMAGES}"
else
    T2I_N_IMAGES_FLAG="--n_images ${T2I_N_IMAGES} --multi_image_aggregation rank --rrf_k ${T2I_RRF_K}"
fi

declare -A model_zoo
model_zoo['gpt-4o']="gpt-4o-2024-08-06"
model_zoo['llama-3.1-8b-instruct']="meta-llama/Meta-Llama-3.1-8B-Instruct"
model_zoo['llama-3.3-70b-instruct']="meta-llama/Llama-3.3-70B-Instruct"
query_rephrase_model=${model_zoo["$QUERY_REPHRASE_MODEL_ALIAS"]}

python run_retrieval.py \
    --query_type ${QUERY_TYPE} ${RUN_ORIG_QUERY_FLAG} \
    --embedding_model_name ${EMBEDDING_MODEL} \
    --query_expansion_model_name ${query_rephrase_model} \
    --t2i_model_name ${T2I_MODEL} ${T2I_N_IMAGES_FLAG} \
    --data_dir ${HOME_DIR}/data/ \
    --output_dir ${HOME_DIR}/logs/retrieval/ \
    --img_generation_dir ${HOME_DIR}/logs/retrieval/t2i_generations/ \
    --run_id $(date +'%Y%m%d-%H%M')

