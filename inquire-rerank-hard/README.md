# Visual-RAG-ME 

This folder contains Inquire-Rerank-Hard, a hard subset of Inquire-Rerank.

## Preparation

#### Data

We release the questions at `data/inquire_rerank_hard_final.json`. To run the experiments, download the iNaturalist 2021 dataset [here](https://github.com/visipedia/inat_comp/tree/master/2021) and place the images mentioned in the data under the `data/images/` folder. Each species should have its own folder, e.g., `data/images/00855_Animalia_Arthropoda_Insecta_Lepidoptera_Cossidae_Prionoxystus_robiniae/`, which contains all the jpg files associated with the species. 

#### Environment

Please follow the instruction in the root folder of this project to setup a conda environment.

#### API Key

Place your OpenAI API key and organization (optional) in `constants.py`.

## Run Retrieval

To run the retrieval experiments, you can run the scirpt `run_retrieval.sh`, which calls `run_retrieval.py` for you. `run_retrieval.sh` takes eight arguments in the following format:

```
bash run_retrieval.sh GPU QUERY_TYPE EMBEDDING_MODEL QUERY_REPHRASE_MODEL_ALIAS RUN_ORIG_QUERY T2I_MODEL T2I_N_IMAGES T2I_RRF_K
```
* `QUERY_TYPE`: `original_query`, `llm_extracted_text`, or `t2i_llm`.
* `EMBEDDING_MODEL`: `clip` or `e5-v`.
* `QUERY_REPHRASE_MODEL_ALIAS`: the model used to expand query. Only used when `QUERY_TYPE` is `llm_extracted_text` or `t2i_llm`. We support OpenAI models or vllm-served local OpenAI emulators.
* `RUN_ORIG_QUERY`: `true` or `false`. Whether run the original T2I retrieval baseline as a reference. Only used when `QUERY_TYPE` is `llm_extracted_text` or `t2i_llm`.
* `T2I_MODEL`: `image-1-high`, `image-1-medium`, `image-1-low`, `dalle`, `sd3`, and `sd3-5`.
* `T2I_N_IMAGES`: the number of images to generate. Only used when `QUERY_TYPE` is `t2i_llm`.
* `T2I_RRF_K`: the damping coefficient for RRF. Only ued when `QUERY_TYPE` is `t2i_llm` and `T2I_N_IMAGES` > 1.

At the end, a retrieval log will be generated under `logs/retrieval/`. Below, we list the sample commands for reproducing the experiments in the paper. 

#### Running baseline T2I retrieval

```
bash run_retrieval.sh 0 original_query clip
```

#### T2I Retrieval with LLM query expansion

```
bash run_retrieval.sh 0 llm_extracted_text clip gpt-4o true
```
* Note: if you use a non-OpenAI model such as llama, follow [this link](https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html) to serve the model locally via vllm. Then, change the port in the function `extract_core_info_with_llm` to reflect the actual port on which you serve the model. 

#### VisRet

```
bash run_retrieval.sh 0 t2i_llm clip gpt-4o true image-1-high 3 1
```
* Note: our code saves the generated images at `logs/retrieval/t2i_generations`. If you wish to reuse the generated images for retrieval evaluation, pass the directory into `run_retrieval.py` via the flag `--image_gen_reuse_dir`.
