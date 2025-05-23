import os
import json
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score
from argparse import ArgumentParser
from datetime import datetime
from generation_util import get_dalle_response, get_image1_response, save_img_from_url
import openai
from openai import OpenAI
from constants import OPENAI_API_KEY
from copy import deepcopy


################################################################################
# Helper functions for Evaluation
################################################################################
def recall_at_k(similarity, labels, k, relevant_label, rrf=False, rrf_k=0):
    if rrf:
        # reciprocal rank fusion
        # rrf_k is the damping constant
        sim = np.asarray(similarity)
        if sim.ndim != 2:
            raise ValueError("For rrf=True, similarity must be 2D (n_runs, n_images).")
        n_runs, n_images = sim.shape
        # compute rank positions (1‑based) in each run
        # ranks[r, j] = position of image j in run r
        ranks = np.zeros_like(sim, dtype=int)
        for r in range(n_runs):
            # argsort descending → indices of images from best to worst
            order = np.argsort(sim[r])[::-1]
            # assign rank 1 to best, 2 to second, etc.
            ranks[r, order] = np.arange(1, n_images + 1)
        # RRF score per image: sum_r 1/(rrf_k + rank[r, j])
        rrf_scores = np.sum(1.0 / (rrf_k + ranks), axis=0)
        # take top k by fused score
        scores = rrf_scores
        top_indices = np.argsort(scores)[::-1]
    else:
        scores = similarity
        top_indices = np.argsort(similarity)[::-1]
    return int(any(labels[j] == relevant_label for j in top_indices[:k])), scores, top_indices

def ndcg_at_k(similarity, labels, k, relevant_label, rrf=False, rrf_k=0):
    if rrf:
        # reciprocal rank fusion
        # rrf_k is the damping constant
        sim = np.asarray(similarity)
        if sim.ndim != 2:
            raise ValueError("For rrf=True, similarity must be 2D (n_runs, n_images).")
        n_runs, n_images = sim.shape
        # compute rank positions (1‑based) in each run
        # ranks[r, j] = position of image j in run r
        ranks = np.zeros_like(sim, dtype=int)
        for r in range(n_runs):
            # argsort descending → indices of images from best to worst
            order = np.argsort(sim[r])[::-1]
            # assign rank 1 to best, 2 to second, etc.
            ranks[r, order] = np.arange(1, n_images + 1)
        # RRF score per image: sum_r 1/(rrf_k + rank[r, j])
        rrf_scores = np.sum(1.0 / (rrf_k + ranks), axis=0)
        # take top k by fused score
        top_k = np.argsort(rrf_scores)[::-1][:k]
    else:
        top_k = np.argsort(similarity)[::-1][:k]
    relevance = [1 if labels[j] == relevant_label else 0 for j in top_k]
    ideal = sorted(relevance, reverse=True)
    return ndcg_score([ideal], [relevance]) if any(ideal) else 0.0

def get_ranked_images(cur_index_images_files, raw_scores, aggregated_scores, top_indices):
    # sort cur_index_images_files, raw_scores, aggregated_scores, by top_indices
    if len(cur_index_images_files) == raw_scores.shape[-1] and len(raw_scores.shape) == 2:
        raw_scores = raw_scores.T
    assert len(cur_index_images_files) == raw_scores.shape[0]
    assert len(cur_index_images_files) == len(aggregated_scores)
    assert len(cur_index_images_files) == len(top_indices)
    
    sorted_cur_index_images_files = [cur_index_images_files[i] for i in top_indices]
    sorted_raw_scores = [raw_scores[i] for i in top_indices]
    sorted_aggregated_scores = [aggregated_scores[i] for i in top_indices]
    return sorted_cur_index_images_files, sorted_raw_scores, sorted_aggregated_scores

def organize_metrics(similarity_matrix, ks, recall_key='recall', ncdg_key='ndcg', multi_image_aggregation=None, rrf_k=0):
    raw_scores = similarity_matrix
    if multi_image_aggregation == 'similarity':
        assert len(similarity_matrix.shape) == 2
        raw_scores = raw_scores.T
        similarity_matrix = similarity_matrix.mean(axis=0)
    metrics = {}
    for k in ks:
        r, merged_scores, top_indices = recall_at_k(similarity_matrix, [x[1] for x in cur_index_images], k, 1, rrf=multi_image_aggregation=='rank', rrf_k=rrf_k)
        if k > 1:
            n = ndcg_at_k(similarity_matrix, [x[1] for x in cur_index_images], k, 1, rrf=multi_image_aggregation=='rank', rrf_k=rrf_k)
        else:
            n = r
        metrics[f'{recall_key}@{k}'] = r
        metrics[f'{ncdg_key}@{k}'] = n
    return metrics, raw_scores, merged_scores, top_indices

################################################################################
# Helper functions for CLIP
################################################################################
def clip_retrieval_with_text_query(images, text_query):
    inputs = processor(text=[text_query], images=images, return_tensors="pt", padding=True).to(model.device)
    outputs = model(**inputs)
    similarity = outputs.logits_per_image.squeeze().detach().cpu().numpy()  # shape: [1, num_images]
    return similarity

def clip_retrieval_with_image_query(images, image_query, expected_n_query=None):
    assert type(images) == list and type(image_query) == list
    n_query = len(image_query)
    if expected_n_query is not None:
        assert n_query == expected_n_query
    all_images = images + image_query
    inputs = processor(text=['dummy query'], images=all_images, return_tensors="pt", padding=True).to(model.device)
    outputs = model(**inputs)
    query_embeddings = outputs.image_embeds[-n_query:]
    image_embeddings = outputs.image_embeds[:-n_query]
    similarity = []
    for i_q in range(query_embeddings.size()[0]):
        similarity.append(torch.cosine_similarity(query_embeddings[i_q], image_embeddings))
    similarity = torch.stack(similarity, dim=0)
    # similarity = torch.cosine_similarity(query_embeddings, image_embeddings.T)
    # if n_query > 1:
    #     similarity = similarity.mean(dim=0)      # We have moved this to metric calculation
    if n_query > 1:
        assert similarity.size()[0] == expected_n_query
    similarity = similarity.squeeze().detach().cpu().numpy()  # shape: [num_images] or [expected_n_query, num_images]
    return similarity

################################################################################
# Helper functions for E5-V
################################################################################
def _e5v_encode_images_in_batches(images, batch_size):
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
    img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
    embs = []
    for i in tqdm(range(0, len(images), batch_size), desc='Encoding image batches'):
        batch_imgs = images[i:i + batch_size]
        inputs = processor(images=batch_imgs, text=[img_prompt] * len(batch_imgs), 
                           return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden = outputs.hidden_states[-1][:, -1, :]
            embs.append(F.normalize(hidden, dim=-1))
    return torch.cat(embs, dim=0)  # [N, D]

def _e5v_encode_single_text(text):
    # Encode text
    llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
    text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
    text_inputs = processor(images=None, text=[text_prompt.replace('<sent>', text)], 
                            return_tensors="pt", padding=True).to(model.device)
    # print(text_inputs)
    text_emb = model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
    text_emb = F.normalize(text_emb, dim=-1)
    return text_emb

def e5v_retrieval_with_text_query(images, text_query, image_encoding_batch_size=4):
    text_emb = _e5v_encode_single_text(text_query)
    img_embs = _e5v_encode_images_in_batches(images, batch_size=image_encoding_batch_size)
    similarity = (text_emb @ img_embs.T).squeeze(0).cpu().numpy()
    return similarity

def e5v_retrieval_with_image_query(images, image_query, expected_n_query=None, image_encoding_batch_size=4):
    assert type(images) == list and type(image_query) == list
    n_query = len(image_query)
    if expected_n_query is not None:
        assert n_query == expected_n_query
    all_images = images + image_query
    img_embs = _e5v_encode_images_in_batches(all_images, batch_size=image_encoding_batch_size)
    query_embeddings = img_embs[-n_query:]
    image_embeddings = img_embs[:-n_query]
    similarity = []
    for i_q in range(query_embeddings.size()[0]):
        similarity.append(torch.cosine_similarity(query_embeddings[i_q], image_embeddings))
    similarity = torch.stack(similarity, dim=0)
    if n_query > 1:
        assert similarity.size()[0] == expected_n_query
    similarity = similarity.squeeze().detach().cpu().numpy()  # shape: [num_images] or [expected_n_query, num_images]
    return similarity

################################################################################
# Helper functions for Query Expansion and T2I Instruction Generation
################################################################################
def extract_core_info_with_llm(question, model_name):
    try:
        # just for testing
        if 'gpt' in model_name:
            client = OpenAI(api_key=OPENAI_API_KEY)
        else:
            client = OpenAI(base_url=f"http://localhost:8001/v1", api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model_name,
            # model="meta-llama/Llama-3.3-70B-Instruct",
            # model="Qwen/Qwen2.5-32B-Instruct",
            messages=[
                {
                "role": "user",
                "content": f"You are given an image retrieval query, rephrase the query to add in a bit detail (no longer than 30 words). The rephrased query should highlight the appearance, posture, actions of the main entity so that it is easier to retrieve the desired image among (1) images of the same entity with different posture and (2) images of diffrent entities with the same posture.\nOriginal query: {question}.\n\nRephrased query:",
            }
            ],
            max_tokens=300,
            temperature=0.0,
        )
        output = response.choices[0].message.content
    except openai.APITimeoutError:
        return question
    return output


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--query_type", type=str, choices=['original_query', 'llm_extracted_text', 't2i_llm'])
    parser.add_argument("--embedding_model_name", type=str, default="clip", choices=['clip', 'e5-v'])
    parser.add_argument("--query_expansion_model_name", type=str)
    parser.add_argument("--t2i_model_name", type=str,  choices=['dalle', 'sd3', 'sd3-5', 'image-1-low', 'image-1-medium', 'image-1-high'])
    parser.add_argument("--n_images", type=int, default=1)
    parser.add_argument("--multi_image_aggregation", type=str, default=None, choices=['similarity', 'rank'])
    parser.add_argument("--rrf_k", type=int, default=0)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--img_generation_dir", type=str)
    parser.add_argument("--image_gen_reuse_dir", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--run_orig_query", action="store_true")
    args = parser.parse_args()

    in_data_file = f'{args.data_dir}/inquire_rerank_hard_final.json'
    in_data = json.load(open(in_data_file))

    image_dir = f'{args.data_dir}/images/'

    if args.embedding_model_name == 'clip':
        model_name = 'openai/clip-vit-large-patch14-336'
        model = CLIPModel.from_pretrained(model_name).cuda()
        processor = CLIPProcessor.from_pretrained(model_name, device="cuda") #, use_fast=True)
    elif args.embedding_model_name == 'e5-v':
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        model_name = 'royokong/e5-v'
        model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
        processor = LlavaNextProcessor.from_pretrained(model_name)
    else:
        print('Embedding model type not implemented!')
        exit()
    
    if args.image_gen_reuse_dir:
        print('Warning: reusing image generated in {} if applicable'.format(args.image_gen_reuse_dir))

    assert (args.multi_image_aggregation and args.n_images > 1) or (not args.multi_image_aggregation and args.n_images == 1)

    os.makedirs(args.output_dir, exist_ok=True)
    if 't2i' in args.query_type:
        img_generation_dir_prefix = '{}/{}'.format(args.img_generation_dir, args.t2i_model_name)
        if args.run_id:
            img_generation_dir = img_generation_dir_prefix + '/run{}'.format(args.run_id)
        os.makedirs(img_generation_dir, exist_ok=True)

    output_file_name = args.output_dir + '{}_{}_{}.json'.format(args.query_type, args.embedding_model_name, datetime.now().strftime("%Y%m%d-%H%M"))
    if 't2i' in args.query_type:
        output_file_name = output_file_name.replace('.json', '_{}_{}imgs.json'.format(args.t2i_model_name, args.n_images))
        if args.n_images > 1:
            output_file_name.replace('.json', '_{}agg.json'.format(args.multi_image_aggregation))
            if args.multi_image_aggregation == 'rank':
                output_file_name.replace('.json', '_rrfk{}.json'.format(args.rrf_k))
    if args.run_id:
        output_file_name = output_file_name.replace('.json', f'_run{args.run_id}.json')

    output_dict = {'query': [], 'num_pos_imgs': [], 'num_neg_imgs': [], 'retrieval_similarity': [], 'aggregated_score': [], 'ranked_imgs': []}
    if args.query_type in ['llm_extracted_text']:
        print('--- Using LLM-based text extraction to improve retrieval performance! ---')
        assert args.query_expansion_model_name is not None
        output_dict["extracted_text"] = []
    elif 't2i' in args.query_type:
        print('--- Using T2I model to improve retrieval performance! ---')
        if 'sd' in args.t2i_model_name:
            from diffusers import StableDiffusion3Pipeline
            # stable diffusion 3
            if args.t2i_model_name == 'sd3':
                model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
            elif args.t2i_model_name == 'sd3-5':
                model_id = "stabilityai/stable-diffusion-3.5-large"
            pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe.set_progress_bar_config(disable=True)
            pipe = pipe.to("cuda:1")
            
        output_dict["generated_image_path"] = []
        output_dict["generation_prompt"] = []

    ks = [1, 5, 10, 20, 30]
    for k in ks:
        output_dict[f'recall@{k}'], output_dict[f'ndcg@{k}'] = [], []
        if args.query_type in ['extracted_text', 'llm_extracted_text']:
            output_dict[f'rephrased_query_recall@{k}'], output_dict[f'rephrased_query_ndcg@{k}'] = [], []
        elif 't2i' in args.query_type:
            output_dict[f't2i_recall@{k}'], output_dict[f't2i_ndcg@{k}'] = [], []

    with torch.no_grad():
        for index in tqdm(range(len(in_data))):
            cur_index_data = in_data[index]

            cur_index_images = [(Image.open(f'{image_dir}/{img}', 'r').convert("RGB"), cur_index_data['images'][img]) for img in cur_index_data['images'] if os.path.exists(f'{image_dir}/{img}')]
            cur_index_images_pos = [x[0] for x in cur_index_images if x[1] == 1]
            cur_index_images_neg = [x[0] for x in cur_index_images if x[1] == 0]

            output_dict['query'].append(cur_index_data['query'])
            output_dict['num_pos_imgs'].append(len(cur_index_images_pos))
            output_dict['num_neg_imgs'].append(len(cur_index_images_neg))
        
            # rerun text-only retrieval baseline
            if args.query_type == 'original_query' or args.run_orig_query:
                if args.embedding_model_name == 'clip':
                    similarity_text_only_query = clip_retrieval_with_text_query([x[0] for x in cur_index_images], cur_index_data['query'])
                elif args.embedding_model_name == 'e5-v':
                    similarity_text_only_query = e5v_retrieval_with_text_query([x[0] for x in cur_index_images], cur_index_data['query'])
                else:
                    raise NotImplementedError
                metrics, raw_scores, merged_scores, top_indices = organize_metrics(similarity_text_only_query, ks, recall_key='recall', ncdg_key='ndcg', multi_image_aggregation=None)
                sorted_imgs, sorted_prob_list, sorted_agg_score_list = get_ranked_images([img for img in cur_index_data['images'] if os.path.exists(f'{image_dir}/{img}')], raw_scores, merged_scores, top_indices)
                for k, v in metrics.items():
                    output_dict[k].append(round(v, 4))
                print(f'Original query retrieval metrics:\n\t', {k: round(v, 4) for k, v in metrics.items()})
                
            if args.query_type == 'llm_extracted_text':
                question = extract_core_info_with_llm(cur_index_data['query'], args.query_expansion_model_name) 
                if args.embedding_model_name == 'clip':
                    similarity_new_text_only_query = clip_retrieval_with_text_query([x[0] for x in cur_index_images], question)
                elif args.embedding_model_name == 'e5-v':
                    similarity_new_text_only_query = e5v_retrieval_with_text_query([x[0] for x in cur_index_images], question)
                else:
                    raise NotImplementedError
                metrics, raw_scores, merged_scores, top_indices = organize_metrics(similarity_new_text_only_query, ks, recall_key='rephrased_query_recall', ncdg_key='rephrased_query_ndcg', multi_image_aggregation=None)
                sorted_imgs, sorted_prob_list, sorted_agg_score_list = get_ranked_images([img for img in cur_index_data['images'] if os.path.exists(f'{image_dir}/{img}')], raw_scores, merged_scores, top_indices)
                print('Rephrased query with LLM:', question)
                print('Rephrased query with LLM, retrieval metrics:\n\t', {k.replace('rephrased_query_', ''): round(v, 4) for k, v in metrics.items()})
                for k, v in metrics.items():
                    output_dict[k].append(round(v, 4))
                output_dict['extracted_text'].append(question)

            elif 't2i' in args.query_type:
                generation_file_name = '{}/{}_gen.png'.format(img_generation_dir, index)

                if args.query_type == 't2i_llm':
                    rephrased_query = extract_core_info_with_llm(cur_index_data['query'], args.query_expansion_model_name)
                else:
                    raise NotImplementedError
                
                if args.t2i_model_name == 'dalle':
                    prompt = 'Generate a small image of the ' + rephrased_query 
                    # Add 'small'
                    # Add 'realistic' / 'natural'
                    t2i_image_url = get_dalle_response(prompt, n=1)
                    save_img_from_url(t2i_image_url, generation_file_name)
                    
                elif args.t2i_model_name in ['image-1-low', 'image-1-medium', 'image-1-high']:
                    quality = args.t2i_model_name.split('-')[-1]
                    prompt = 'Generate a small image of the ' + rephrased_query #

                    file_names = [f'{args.image_gen_reuse_dir}/{index}_gen{i_img}.png' for i_img in range(args.n_images)]
                    if args.image_gen_reuse_dir and all([os.path.exists(x) for x in file_names]):
                        print(f'Found images, skipping generation for: {file_names}')
                        for i_img in range(args.n_images):
                            os.system(f'cp {file_names[i_img]} {generation_file_name.replace(".png", f"{i_img}.png")}')
                    else:
                        try:
                            t2i_images_bytes = get_image1_response(prompt, quality=quality, n=args.n_images)
                            for i_img, t2i_image_bytes in enumerate(t2i_images_bytes):
                                with open(generation_file_name.replace('.png', f'{i_img}.png'), "wb") as f:
                                    f.write(t2i_image_bytes)
                        except Exception as e:
                            print(f'Image generation failed due to: {e}')
                    
                    found_images = [generation_file_name.replace('.png', f'{i_img}.png') for i_img in range(args.n_images)]
                    found_images = [img for img in found_images if os.path.exists(img)]
                    for i_img in range(args.n_images):
                        if generation_file_name.replace('.png', f'{i_img}.png') not in found_images:
                            if len(found_images) == 0:
                                print(f'Image generation failed for all images, creating empty image {i_img}')
                                t2i_image_bytes = np.zeros((1024, 1024, 3), dtype=np.uint8)
                                t2i_image_bytes = Image.fromarray(t2i_image_bytes)
                                t2i_image_bytes.save(generation_file_name.replace('.png', f'{i_img}.png'))
                            else:
                                # copy the first found image
                                print(f'Image generation failed for image {i_img}, copying first found image')
                                os.system(f'cp {found_images[0]} {generation_file_name.replace(".png", f"{i_img}.png")}')
                    
                elif 'sd' in args.t2i_model_name:
                    prompt = 'A small and realistic image of the ' + rephrased_query 
                    image = pipe(prompt).images[0]
                    image.save(generation_file_name)

                else:
                    raise NotImplementedError

                if args.n_images == 1:
                    try:
                        new_query_img = Image.open(generation_file_name).convert("RGB")
                    except:
                        new_query_img = Image.open(generation_file_name.replace('.png', f'{i_img}.png')).convert("RGB")
                    if args.embedding_model_name == 'clip':
                        new_query_img_similarity = clip_retrieval_with_image_query([x[0] for x in cur_index_images], [new_query_img], expected_n_query=1)
                    elif args.embedding_model_name == 'e5-v':
                        new_query_img_similarity = e5v_retrieval_with_image_query([x[0] for x in cur_index_images], [new_query_img], expected_n_query=1)
                    else:
                        raise NotImplementedError
                else:
                    # merge at similarity level or merge at rank level
                    new_query_imgs = []
                    for i_img in range(args.n_images):
                        new_query_img = Image.open(generation_file_name.replace('.png', f'{i_img}.png')).convert("RGB")
                        new_query_imgs.append(new_query_img)
                    if args.embedding_model_name == 'clip':
                        new_query_img_similarity = clip_retrieval_with_image_query([x[0] for x in cur_index_images], new_query_imgs, expected_n_query=len(new_query_imgs))
                    elif args.embedding_model_name == 'e5-v':
                        new_query_img_similarity = e5v_retrieval_with_image_query([x[0] for x in cur_index_images], new_query_imgs, expected_n_query=len(new_query_imgs))
                    else:
                        raise NotImplementedError
                metrics, raw_scores, merged_scores, top_indices = organize_metrics(new_query_img_similarity, ks, recall_key='t2i_recall', ncdg_key='t2i_ndcg', multi_image_aggregation=args.multi_image_aggregation, rrf_k=args.rrf_k)
                sorted_imgs, sorted_prob_list, sorted_agg_score_list = get_ranked_images([img for img in cur_index_data['images'] if os.path.exists(f'{image_dir}/{img}')], raw_scores, merged_scores, top_indices)
                print('Image generation prompt:', prompt)
                print('Image-only retrieval metrics:\n\t', {k.replace('t2i_', ''): round(v, 4) for k, v in metrics.items()})
                print('metrics: ', metrics)
                for k, v in metrics.items():
                    output_dict[k].append(round(v, 4))
                output_dict['generation_prompt'].append(prompt)
                output_dict["generated_image_path"].append(generation_file_name)

            similarity_dict, aggregated_similarity_dict = {}, {}
            for j in range(len(sorted_imgs)):
                similarity_dict[sorted_imgs[j]] = sorted_prob_list[j].tolist()
                aggregated_similarity_dict[sorted_imgs[j]] = sorted_agg_score_list[j].tolist()
            output_dict['retrieval_similarity'].append(similarity_dict)
            output_dict['aggregated_score'].append(aggregated_similarity_dict)
            output_dict['ranked_imgs'].append(sorted_imgs)

    # print averaged metrics
    for k, v in output_dict.items():
        if 'recall' in k or 'ndcg' in k:
            if v:
                print(f'{k}: {round(sum(v) / len(v), 4)}')
    
    # save output
    output_dict = {k: v for k, v in output_dict.items() if v}
    output_list = [dict(zip(output_dict.keys(), t)) for t in zip(*output_dict.values())]
    with open(output_file_name, 'w') as f:
        json.dump(output_list, f, indent=4)
        print(f'Output saved to {output_file_name}')
