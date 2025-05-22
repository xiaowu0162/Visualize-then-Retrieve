import os
import json
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
from generation_util import generate_visual_rag_answer_chatgpt, generate_multi_entity_visual_rag_answer_chatgpt, generate_chatgpt_original, generate_response_phi3v, generate_response_qwen
import re
import pandas as pd
from prompts import EVAL_PROMPT, EVAL_PROMPT_DEMO, ZEROSHOT_QA_PROMPT, QA_PROMPT_MULTI_ENTITY_COT_V1
from PIL import Image 
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
now = datetime.now()

def find_datetime_str(file_name):
    datetime_pattern = r'(\d{8}-\d{4})'

    # Check if the file name matches the pattern
    match = re.search(datetime_pattern, file_name)
    if match:
        # Extract the datetime string and append it to the list
        return match.group(1)

def extract_score(text):
    """
    Extracts the score from a model-generated string.
    Assumes the score appears after "Score:" and may be followed by a space, |, or parenthetical.
    Returns the score as a float, or None if not found.
    """
    match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', text)
    if match:
        return float(match.group(1))
    return None

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-if", "--input_file", type=str, default='/local/elaine1wan/vis-retrieve/visual-mrag/visual-rag/retrieve_outputs/t2i_llm_clip_20250506-2152_image-1-low_2imgs_test_run20250506-2152.json')
    # parser.add_argument("-m", "--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("-b", "--baseline_type", type=str, default="oracle")
    parser.add_argument("-d", "--data_dir", type=str, default='/local/elaine1wan/vis-retrieve/visual-mrag/visual-rag/')
    parser.add_argument("-o", "--output_dir", type=str, default='/local/elaine1wan/vis-retrieve/visual-mrag/visual-rag/qa_outputs/')
    parser.add_argument("--model_name_short", type=str, choices=['gpt4o', 'gpt4omini', 'gpt4.1', 'phi3v', 'qwen'])
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--do_evaluate", action="store_true")
    args = parser.parse_args()
    in_data_file = f'{args.data_dir}/multi_entity_visual_rag_final_v1.jsonl'
    image_dir = f'{args.data_dir}/images_v1_final/'
    in_data = [json.loads(line) for line in open(in_data_file).readlines()]
    if args.test:
        in_data = in_data[:50]
    
    if args.skip_generation:
        print('Using generated model answers for direct evaluation...')
        assert args.do_evaluate, 'Need to be running at least one from QA and evaluation'
        output_file_name = args.input_file
        output_dict = pd.read_csv(args.input_file).to_dict('list')
    elif args.baseline_type == 'oracle':
        output_file_name = args.output_dir + 'qa_baseline_{}_oracle_{}.csv'.format(args.model_name_short,datetime.now().strftime("%Y%m%d-%H%M"))
        output_dict = {'question': [], 'answer': [], 'oracle_image': [], 'model_answer': []}
    elif args.baseline_type == 'zero_shot':
        output_file_name = args.output_dir + 'qa_baseline_{}_zero_shot_{}.csv'.format(args.model_name_short,datetime.now().strftime("%Y%m%d-%H%M"))
        output_dict = {'question': [], 'answer': [], 'model_answer': []}
    elif args.baseline_type == 'rand_irrelev':
        output_file_name = args.output_dir + 'qa_baseline_{}_random_irrelevant_{}.csv'.format(args.model_name_short,datetime.now().strftime("%Y%m%d-%H%M"))
        output_dict = {'question': [], 'answer': [], 'irrelevant_image': [], 'model_answer': []}
    else:
        raise NotImplementedError
    
    if args.test:
        output_file_name = output_file_name.replace('.csv', '_test.csv')

    if args.do_evaluate:
        print('--- Running QA evaluation ---')
        output_dict['llm_eval_output'] = [] # output_dict['llm_eval_score'] = []

    if os.path.exists(output_file_name):
        tmp_output_dict = pd.read_csv(output_file_name).to_dict('list')
        keys = ['model_answer']
        if args.do_evaluate:
            keys.append('llm_eval_output')
        for k in keys:
            output_dict[k] = tmp_output_dict[k]
        start = len(output_dict['model_answer'])
    else:
        tmp_output_dict = {}
        start = 0

    
    with torch.no_grad():
        acc = []
        for index in tqdm(range(start, len(in_data))):
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            if start != 0:
                print('---Resuming from Index {}---'.format(start))
            cur_index_data = in_data[index]

            if args.skip_generation:
                assert output_dict['question'][index] == cur_index_data['question']
                model_answer = output_dict['model_answer'][index]
            else:
                species_name_1 = ' '.join(list(cur_index_data['images']['entity_1'].keys())[0].split('/')[-2].split('_')[-2:])
                species_name_2 = ' '.join(list(cur_index_data['images']['entity_2'].keys())[0].split('/')[-2].split('_')[-2:])
                if args.baseline_type == 'oracle':
                    used_images_1 = [img for img in cur_index_data['images']['entity_1'].keys() if cur_index_data['images']['entity_1'][img] == 1]
                    used_images_2 = [img for img in cur_index_data['images']['entity_2'].keys() if cur_index_data['images']['entity_2'][img] == 1]
                    # assert len(used_images) >= 1, 'No positive oracle image provided!'
                    if len(used_images_1) < 1 or len(used_images_2) < 1:
                        print('No positive oracle image provided for index {}!'.format(index))
                    else:
                        used_images_1 = random.sample(used_images_1, 1)
                        used_images_2 = random.sample(used_images_2, 1)
                    output_dict['oracle_image'].append(used_images_1 + used_images_2)
                    qa_prompt = QA_PROMPT_MULTI_ENTITY_COT_V1
                elif args.baseline_type == 'rand_irrelev':
                    used_images_1 = random.sample([img for img in cur_index_data['images']['entity_1'].keys() if cur_index_data['images']['entity_1'][img] == 0], 1)
                    used_images_2 = random.sample([img for img in cur_index_data['images']['entity_2'].keys() if cur_index_data['images']['entity_2'][img] == 0], 1)
                    output_dict['irrelevant_image'].append(used_images_1 + used_images_2)
                    qa_prompt = QA_PROMPT_MULTI_ENTITY_COT_V1
                elif args.baseline_type == 'zero_shot':
                    qa_prompt = ZEROSHOT_QA_PROMPT
                else:
                    qa_prompt = QA_PROMPT_MULTI_ENTITY_COT_V1
                    
                output_dict['question'].append(cur_index_data['question'])
                output_dict['answer'].append(cur_index_data['answer'])
                # species_name = ' '.join(list(in_data[index]['images'].keys())[0].split('/')[-2].split('_')[-2:])
                if args.baseline_type == 'zero_shot':
                    if 'phi3v' in args.model_name_short or 'qwen' in args.model_name_short:
                        qa_prompt = qa_prompt + ' ' + cur_index_data['question']
                        model_answer = generate_function(qa_prompt, [], processor, model)
                    else:
                        model_answer = generate_visual_rag_answer_chatgpt(qa_prompt, cur_index_data['question'], [], model=args.model_name_short)
                else:
                    if 'phi3v' in args.model_name_short or 'qwen' in args.model_name_short:
                        qa_prompt = qa_prompt + ' ' + cur_index_data['question']
                        model_answer = generate_function(qa_prompt, [f'{image_dir}/{img}' for img in used_images], processor, model)
                    else:
                        # model_answer = generate_multi_entity_visual_rag_answer_chatgpt(qa_prompt, cur_index_data['question'], 
                        #                                                                [f'{image_dir}/{img}' for img in used_images], 
                        #                                                                model=args.model_name_short)
                        model_answer = generate_multi_entity_visual_rag_answer_chatgpt(qa_prompt, cur_index_data['question'], 
                                                                                       species_name_1, [f'{image_dir}/{img}' for img in used_images_1], 
                                                                                       species_name_2, [f'{image_dir}/{img}' for img in used_images_2], 
                                                                                       model=args.model_name_short)
                
                output_dict['model_answer'].append(model_answer)

            if args.do_evaluate:
                # print(EVAL_PROMPT + EVAL_PROMPT_DEMO + '\nQuestion: ' + cur_index_data['question'] + '\nStudent Answer: ' + model_answer)
                # exit()
                try:
                    model_answer = model_answer.split('Answer: ')[1]
                except IndexError:
                    pass
                llm_eval_output = generate_chatgpt_original(EVAL_PROMPT + EVAL_PROMPT_DEMO + 'Question: ' + cur_index_data['question'] + '\nReference Answer: ' + ', or'.join(cur_index_data['answer']) + '\nStudent Answer: ' + model_answer, model='gpt4o')
                output_dict['llm_eval_output'].append(llm_eval_output)

                extract_scored_score = extract_score(llm_eval_output)
                if extract_scored_score is not None:
                    acc.append(extract_scored_score)
                else:
                    acc.append(0)

            torch.cuda.empty_cache()

    if args.do_evaluate:
        try:
            print('Final Evaluation Score: ', sum(acc) * 100 / len(acc))
        except:
            pass

    if args.test:
        for k in output_dict.keys():
            output_dict[k] = output_dict[k][:50]
    output_df = pd.DataFrame.from_dict(output_dict)
    output_df.to_csv(output_file_name, index=False)