import os
import json
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
from generation_util import generate_visual_rag_answer_chatgpt, generate_chatgpt_original
import re
import pandas as pd
from prompts import EVAL_PROMPT, EVAL_PROMPT_DEMO, ZEROSHOT_QA_PROMPT, QA_PROMPT_COT
import random

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
    parser.add_argument("-if", "--input_file", type=str)
    parser.add_argument("-b", "--baseline_type", type=str)
    parser.add_argument("-d", "--data_dir", type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("-m", "--model_name_short", type=str, choices=['gpt4o', 'gpt4omini', 'gpt4.1'])
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--do_evaluate", action="store_true")
    args = parser.parse_args()
    in_data_file = f'{args.data_dir}/visual_rag_v1_anno_cleaned.jsonl'
    image_dir = f'{args.data_dir}/images/'
    in_data = [json.loads(line) for line in open(in_data_file).readlines()]
    
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
            if start != 0:
                print('---Resuming from Index {}---'.format(start))
            cur_index_data = in_data[index]

            if args.skip_generation:
                assert output_dict['question'][index] == cur_index_data['question']
                model_answer = output_dict['model_answer'][index]
            else:
                species_name = ' '.join(list(cur_index_data['images'].keys())[0].split('/')[-2].split('_')[-2:])
                if args.baseline_type == 'oracle':
                    used_images = [img for img in cur_index_data['images'].keys() if cur_index_data['images'][img] == 1]
                    # assert len(used_images) >= 1, 'No positive oracle image provided!'
                    if len(used_images) < 1:
                        print('No positive oracle image provided for index {}!'.format(index))
                    else:
                        used_images = random.sample(used_images, 1)
                    output_dict['oracle_image'].append(used_images)
                    qa_prompt = QA_PROMPT_COT
                elif args.baseline_type == 'rand_irrelev':
                    used_images = random.sample([img for img in cur_index_data['images'].keys() if cur_index_data['images'][img] == 0], 1)
                    output_dict['irrelevant_image'].append(used_images)
                    qa_prompt = QA_PROMPT_COT
                elif args.baseline_type == 'zero_shot':
                    qa_prompt = ZEROSHOT_QA_PROMPT
                else:
                    qa_prompt = QA_PROMPT_COT
                    
                output_dict['question'].append(cur_index_data['question'])
                output_dict['answer'].append(cur_index_data['answer'])
                # species_name = ' '.join(list(in_data[index]['images'].keys())[0].split('/')[-2].split('_')[-2:])
                if args.baseline_type == 'zero_shot':
                    model_answer = generate_visual_rag_answer_chatgpt(qa_prompt, cur_index_data['question'], [], model=args.model_name_short)
                else:
                    model_answer = generate_visual_rag_answer_chatgpt(qa_prompt, cur_index_data['question'], 
                                                                      [f'{image_dir}/{img}' for img in used_images], 
                                                                      model=args.model_name_short)
                
                output_dict['model_answer'].append(model_answer)

            if args.do_evaluate:
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

    os.makedirs(args.output_dir, exist_ok=True)
    output_df = pd.DataFrame.from_dict(output_dict)
    output_df.to_csv(output_file_name, index=False)