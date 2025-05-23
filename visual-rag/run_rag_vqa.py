import os
import json
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
from generation_util import generate_visual_rag_answer_chatgpt, generate_chatgpt_original
import re
import pandas as pd
from prompts import EVAL_PROMPT, EVAL_PROMPT_DEMO, QA_PROMPT_COT


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
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--retrieval_log", type=str, default=None)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--do_evaluate", action="store_true")
    args = parser.parse_args()
    
    print('---Running VQA on top {}---'.format(args.top_k))

    in_data_file = f'{args.data_dir}/visual_rag_v1_anno_cleaned.jsonl'
    image_dir = f'{args.data_dir}/images/'
    in_data = [json.loads(line) for line in open(in_data_file).readlines()]
    
    input_retrieval_data = json.load(open(args.retrieval_log))

    os.makedirs(args.output_dir, exist_ok=True)
    output_file_name = args.output_dir + 'qa_' + args.model_name + args.retrieval_log.split('/')[-1].replace('.json', '_top{}.json'.format(args.top_k))
    if args.run_id is not None:
        output_file_name = output_file_name.replace('.json', '_run{}.json'.format(args.run_id))
    
    try:
        output_dict_list = json.load(open(in_data_file))
    except:
        output_dict_list = [json.loads(line) for line in open(in_data_file).readlines()]
    # reverse record orientation to list orienation
    output_dict = {}
    for k in output_dict_list[0].keys():
        output_dict[k] = []
    for i in range(len(output_dict_list)):
        for k in output_dict_list[i].keys():
            output_dict[k].append(output_dict_list[i][k])
    output_dict['model_answer'], output_dict['topk_imgs'], output_dict['topk_scores'] = [], [], []

    if args.do_evaluate:
        print('--- Running QA evaluation ---')
        output_dict['llm_eval_output'] = [] # output_dict['llm_eval_score'] = []
    
    if os.path.exists(output_file_name):
        tmp_output_dict = pd.read_json(output_file_name).to_dict('list')
        keys = ['model_answer', 'topk_imgs', 'topk_scores']
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

            assert output_dict['question'][index] == cur_index_data['question'] == input_retrieval_data[index]['question']
            topk_img_files = input_retrieval_data[index]['ranked_imgs'][:args.top_k]
            aggregated_score = input_retrieval_data[index]['aggregated_score']
            topk_probs = [aggregated_score[f] for f in topk_img_files]

            qa_prompt = QA_PROMPT_COT
                
            species_name = ' '.join(list(in_data[index]['images'].keys())[0].split('/')[-2].split('_')[-2:])

            model_answer = generate_visual_rag_answer_chatgpt(qa_prompt, cur_index_data['question'] + '\nImages depicting {}:\n'.format(species_name), [f'{image_dir}/{img}' for img in topk_img_files], args.model_name)

            output_dict['model_answer'].append(model_answer)
            output_dict['topk_imgs'].append(topk_img_files)
            output_dict['topk_scores'].append(topk_probs)

            if args.do_evaluate:
                try:
                    model_answer = model_answer.split('Answer: ')[1]
                except IndexError:
                    pass
                llm_eval_output = generate_chatgpt_original(EVAL_PROMPT + EVAL_PROMPT_DEMO + 'Question: ' + cur_index_data['question'] + '\nReference Answer: ' + ', or'.join(cur_index_data['answer']) + '\nStudent Answer: ' + model_answer)
                output_dict['llm_eval_output'].append(llm_eval_output)

                extracted_score = extract_score(llm_eval_output)
                if extracted_score is not None:
                    acc.append(extracted_score)
                else:
                    acc.append(0)

    if args.do_evaluate:
        try:
            print('Final Evaluation Score: ', sum(acc) * 100 / len(acc))
        except:
            pass

    output_dict_entry_orientation = [dict(zip(output_dict.keys(), t)) for t in zip(*output_dict.values())]
    json.dump(output_dict_entry_orientation, open(output_file_name, 'w'), indent=4)
                             
