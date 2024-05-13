import os
import pandas as pd
import argparse
import json
import random
import jsonlines

from string import Template
from openai import OpenAI

from eval_mmlu.evaluate_llama_orig import TASKS, format_example, format_subject

OPENAI_API_KEY= #OPENAI API # LG AI from Dasol

def read_prompt(prompt_path):
    with open(prompt_path, "r") as f:
        prompt = f.readlines()    
    return "\n".join(prompt)

def gen_text(client, prompt, model_name):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
    )
    
    response = chat_completion.choices[0].message.content

    return response


def format_examples_with_rationale(data_path):
    lst_few_examples = json.load(open(data_path))
    dic_few_examples = {sub: "" for sub in TASKS}
    options = ["A", "B", "C", "D"]

    for qa in lst_few_examples:
        lst_choices = [options[i] + ". " + qa['choices'][i] for i in range(4)]
        txt = "\nQuestion:\n" + qa['question']
        txt += "\n" + "\n".join(lst_choices) + "\nExplanation:\n"
        if 'explanation' in qa:
            txt += qa['explanation']
        txt += "\nAnswer: " + options[qa['answer']] + "\n\n"
        dic_few_examples[qa['subject']] += txt

    return dic_few_examples


def get_leaf_nodes(json_syllabus):
    lst_leaves = []

    for k, v in json_syllabus.items():
        leaf = k
        if isinstance(v, dict):
            for k1, v1 in v.items():
                if len(v1) == 0:
                    lst_leaves.append(leaf + "\n" + k1)
                    continue
                for k2 in v1:
                    lst_leaves.append(leaf + "\n" + k1 + "\n" + k2)
        else:
            lst_leaves.append(v)
    return lst_leaves


def get_args():

    parser = argparse.ArgumentParser(description='Generate MMLU')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to dataset')
    parser.add_argument('--prompt_dir', type=str, required=True, help='Directory to prompts')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to outputs')
    parser.add_argument('--syllabus_path', type=str, required=True, help='')
    parser.add_argument('--model_name', type=str, default='gpt-4-turbo', help='')

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    client = OpenAI(api_key=OPENAI_API_KEY)

    dic_few_examples = format_examples_with_rationale(os.path.join(args.data_dir, "cot/dev.json"))

    ### get prompts
    mcqa_prompt_str = read_prompt(os.path.join(args.prompt_dir, "generate_mcqa.txt"))
    mcqa_prompt = Template(mcqa_prompt_str)

    os.makedirs(os.path.join(args.output_dir, "mcqas"), exist_ok=True)
    nms_lectures = os.listdir(os.path.join(args.output_dir, "lectures"))
    nms_lectures.sort()
    
    for i, nm_lecture in enumerate(nms_lectures):
        if i % 5 == 0:
            print("current subject: ", nm_lecture)
            print(f"Generating syllabus for {i}/{len(nms_lectures)} subjects")
        
        with open(os.path.join(args.output_dir, "lectures", nm_lecture), 'r') as f:
            lines = f.readlines()
        
        lst_lectures = [json.loads(line) for line in lines]
        
        subject = nm_lecture.replace(".jsonl", "")
        str_subject = format_subject(subject)
        
        out_path = os.path.join(args.output_dir, "mcqas", nm_lecture)
        
        ## TODO: start again from checkpoint
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                mcqa_lines = f.readlines()
            
            lst_saved = [json.loads(line) for line in mcqa_lines]
            
            if len(lst_saved) >= len(lst_lectures):
                print(f"found existing datasets for {str_subject}. skip it")
                continue
            
            lst_lectures = lst_lectures[len(mcqa_lines):]
            print(f"initiate the process from {len(lines)}-th line in {str_subject}")

        for lecture in lst_lectures:
            # lecture_note = gen_text(client, 
            #                         lecture_prompt.substitute(subject=str_subject,
            #                                                 chapter=chapter), 
            #                         args.model_name)

            mcqa_prompt_txt = mcqa_prompt.substitute(subject=str_subject,
                                                chapter=lecture["chapter"],
                                                lecture_note=lecture["lecture_note"],
                                                num_gen_qas="5",
                                                few_examples=dic_few_examples[subject])
            mcqa = gen_text(client, 
                            mcqa_prompt_txt, 
                            args.model_name)

            lecture.update({"mcqa": mcqa})
            
            with jsonlines.open(out_path, "a") as writer:
                writer.write(lecture)


if __name__ == "__main__":

    main()