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

    syllabus= json.load(open(args.syllabus_path, "r"))
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "lectures"), exist_ok=True)
    with open(os.path.join(args.output_dir, "syllabus.json"), 'w') as f:
        json.dump(syllabus, f)
    
    ### get prompts
    lecture_prompt_str = read_prompt(os.path.join(args.prompt_dir, "generate_lecture_note.txt"))
    lecture_prompt = Template(lecture_prompt_str)

    for i, subject in enumerate(TASKS):
        if i % 5 == 0:
            print(f"Generating syllabus for {i}/{len(TASKS)} subjects")
            
        out_path = os.path.join(args.output_dir, "lectures", f"{subject}.jsonl")

        lst_chapters = get_leaf_nodes(syllabus[subject])
        str_subject = format_subject(subject)
        
        ## start again from checkpoint
        if os.path.exists(out_path):
            with open(out_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == len(lst_chapters):
                print(f"found existing datasets for {str_subject}. skip it")
                continue
            
            lst_chapters = lst_chapters[len(lines):]
            print(f"initiate the process from {len(lines)}-th line in {str_subject}")

        for chapter in lst_chapters:
            lecture_note = gen_text(client, 
                                    lecture_prompt.substitute(subject=str_subject,
                                                            chapter=chapter), 
                                    args.model_name)

            dic_out = {"subject": subject, "chapter": chapter, "lecture_note": lecture_note}
            
            with jsonlines.open(out_path, "a") as writer:
                writer.write(dic_out)


if __name__ == "__main__":

    main()