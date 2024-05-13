import os
import pandas as pd
import argparse
import json
import random

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


def gen_syllabus(client, prompt, model_name):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
        response_format={"type": "json_object"},
    )
    
    out = chat_completion.choices[0].message.content
    json_response = json.loads(out)

    return json_response


def get_args():

    parser = argparse.ArgumentParser(description='Generate MMLU')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory to dataset')
    parser.add_argument('--prompt_dir', type=str, required=True, help='Directory to prompts')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to outputs')
    parser.add_argument('--few_shot_split', type=str, default='dev', help='')
    parser.add_argument('--model_name', type=str, default='gpt-4-turbo', help='')

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    client = OpenAI(api_key=OPENAI_API_KEY)

    ### get prompts
    desc_prompt_str = read_prompt(os.path.join(args.prompt_dir, "subject_description.txt"))
    desc_prompt = Template(desc_prompt_str)

    syl_prompt_str = read_prompt(os.path.join(args.prompt_dir, "generate_syllabus.txt"))
    syl_prompt = Template(syl_prompt_str)

    all_syllabus = {}
    for i, subject in enumerate(TASKS):
        str_subject = format_subject(subject)

        if i % 5 == 0:
            print(f"Generating syllabus for {i}/{len(TASKS)} subjects")

        ### get few-shot examples
        data_name = f"{subject}_dev.csv"
        data_path = os.path.join(args.data_dir, 'dev', data_name)
        df = pd.read_csv(data_path, header=None)

        few_examples = ""
        for i in range(5):
            few_examples += "Question: \n" + format_example(df, i) + "\n"

        ## 1. describe the subject
        desc = gen_text(client, 
                    desc_prompt.substitute(subject=str_subject, few_examples=few_examples), 
                    args.model_name
        )

        ## 2. generate syllabus in json format
        syllabus = gen_syllabus(client, 
                    syl_prompt.substitute(subject=str_subject, 
                                        description=desc), 
                    args.model_name
        )
        all_syllabus[subject] = syllabus
        
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "syllabus.json"), "w") as f:
        json.dump(all_syllabus, f, indent=4)

if __name__ == "__main__":

    main()