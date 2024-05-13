import argparse
import json
import os
import random
from pprint import pprint
from tqdm import tqdm
from typing import List, Union
from string import Template

import jsonlines
import torch
from vllm import LLM, SamplingParams

PROMPT_PTH = {
    "prometheus": "prompt/rm_noref.txt",
}

def get_prompt(src: dict, prompt_type: str):
    with open(src[prompt_type], "r") as f:
        prompt_raw = f.readlines()
    return Template(" ".join(prompt_raw))

def load_model(model_name, num_gpus=8):
    model = LLM(model_name, tensor_parallel_size=num_gpus)
    return model

def get_lm_output(lm, lm_input: Union[str, List[str]]):
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=0.95, 
        max_tokens=256, 
        repetition_penalty=1.03,
        length_penalty=1.0,
        top_k=50,
    )
    outputs = lm.generate(lm_input, sampling_params)
    lm_outputs = [op.outputs[0].text for op in outputs]
    return lm_outputs

def get_preference_prometheus(reward_model, instruction: Union[str, List[str]]): 
    """Generate scores of candidate responses"""
    results = []
    outputs = get_lm_output(lm=reward_model, lm_input=instruction)
    for idx, output in enumerate(outputs):
        result = {}
        while True:
            max_iter, cnt_iter = 5, 0
            try:
                score = int(output.split("[RESULT]")[1].strip())
                reason = output.split("[RESULT]")[0].strip()
                break
            except Exception as e:  # Parsing error
                print(e)
                cnt_iter += 1
                if cnt_iter > max_iter:
                    print("Max request exceeded")
                    score = 0
                    reason = "None"
                    break
                output = get_lm_output(lm=reward_model, lm_input=instruction[idx])[0]
        result["score"] = score
        result["rationale"] = reason
        results.append(result)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_pth", type=str)
    parser.add_argument("--save_pth_partial", type=str)
    parser.add_argument("--rm_model_name", type=str, default="kaist-ai/prometheus-13b-v1.0")
    parser.add_argument("--rm_eval_criteria", type=str)
    parser.add_argument("--rm_eval_aspect", type=str)
    parser.add_argument("--qplanner_pth", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    cfg = parser.parse_args()
    
    print(f"### BATCH SIZE: {cfg.batch_size}")
    print(f"### ASPECT: {cfg.rm_eval_aspect}")

    # Load reward model and criteria
    num_device = torch.cuda.device_count()
    num_device = num_device if num_device < 8 else 8
    rm = load_model(model_name=cfg.rm_model_name, num_gpus=num_device)
    rm_prompt = get_prompt(PROMPT_PTH, "prometheus")
    
    with open(cfg.rm_eval_criteria, "r") as json_file:
        eval_criteria = json.load(json_file)
    ASPECT = cfg.rm_eval_aspect

    # Load original dataset
    with jsonlines.open(cfg.dset_pth) as f:
        dataset = [line for line in f.iter()]
    print(f"# of full dataset: {len(dataset)}")
    save_pth = f"{cfg.save_pth_partial}_{ASPECT}.jsonl"
    if not os.path.exists(save_pth):
        with open(save_pth, "a+", encoding="utf-8") as f:
            pass
    with jsonlines.open(save_pth) as f:
        cache = [line for line in f.iter()]
    cache = {item["question"]: None for item in cache}
    dataset = [item for item in dataset if item["question"] not in cache]
    print(f"# of dataset already generated: {len(cache)}")
    print(f"# of dataset to be generated: {len(dataset)}")

    def zip_batch(items, bsz=1):
        return list(zip(*[items[i::bsz] for i in range(bsz)]))

    iteration = zip_batch(dataset, bsz=cfg.batch_size)
    for batch in tqdm(iteration, desc="Sample", total=len(iteration)):
        batch_input = []

        # Compose batch input
        for idx, each in enumerate(batch):
            ques, inst = each["question"], each["instruction"]
            for jdx, cand in enumerate(each["candidates"]):
                resp = cand["response"]
                rm_instruction = " ".join([ques, inst])
                rm_input = rm_prompt.substitute(
                    instruction=rm_instruction,
                    response=resp,
                    description=eval_criteria[ASPECT]["description"],
                    score_1 = eval_criteria[ASPECT]["score_1"],
                    score_2 = eval_criteria[ASPECT]["score_2"],
                    score_3 = eval_criteria[ASPECT]["score_3"],
                    score_4 = eval_criteria[ASPECT]["score_4"],
                    score_5 = eval_criteria[ASPECT]["score_5"],
                )
                batch_input.append(rm_input)
            
        # Get preference
        if cfg.debug:
            print(f"... Start generating {len(batch_input)} evaluations.")
        preference = get_preference_prometheus(reward_model=rm, instruction=batch_input)
        if cfg.debug:
            print(f"... Get {len(preference)} results.")

        # Save
        index = 0
        for each in batch:
            result = { 
                "question": each["question"],
                "instruction": each["instruction"],
                f"eval_{ASPECT}": [],
            }
            safe = True
            for cand in each["candidates"]:
                detail = {"subtree": cand["subtree"]}
                detail["score"] = preference[index]["score"]
                detail["rationale"] = preference[index]["rationale"]
                index += 1
                if (detail["score"] == 0) or (detail["rationale"] is None):
                    safe = False
                    break
                result[f"eval_{ASPECT}"].append(detail)

            if safe:
                with open(save_pth, "a+", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False) 
                    f.write("\n") 
                

"""각 aspect별 저장 형태

question: ""
instruction: ""
eval_{aspect} : [ 
    subtree 1: {score, rationale}
    subtree 2: {score, rationale}
    subtree 3: {score, rationale}
]
"""