import json
import sys
import torch
import transformers
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import math
import time
import json
from tqdm import tqdm
from fsd import (
    fsd_vec_decoding,
    topp_decoding,
    magic_decoding,
    fsd_decoding,
    contrastive_decoding3,
    mirostat_decoding,
    dola,
)
from multiprocessing import Process
import pandas as pd
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    add_start_docstrings,
)
import numpy as np


class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list):
        self.token_id_list = token_id_list
        self.stop_tag = None

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for _ in range(len(self.token_id_list)):
            stop_states = [
                np.array_equal(
                    self.token_id_list[_],
                    input_ids[i][-len(self.token_id_list[_]) :].detach().cpu().numpy(),
                )
                for i in range(input_ids.size(0))
            ]
            if self.stop_tag is None:
                self.stop_tag = stop_states
            else:
                self.stop_tag = [
                    self.stop_tag[i] or stop_states[i]
                    for i in range(len(self.stop_tag))
                ]
            if all(self.stop_tag):
                self.stop_tag = None
                return True
        return False


def args_parse():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--infile", type=str, help="the data used for instructing tuning"
    )
    parser.add_argument(
        "--outfile", type=str, help="the data used for instructing tuning"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--decoding_method", type=str, default="topp")
    parser.add_argument("--gpus_per_model", type=int, default=2)
    parser.add_argument(
        "--model_name_or_path", default="decapoda-research/llama-7b-hf", type=str
    )

    parser.add_argument(
        "--student_model_name_or_path",
        default="decapoda-research/llama-7b-hf",
        type=str,
    )

    parser.add_argument("--fsd_alpha", default=0.4, type=float)
    parser.add_argument("--fsd_k", default=5, type=int)
    parser.add_argument("--fsd_n", default=2, type=int)

    parser.add_argument(
        "--fsd_d_alpha", default=0.6, type=float
    )  # 0.6 is not good, maybe lower alpha
    parser.add_argument("--fsd_d_k", default=5, type=int)
    parser.add_argument("--fsd_d_n", default=3, type=int)

    parser.add_argument("--magic_alpha", default=0.4, type=float)
    parser.add_argument("--magic_p", default=0.95, type=float)
    parser.add_argument("--magic_k", default=5, type=int)
    parser.add_argument("--magic_n", default=2, type=int)

    parser.add_argument("--cs_alpha", default=0.6, type=float)
    parser.add_argument("--cs_k", default=5, type=int)

    parser.add_argument(
        "--cd_alpha", default=0.1, type=float
    )  # same notation as original paper
    parser.add_argument("--cd_tt", default=1.0, type=float, help="teacher temperature")
    parser.add_argument("--cd_st", default=0.5, type=float, help="student temperature")
    parser.add_argument("--cd_ignore_prefix", default="yes", type=str)

    parser.add_argument(
        "--cd2_alpha", default=0.1, type=float
    )  # algorithm 1 in the paper cd improves...
    parser.add_argument("--cd2_tt", default=1.0, type=float, help="teacher temperature")
    parser.add_argument("--cd2_st", default=0.5, type=float, help="student temperature")

    parser.add_argument("--cd3_alpha", default=0.1, type=float)
    parser.add_argument("--cd3_beta", default=0.5, type=float)
    parser.add_argument(
        "--cd3_tt", default=1.0, type=float, help="teacher temperature"
    )  # algorithm 2 in the paper cd improves...
    parser.add_argument("--cd3_st", default=1.0, type=float, help="student temperature")

    parser.add_argument(
        "--topp_p", default=0.95, type=float, help="used for topp and topp2"
    )
    parser.add_argument("--typical_p", default=0.95, type=float)
    parser.add_argument("--epsilon_cutoff", default=3e-4, type=float)
    parser.add_argument("--eta_cutoff", default=3e-4, type=float)

    parser.add_argument("--beam_n", default=4, type=int)
    parser.add_argument("--diverse_beam_n", default=4, type=int)
    parser.add_argument("--diverse_beam_groups", default=2, type=int)

    parser.add_argument("--mirostat_tau", default=3.0, type=float)

    parser.add_argument("--dola_early_exit_layers", default=0, type=str)
    parser.add_argument("--dola_mature_layer", default=32, type=int)
    
    parser.add_argument("--begin_gpu", default=0, type=int)
    args = parser.parse_args()
    return args


def out_file(outfile_path, generation_lst):
    with open(outfile_path, "w", encoding="utf-8") as f:
        json.dump(generation_lst, f, indent=4)

    print(f"written to {outfile_path}")


def generate(rank, args):
    visible_devices = [
        str(rank * args.gpus_per_model + i + args.begin_gpu) for i in range(args.gpus_per_model)
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

    if tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token_id = tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if args.decoding_method in ["cd", "cd2", "cd3"]:
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_name_or_path,
            # torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        student_model.eval()

    prompt_lst = []

    with open(args.infile) as f:
        idx = 0
        for line in f.readlines():
            d = json.loads(line.strip())
            d["idx"] = idx
            prompt_lst.append(d)
            idx += 1

    print(f"the total number of prompts: {len(prompt_lst)}")
    prompt_lst = prompt_lst[rank :: args.num_processes]
    print(f"the total number of prompts for rank {rank}: {len(prompt_lst)}")
    if os.path.exists(args.outfile + f"{rank}"):
        generated = pd.read_json(args.outfile + f"{rank}", lines=True)
        remove_list = []
        for _ in range(len(prompt_lst)):
            if prompt_lst[_]["idx"] in generated["idx"].values and _ not in remove_list:
                remove_list.append(_)
        prompt_lst = [
            prompt_lst[_] for _ in range(len(prompt_lst)) if _ not in remove_list
        ]
    print(f"the total number of prompts for rank {rank} to generate: {len(prompt_lst)}")

    # generation_res = []

    s = time.time()
    max_new_tokens = args.max_new_tokens
    for start in tqdm(range(0, len(prompt_lst), args.batch_size), disable=rank != 0):
        stopping_criteria = StoppingCriteriaList()
        if start % 20 == 0 and rank == 0:
            print(f"rank {rank} has generated {start} prompts")
        cur_prompt_lst = prompt_lst[start : start + args.batch_size]
        prompt_text = [f"{x['instructions']}" for x in cur_prompt_lst]
        model_inputs = tokenizer(
            prompt_text, padding=True, add_special_tokens=True, return_tensors="pt"
        )
        input_ids = model_inputs["input_ids"].to(model.device)
        attention_mask = model_inputs["attention_mask"].to(model.device)
        prompt_len = input_ids.size(1)
        args.max_new_tokens = min(max_new_tokens, args.max_length - prompt_len)
        if args.max_new_tokens < 0:
            generation_text = [""] * len(cur_prompt_lst)
            for prompt, generation in zip(cur_prompt_lst, generation_text):
                json_str = json.dumps(
                    {
                        "idx": prompt["idx"],
                        # "instructions": prompt["instructions"],
                        "completion": generation.strip(),
                    }
                )
                with open(args.outfile + f"{rank}", "a", encoding="utf-8") as f:
                    f.write(json_str + "\n")
            continue
        if args.decoding_method == "fsd":
            outputs = fsd_vec_decoding(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                k=args.fsd_k,
                alpha=args.fsd_alpha,
                max_new_tokens=args.max_new_tokens,
                n=args.fsd_n,
                early_stop=args.early_stop,
                stopping_criteria=stopping_criteria,
            )

        if args.decoding_method == "fsd-d":
            outputs = fsd_decoding(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                k=args.fsd_d_k,
                alpha=args.fsd_d_alpha,
                max_new_tokens=args.max_new_tokens,
                n=args.fsd_d_n,
                early_stop=args.early_stop,
                stopping_criteria=stopping_criteria,
            )
        if args.decoding_method == "topp":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                top_k=0,
                top_p=args.topp_p,
                do_sample=True,
                stopping_criteria=stopping_criteria,
            )
        if args.decoding_method == "topp2":
            outputs = topp_decoding(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                top_p=args.topp_p,
                max_new_tokens=args.max_new_tokens,
                early_stop=args.early_stop,
            )
        if args.decoding_method == "topk":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                top_k=args.topk_k,
                do_sample=True,
                stopping_criteria=stopping_criteria,
            )
        if args.decoding_method == "greedy":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                stopping_criteria=stopping_criteria,
            )
        if args.decoding_method == "temperature":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature_t,
                top_k=0,
                stopping_criteria=stopping_criteria,
            )
        if args.decoding_method == "eta":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                eta_cutoff=args.eta_cutoff,
                top_k=0,
                do_sample=True,
                stopping_criteria=stopping_criteria,
            )
            # https://aclanthology.org/2022.findings-emnlp.249/

        if args.decoding_method == "typical":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                typical_p=args.typical_p,
                top_k=0,
                do_sample=True,
                stopping_criteria=stopping_criteria,
            )
            # https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00536/114593/Locally-Typical-Sampling
            # outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens, typical_p=args.typical_p,top_k=0,do_sample=True)

        if args.decoding_method == "cs":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                penalty_alpha=args.cs_alpha,
                top_k=args.cs_k,
                stopping_criteria=stopping_criteria,
            )
            # https://openreview.net/pdf?id=V88BafmH9Pj
            # outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens, penalty_alpha=args.cs_alpha, top_k=args.cs_k)

        if args.decoding_method == "cd3":
            outputs = contrastive_decoding3(
                model,
                student_model,
                teacher_t=args.cd3_tt,
                student_t=args.cd3_st,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                early_stop=True,
                alpha=args.cd3_alpha,
                beta=args.cd3_beta,
                stopping_criteria=stopping_criteria,
            )

        if args.decoding_method == "beam":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.beam_n,
                stopping_criteria=stopping_criteria,
                do_sample=False,
            )

        if args.decoding_method == "diverse_beam":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.diverse_beam_n,
                num_beam_groups=args.diverse_beam_groups,
                diversity_penalty=1.0,
                stopping_criteria=stopping_criteria,
                do_sample=False,
            )

        if args.decoding_method == "mirostat":
            outputs = mirostat_decoding(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                tau=args.mirostat_tau,
                max_new_tokens=args.max_new_tokens,
                early_stop=args.early_stop,
                stopping_criteria=stopping_criteria,
            )
        if args.decoding_method == "dola":
            early_exit_layers = [int(x) for x in args.dola_early_exit_layers.split(",")]
            outputs = dola(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.2,
                mature_layer=args.dola_mature_layer,
                base_layer=None,
                candidate_premature_layers=early_exit_layers,
                relative_top=0.1,
                eos_token_id=None,
                early_stop=False,
                stopping_criteria=stopping_criteria,
            )

        generation_text = tokenizer.batch_decode(
            outputs[:, prompt_len:],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )
        for prompt, generation in zip(cur_prompt_lst, generation_text):
            json_str = json.dumps(
                {
                    "idx": prompt["idx"],
                    # "instructions": prompt["instructions"],
                    "completion": generation.strip(),
                }
            )
            with open(args.outfile + f"{rank}", "a", encoding="utf-8") as f:
                f.write(json_str + "\n")

    t = time.time()
    print("time used: ", t - s)


if __name__ == "__main__":
    args = args_parse()
    args.early_stop = True
    print(args)
    assert args.world_size % args.gpus_per_model == 0
    args.num_processes = args.world_size // args.gpus_per_model
    if os.path.exists(args.outfile):
        try:
            all_ret = pd.read_json(args.outfile, lines=True)
            all_ret = all_ret.drop_duplicates(subset=["idx"], keep="first")
            all_ret.reset_index(drop=True, inplace=True)
            all_ret = all_ret[all_ret["completion"] != ""]
            all_ret.reset_index(drop=True, inplace=True)
            input_file = pd.read_json(args.infile, lines=True)
            if len(all_ret) == len(input_file):
                print(f"{args.outfile} already generated.")
                sys.exit(0)
            else:
                print("some prompts are not generated, regenerate them.")
                for _ in range(args.num_processes):
                    if os.path.exists(args.outfile + f"{_}"):
                        os.remove(args.outfile + f"{_}")
                for _ in range(len(all_ret)):
                    to_write_id = all_ret.iloc[_]["idx"] % args.num_processes
                    with open(
                        args.outfile + f"{to_write_id}", "a", encoding="utf-8"
                    ) as f:
                        json_str = json.dumps(
                            {
                                "idx": int(all_ret.iloc[_]["idx"]),
                                # "instructions": all_ret.iloc[_]["instructions"],
                                "completion": all_ret.iloc[_]["completion"],
                            }
                        )
                        f.write(json_str + "\n")
        except:
            print("bad output file")
            sys.exit(0)

    process_list = []
    for i in range(args.num_processes):
        p = Process(target=generate, args=(i, args))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    all_ret = pd.DataFrame()
    for rank in range(args.num_processes):
        with open(args.outfile + f"{rank}", "r", encoding="utf-8") as f:
            all_ret = pd.concat(
                [all_ret, pd.read_json(f, lines=True)], ignore_index=True
            )
    all_ret.sort_values(by="idx", inplace=True)
    all_ret.to_json(args.outfile, orient="records", lines=True, force_ascii=False)
    for rank in range(args.num_processes):
        os.remove(args.outfile + f"{rank}")
