import inspect
from pprint import pprint
from . import (mbpp, human_eval, commonsense_qa, strategy_qa, math, gsm8k, wmt, xsum, cnndailymail)

TASK_REGISTRY = {
    "mbpp": mbpp.MBPP,
    "human_eval": human_eval.HumanEval,
    "commonsense_qa": commonsense_qa.CommonsenseQA,
    "strategy_qa": strategy_qa.StrategyQA,
    'math': math.MATH,
    'gsm8k': gsm8k.GSM8K,
    'wmt': wmt.WMT,
    'xsum': xsum.XSUM,
    'cnndailymail': cnndailymail.CNNDailyMail,
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, args=None):
    try:
        kwargs = {}
        if "postprocessed_output_path" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["postprocessed_output_path"] = args.postprocessed_output_path
        if "sft" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["sft"] = args.sft
        if "comet" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["comet"] = args.comet
        if "bertscore" in inspect.signature(TASK_REGISTRY[task_name]).parameters:
            kwargs["bertscore"] = args.bertscore
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
