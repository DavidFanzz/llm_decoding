"""Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""

import re

from evaluate import load

from lm_eval.base import Task

import pandas as pd

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import subprocess
import os
from functools import partial

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""

def run_pylint(i, generation):
    file_name = f"file_{i}_tmp.py"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(generation[i][0])
        command = f"pylint {file_name} --errors-only"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    # if result.stdout == "":
    os.remove(file_name)
    return result


class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "data/mbpp"

    def __init__(self, postprocessed_output_path, sft):
        self.postprocessed_output_path = postprocessed_output_path
        self.sft = sft
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])

    def postprocess_generation(self, generation):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        completion = generation[0]
        completion = completion.replace("\r", "")  
        if self.sft:
            import_pattern = r'(^|\n| )import .*'
            main_func_pattern = r'def.*?\n[^\n\s#]'
            prefix = ""
            import_pattern_result = re.findall(import_pattern, completion)
            if import_pattern_result:
                for line in import_pattern_result:
                    prefix = line.strip() + "\n" + prefix
    
            main_func_pattern_result = re.search(main_func_pattern, completion, re.DOTALL)
            if main_func_pattern_result:
                completion = main_func_pattern_result.group(0)[:-1]
            elif "def " in completion:
                completion = "def " +completion.split("def ")[-1]
            else:
                print(generation[0])
                print("=" * 50 + "\n")
            return [prefix + completion.strip() + "\n"]
        else:
            if "[DONE]" in completion:
                next_line = completion.index("[DONE]")
                completion = completion[:next_line].strip()        
            if '```python' in completion: 
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('```')
                    completion = completion[:next_line].strip()
                except:
                    print(completion)
                    print("=" * 50 + "\n")
            if "__name__" in completion:
                next_line = completion.index('__name__')
                completion = completion[:next_line].strip()[:-2]
                
            if "# Example usage" in completion:
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            return [completion]

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        generations = [self.postprocess_generation(generations[_]) for _ in range(len(generations))]
        if self.postprocessed_output_path:
            postprocessed_output = pd.DataFrame()
            postprocessed_output['results'] = generations
            postprocessed_output.to_json(self.postprocessed_output_path, orient='records', lines=True)

        num_processes = 8
        run_pylint_with_generation = partial(run_pylint, generation=generations)
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            code_results = list(tqdm(executor.map(run_pylint_with_generation, range(len(generations))), total=len(generations)))
        cnt = 0
        for _ in range(len(code_results)):
            if code_results[_].stdout == "":
                cnt += 1
        
        code_metric = load("code_eval")
        results, _ = code_metric.compute(
            references=references,
            predictions=generations,
        )
        results['code_rate'] = cnt / len(generations)
        return results
