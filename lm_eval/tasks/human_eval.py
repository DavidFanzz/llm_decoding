"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
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
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""

def run_pylint(i, generation):
    file_name = f"file_{i}_tmp.py_"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(generation[i][0])
        command = f"pylint {file_name} --errors-only"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    # if result.stdout == "":
    os.remove(file_name)
    return result

def extract_between_chars(text, char1, char2):
    pattern = f"{char1}(.*?){char2}"
    result = re.search(pattern, text, re.DOTALL)
    if result:
        return result.group(1)
    else:
        return None
    
class HumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "data/human_eval"
    
    def __init__(self, postprocessed_output_path, sft):
        self.postprocessed_output_path = postprocessed_output_path
        self.sft = sft

        super().__init__(
            requires_execution=True,
        )
        # self.strip_prompt = strip_prompt
    
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of commonsense_qa can be loaded with old datasets cache
        assert (
            len(dataset) == 164
        ), "please ensure you have the latest version of commonsense_qa dataset, try deleting its old cache"
        return dataset
    

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point


    def postprocess_generation(self, generation, prefix):
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
            return [prefix.split('def ')[0] + completion.strip() + "\n"]
        else:
            if '```python' in completion: 
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('```')
                    completion = completion[:next_line].strip()
                except:
                    print(generation[0])
                    print("=" * 50 + "\n")

            if "__name__" in completion:
                next_line = completion.index('__name__')
                completion = completion[:next_line].strip()[:-2]
                
            if "# Example usage" in completion:
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()

        return [prefix + "    " + completion.strip() + "\n"]

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        generations = [self.postprocess_generation(generations[_], self.dataset['test']['prompt'][_]) for _ in range(len(generations))]
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
