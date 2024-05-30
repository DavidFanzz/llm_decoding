import re
from evaluate import load
from lm_eval.base import Task
import numpy as np

class CNNDailyMail(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "data/cnndailymail"

    def __init__(self, postprocessed_output_path, sft, bertscore):
        self.postprocessed_output_path = postprocessed_output_path
        self.sft = sft
        self.bertscore = bertscore
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of commonsense_qa can be loaded with old datasets cache
        assert (
            len(dataset) == 11490
        ), "please ensure you have the latest version of commonsense_qa dataset, try deleting its old cache"
        return dataset

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "".join(doc["highlights"])

    def postprocess_generation(self, generation):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        begin_keys = ["sure", "of course", "summary", "summarized", "summarization"]
        completion = generation[0].strip() + "\n"
        if self.sft:
            newline_separator = "\n\n" if "\n\n" in completion else "\n"
            lines = completion.split(newline_separator)
            for k in begin_keys:
                if k in lines[0].lower():
                    next_line = completion.index(newline_separator)
                    completion = completion[next_line:].strip()
                    if "\n\n" in completion:
                        next_line = completion.index('\n')
                        completion = completion[:next_line].strip()
                    break
        else:
            if "\n\n" in completion:
                completion = completion.split("\n\n")[0]
        return completion

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        generations = [self.postprocess_generation(_) for _ in generations]
        rouge = load('rouge')
        results = rouge.compute(predictions=generations, references=references)
        if self.bertscore:
            bertscore = load('bertscore')
            bertscore_results = bertscore.compute(predictions=generations, references=references, lang="en")
            results['bertscore_f1'] = np.mean(bertscore_results['f1'])
            results['bertscore_precision'] = np.mean(bertscore_results['precision'])
            results['bertscore_recall'] = np.mean(bertscore_results['recall'])
        return results

