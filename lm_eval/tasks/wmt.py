import re

from evaluate import load

from lm_eval.base import Task


class WMT(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "data/wmt"

    def __init__(self, postprocessed_output_path, sft, comet):
        self.postprocessed_output_path = postprocessed_output_path
        self.sft = sft
        self.comet = comet
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of commonsense_qa can be loaded with old datasets cache
        assert (
            len(dataset) == 7933
        ), "please ensure you have the latest version of commonsense_qa dataset, try deleting its old cache"
        return dataset

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "".join(doc["tgr"])


    def postprocess_generation(self, generation):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        completion = generation[0].strip()
        if self.sft:
            print("Post processing to be updated")
            pass
        else:
            if "\n\n" in completion:
                completion = completion.split("\n\n")[0].strip()
            elif "\n" in completion:
                completion = completion.split("\n")[0].strip()
        return completion

    def get_sources(self):
        """Returns the possible sources for the task."""
        return [_['src'] for _ in self.dataset["test"]]
    
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        
        generations = [self.postprocess_generation(_) for _ in generations]
        bleu_metric = load("sacrebleu")
        results = {}
        tasks = {
            "de2en": (0, 1984),
            "en2de": (1984, 4021),
            "en2zh": (4021, 6058),
            "zh2en": (6058, None),
        }

        for task, (start, end) in tasks.items():
            references_ = references[start:end]
            generations_ = generations[start:end]
            single_results = bleu_metric.compute(references=references_, predictions=generations_)
            results[task] = single_results['score']

        if self.comet:
            comet_metric = load("comet")
            source = self.get_sources()
            comet_results = comet_metric.compute(references=references, predictions=generations, sources=source)
            results['comet'] = comet_results['mean_score']

        return results
