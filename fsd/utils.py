import torch
import time
import math
import random
import numpy as np
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    LogitsProcessorList,
)
from torch.nn import functional as F
from .ngram_model.fsd import NGram
from .ngram_model.fsd_vec import HiddenSoftNGram
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    add_start_docstrings,
)


def topk_logits_filter(scores, k):
    # Safety check
    top_k = min(max(k, 1), scores.size(-1))
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, -float("Inf"))
    return scores


def topp_logits_filter(scores, p):
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    # Keep at least 1 token
    sorted_indices_to_remove[..., -1:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    scores = scores.masked_fill(indices_to_remove, -float("Inf"))
    return scores


def estimate_s(prob):
    num = 0
    den = 0
    epsilon = 1e-9
    for i in range(100):
        b = max(prob[i] / (prob[i + 1] + epsilon), 1)
        t = (i + 2) / (i + 1)
        num += math.log(b) * math.log(t)
        den += math.log(t) ** 2
    return num / den


def compute_k(n, s, tau):
    try:
        eps = s - 1
        k = ((eps * (2 ** (tau))) / (1 - n ** (-eps))) ** (1 / s)
        k = round(k)
        return k
    except KeyboardInterrupt:
        exit()
    except:
        return n - 1


@torch.no_grad()
def fsd_decoding(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    k,
    alpha,
    max_new_tokens,
    temperature=1.0,
    n=3,
    beta=0.9,
    sw_coeff=0.0,
    stop_words_ids=[],
    eos_token_id=None,
    early_stop=False,
    stopping_criteria=None,
):
    """
    - k: top-k candidate words are selected, default 3
    - alpha: (1-alpha)p_lm -(alpha)*penalty
    - max_length: decoding max_length-prompt_length steps
    - n: the order of n-gram models
    - beta: the smoothness of n-gram models, default 0.9
    - sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1), default 0.
    - stop_words=[]: the list of stopwords. If you use GPT-2, you at least need to add two special tokens ('Ċ' and 'ĊĊ') to avoid grammars errors.
    """

    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}

    prompt_len = torch.sum(attention_mask, dim=1)
    model_kwargs["attention_mask"] = attention_mask
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)

    # init ngram model
    ng_list = []
    for i, inputs in enumerate(input_ids):
        ng = NGram(
            inputs.tolist()[prefix_len - prompt_len[i] :],
            n,
            model.config.vocab_size,
            beta,
            sw_coeff,
            stop_words_ids,
        )
        ng_list.append(ng)
    # init ngram model
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
        next_token_scores = outputs.logits[:, -1, :]

        # avoid generating eos
        if not early_stop and eos_token_id != None:
            next_token_scores[:, eos_token_id] = -float("inf")

        # fsd
        next_token_scores = torch.nn.functional.softmax(
            next_token_scores / temperature, dim=-1
        )
        next_token_scores = topk_logits_filter(next_token_scores, k)

        penalty_list = []
        for i, inputs in enumerate(input_ids):
            _, b = torch.topk(next_token_scores[i], k=k)
            penalty_i = ng_list[i].penalize(
                inputs.tolist()[prefix_len - prompt_len[i] :], b.tolist()
            )
            penalty_list.append(penalty_i.view(1, -1))

        batch_penalty = torch.cat(penalty_list, dim=0)
        batch_penalty = batch_penalty.to(model.device)

        next_token_scores = (1 - alpha) * next_token_scores - alpha * batch_penalty
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        # fsd
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )

        # update ngram model
        for i, token in enumerate(next_tokens):
            ng_list[i].update(token.tolist())
        # update ngram model

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, None
        )
        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
    return input_ids


@torch.no_grad()
def fsd_vec_decoding(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    k,
    alpha,
    max_new_tokens,
    temperature=1.0,
    n=2,
    sw_coeff=0.0,
    stop_words_ids=[],
    eos_token_id=None,
    early_stop=False,
    stopping_criteria=None,
):
    """
    - k: top-k candidate words are selected, default 3
    - alpha: (1-alpha)p_lm -(alpha)*penalty
    - max_length: decoding max_length-prompt_length steps
    - n: the order of n-gram models
    - sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1), default 0.
    - stop_words=[]: the list of stopwords. If you use GPT-2, you at least need to add two special tokens ('Ċ' and 'ĊĊ') to avoid grammars errors.
    """
    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    prompt_len = torch.sum(attention_mask, dim=1)
    model_kwargs["attention_mask"] = attention_mask

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)

    # init n-gram model
    ng = HiddenSoftNGram(
        n, model.device, model.config.vocab_size, sw_coeff, stop_words_ids
    )
    # init n-gram model
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # print("model inputs:",model_inputs)
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
        next_token_scores = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states[-1]

        # avoid generating eos
        if not early_stop and eos_token_id != None:
            next_token_scores[:, eos_token_id] = -float("inf")

        # update ngram
        ng.update(hidden_states)
        # update ngram

        # fsd-vec
        next_token_scores = torch.nn.functional.softmax(
            next_token_scores / temperature, dim=-1
        )
        next_token_scores = topk_logits_filter(next_token_scores, k)

        batch_penalty = ng.penalize(input_ids, hidden_states.dtype)
        batch_penalty = batch_penalty.to(model.device)
        next_token_scores = (1 - alpha) * next_token_scores - alpha * batch_penalty

        next_tokens = torch.argmax(next_token_scores, dim=-1)
        # fsd-vec
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, None
        )
        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
    return input_ids


@torch.no_grad()
def topp_decoding(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    top_p,
    max_new_tokens,
    temperature=1.0,
    eos_token_id=None,
    early_stop=False,
):

    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}

    prompt_len = torch.sum(attention_mask, dim=1)
    model_kwargs["attention_mask"] = attention_mask

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)

    for step in range(max_length - prefix_len):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True)
        next_token_scores = outputs.logits[:, -1, :]

        # avoid generating eos
        if not early_stop and eos_token_id != None:
            next_token_scores[:, eos_token_id] = -float("inf")

        # top-p sampling
        next_token_scores = topp_logits_filter(next_token_scores / temperature, top_p)
        probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        # top-p sampling

        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

        if unfinished_sequences.max() == 0 or step == max_length - prefix_len - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
    return input_ids


@torch.no_grad()
def magic_decoding(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    k,
    alpha,
    top_p,
    max_new_tokens,
    temperature=1.0,
    n=2,
    sw_coeff=0.0,
    stop_words_ids=[],
    eos_token_id=None,
    early_stop=False,
):
    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    prompt_len = torch.sum(attention_mask, dim=1)
    model_kwargs["attention_mask"] = attention_mask

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)

    # init n-gram model
    ng = HiddenSoftNGram(
        n, model.device, model.config.vocab_size, sw_coeff, stop_words_ids
    )
    # init n-gram model

    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
        next_token_scores = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states[-1]

        # avoid generating eos
        if not early_stop and eos_token_id != None:
            next_token_scores[:, eos_token_id] = -float("inf")

        # update ngram
        ng.update(hidden_states)
        # update ngram

        if not (step < 10 and random.random() < 0.5):
            # fsd-vec
            next_token_scores = torch.nn.functional.softmax(
                next_token_scores / temperature, dim=-1
            )
            next_token_scores = topk_logits_filter(next_token_scores, k)

            batch_penalty = ng.penalize(input_ids, hidden_states.dtype)
            batch_penalty = batch_penalty.to(model.device)
            next_token_scores = (1 - alpha) * next_token_scores - alpha * batch_penalty
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            # fsd-vec
        else:
            # top-p sampling
            next_token_scores = topp_logits_filter(next_token_scores, top_p)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            # top-p sampling

        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )

    return input_ids


@torch.no_grad()
def mirostat_decoding(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    tau,
    max_new_tokens,
    eos_token_id=None,
    early_stop=False,
    stopping_criteria=None,
):
    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    model_kwargs["attention_mask"] = attention_mask

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)
    n = len(tokenizer)

    target_surprise = tau
    max_surprise = [2 * target_surprise] * batch_size
    error_surprise = [0] * batch_size
    running_tot_surprise = [0] * batch_size
    learning_rate = 1
    indices_surprise = [[] for _ in range(batch_size)]
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
        logits = outputs.logits[:, -1, :]
        if not early_stop and eos_token_id != None:
            logits[:, eos_token_id] = -float("inf")

        sorted_logits_bs, sorted_indices_bs = torch.sort(logits, descending=True)
        prob_original_bs = torch.softmax(sorted_logits_bs, dim=-1).tolist()
        next_tokens = []
        for _ in range(batch_size):
            prob_original = prob_original_bs[_]
            sorted_logits = sorted_logits_bs[_]
            sorted_indices = sorted_indices_bs[_]
            # Estimate s
            s = estimate_s(prob_original)
            # Compute k
            k = compute_k(n, s, max_surprise[_]) + 1

            sorted_logits = sorted_logits[0:k]
            sorted_indices = sorted_indices[0:k]

            prob_topk = torch.softmax(sorted_logits, dim=0)
            prev_i = torch.multinomial(prob_topk, num_samples=1, replacement=True)
            index_surprise = math.log2(1 / prob_original[prev_i])
            indices_surprise[_].append(index_surprise)

            running_tot_surprise[_] += index_surprise
            prev = sorted_indices[prev_i]
            next_tokens.append(prev)
            # generated += prev.tolist()
            # context = torch.tensor([prev.tolist()])  # add ".to('cuda')" if you have a GPU

            # adjust max_surprise
            error_surprise[_] = index_surprise - target_surprise
            max_surprise[_] -= learning_rate * error_surprise[_]
        next_tokens = torch.tensor(next_tokens).to(model.device)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, None
        )

        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )

    return input_ids


@torch.no_grad()
def contrastive_decoding3(
    teacher_model,
    student_model,
    teacher_t,
    student_t,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    eos_token_id=None,
    early_stop=False,
    alpha=0.1,
    beta=0.5,
    stopping_criteria=None,
):
    # formulation of "CONTRASTIVE DECODING IMPROVES REASONING IN LARGE LANGUAGE MODELS"
    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    model_kwargs_student = {}
    model_kwargs["attention_mask"] = attention_mask
    model_kwargs_student["attention_mask"] = attention_mask
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(teacher_model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = teacher_model.prepare_inputs_for_generation(
            input_ids, **model_kwargs
        )
        outputs = teacher_model(
            **model_inputs, return_dict=True, output_hidden_states=True
        )
        next_token_scores = outputs.logits[:, -1, :]
        next_token_scores = next_token_scores / teacher_t
        cutoff = (
            torch.log(torch.tensor(alpha, device=next_token_scores.device))
            + next_token_scores.max(dim=-1, keepdim=True).values
        )

        model_inputs_student = student_model.prepare_inputs_for_generation(
            input_ids, **model_kwargs_student
        )
        outputs_student = student_model(
            **model_inputs_student,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True,
        )

        next_token_logits_student = outputs_student.logits[:, -1, :]
        next_token_logits_student = next_token_logits_student / student_t
        diffs = (1 + beta) * next_token_scores - beta * next_token_logits_student
        cdlogits = diffs.masked_fill(next_token_scores < cutoff, -float("inf"))
        if not early_stop and eos_token_id != None:
            cdlogits[:, eos_token_id] = -float("inf")

        # next_tokens_scores = next_token_scores - alpha * next_token_logits_student
        next_tokens = torch.argmax(cdlogits, dim=-1)
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, None
        )
        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = teacher_model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=teacher_model.config.is_encoder_decoder,
        )
        model_kwargs_student = student_model._update_model_kwargs_for_generation(
            outputs_student,
            model_kwargs_student,
            is_encoder_decoder=student_model.config.is_encoder_decoder,
        )
    return input_ids


import numpy as np
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    LogitsProcessorList,
)
from torch.nn import functional as F


def relative_top_filter(
    scores: torch.FloatTensor,
    relative_top: float = 0.1,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized


@torch.no_grad()
def dola(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens=512,
    repetition_penalty=1.2,
    mature_layer=None,
    base_layer=None,
    candidate_premature_layers=None,
    relative_top=0.1,
    eos_token_id=None,
    early_stop=False,
    stopping_criteria=None,
):
    """
    - k: top-k candidate words are selected, default 3
    - alpha: (1-alpha)p_lm -(alpha)*penalty
    - max_length: decoding max_length-prompt_length steps
    - n: the order of n-gram models
    - sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1), default 0.
    - stop_words=[]: the list of stopwords. If you use GPT-2, you at least need to add two special tokens ('Ċ' and 'ĊĊ') to avoid grammars errors.
    """
    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    prompt_len = torch.sum(attention_mask, dim=1)
    model_kwargs["attention_mask"] = attention_mask

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)

    early_exit_layers = candidate_premature_layers + [mature_layer]
    premature_layer_dist = {l: 0 for l in candidate_premature_layers}

    processors = LogitsProcessorList()
    processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # print("model inputs:",model_inputs)
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
        if early_exit_layers is not None:
            dict_outputs = {}
            # loss_dict = {}
            for i, early_exit_layer in enumerate(early_exit_layers):
                # print(outputs.hidden_states.shape)
                # print(early_exit_layer)
                logits = model.lm_head(outputs.hidden_states[early_exit_layer])
                dict_outputs[early_exit_layer] = logits

        if base_layer is not None:
            base_logits = dict_outputs[base_layer][:, -1, :]
            final_logits = dict_outputs[mature_layer][:, -1, :]
            if relative_top > 0.0:
                final_logits = relative_top_filter(final_logits, relative_top)
                base_logits = base_logits.log_softmax(dim=-1)
                mask = final_logits[0] < -1e3
                base_logits[0][mask] = -1e3

            logits = final_logits - base_logits
            next_token_logits = logits
        else:
            # 1. Stacking all premature_layers into a new dimension
            stacked_premature_layers = torch.stack(
                [dict_outputs[i][:, -1, :] for i in candidate_premature_layers], dim=0
            )

            # 2. Calculate the softmax values for mature_layer and all premature_layers
            softmax_mature_layer = F.softmax(
                dict_outputs[mature_layer][:, -1, :], dim=-1
            )  # shape: (batch_size, num_features)
            softmax_premature_layers = F.softmax(
                stacked_premature_layers, dim=-1
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 3. Calculate M, the average distribution
            M = 0.5 * (
                softmax_mature_layer[None, :, :] + softmax_premature_layers
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 4. Calculate log-softmax for the KL divergence
            log_softmax_mature_layer = F.log_softmax(
                dict_outputs[mature_layer][:, -1, :], dim=-1
            )  # shape: (batch_size, num_features)
            log_softmax_premature_layers = F.log_softmax(
                stacked_premature_layers, dim=-1
            )  # shape: (num_premature_layers, batch_size, num_features)

            # 5. Calculate the KL divergences and then the JS divergences
            kl1 = F.kl_div(
                log_softmax_mature_layer[None, :, :], M, reduction="none"
            ).mean(
                -1
            )  # shape: (num_premature_layers, batch_size)
            kl2 = F.kl_div(log_softmax_premature_layers, M, reduction="none").mean(
                -1
            )  # shape: (num_premature_layers, batch_size)
            js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

            # 6. Reduce the batchmean
            js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
            premature_layer = candidate_premature_layers[
                int(js_divs.argmax().cpu().item())
            ]
            premature_layer_dist[premature_layer] += 1

            base_logits = dict_outputs[premature_layer][:, -1, :]
            final_logits = dict_outputs[mature_layer][:, -1, :]

            if relative_top > 0.0:
                final_logits = relative_top_filter(final_logits, relative_top)
                base_logits = base_logits.log_softmax(dim=-1)
                mask = final_logits[0] < -1e3
                base_logits[0][mask] = -1e3
            logits = final_logits - base_logits
            next_token_logits = logits
            # pre-process distribution
        import copy

        new_next_token_logits = copy.deepcopy(next_token_logits)
        new_next_token_logits = new_next_token_logits.to(input_ids.device)
        next_tokens_scores = processors(input_ids, new_next_token_logits)

        # avoid generating eos
        if not early_stop and eos_token_id != None:
            next_tokens_scores[:, eos_token_id] = -float("inf")

        next_tokens = torch.argmax(next_tokens_scores, dim=-1)
        # fsd-vec
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, None
        )
        if unfinished_sequences.max() == 0 or step == max_new_tokens:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
    return input_ids


def ignore_prefix_prepare_inputs_for_generation(input_ids, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    input_ids = input_ids[:, -1].unsqueeze(-1)
    if token_type_ids is not None:
        token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None

    # if attention_mask is not None:
    #     attention_mask[:, -1].unsqueeze(-1)

    return {
        "input_ids": input_ids,
        "past_key_values": kwargs.get("past_key_values", None),
        "use_cache": kwargs.get("use_cache"),
        "position_ids": None,
        "attention_mask": None,
    }
