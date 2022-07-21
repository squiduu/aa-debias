import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple, Union, List, Dict
from logging import Logger
from torch.nn.parallel import DataParallel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.training_args import TrainingArguments
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
from transformers.models.albert.tokenization_albert import AlbertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertForSequenceClassification, BertModel
from transformers.models.roberta.modeling_roberta import (
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    RobertaModel,
)
from transformers.models.albert.modeling_albert import AlbertForMaskedLM, AlbertForSequenceClassification, AlbertModel
from ada_config import DataArguments, ModelArguments


def clear_console():
    # default to Ubuntu
    command = "clear"
    # if machine is running on Windows
    if os.name in ["nt", "dos"]:
        command = "cls"
    os.system(command)


def get_ada_logger(train_args: TrainingArguments, data_args: DataArguments) -> Logger:
    """Create and set environments for logging.

    Args:
        args (Namespace): A parsed arguments.

    Returns:
        logger (Logger): A logger for checking progress.
    """
    # init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmtr = logging.Formatter(fmt="%(asctime)s | %(module)s | %(levelname)s > %(message)s", datefmt="%Y-%m-%d %H:%M")
    # handler for console
    console_hdlr = logging.StreamHandler()
    console_hdlr.setFormatter(fmtr)
    logger.addHandler(console_hdlr)
    # handler for .log file
    os.makedirs(train_args.output_dir, exist_ok=True)
    if data_args.run_type == "prompt":
        file_hdlr = logging.FileHandler(filename=train_args.output_dir + f"prompt_{train_args.run_name}.log")
    else:
        file_hdlr = logging.FileHandler(filename=train_args.output_dir + f"ada_{train_args.run_name}.log")
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run name: {train_args.run_name}")

    return logger


def prepare_model_and_tokenizer(
    model_name_or_path: str, run_type: str
) -> Union[
    Tuple[
        Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
        Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    ],
    Tuple[
        Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
        Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
        Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    ],
]:
    """Download and prepare the pre-trained model and tokenizer.

    Args:
        model_name (str): A name of pre-trained model.
        run_type (str): A status of either 'prompt' or 'debias'.
    """
    if "bert" in model_name_or_path:
        model_class = BertForMaskedLM
        tokenizer_class = BertTokenizer
    elif "roberta" in model_name_or_path:
        model_class = RobertaForMaskedLM
        tokenizer_class = RobertaTokenizer
    else:
        model_class = AlbertForMaskedLM
        tokenizer_class = AlbertTokenizer

    # get common tokenizer regardless of run type
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)

    if run_type == "prompt":
        model = model_class.from_pretrained(model_name_or_path)

        model = DataParallel(model).eval().cuda()

        return model, tokenizer

    else:
        fixed_model = model_class.from_pretrained(model_name_or_path)
        tuning_model = model_class.from_pretrained(model_name_or_path)

        fixed_model = DataParallel(fixed_model).eval().cuda()
        tuning_model = DataParallel(tuning_model).train().cuda()

        return fixed_model, tuning_model, tokenizer


class JSDivergence(nn.Module):
    def __init__(self, reduction: str = "batchmean") -> None:
        """Get average JS-Divergence between two networks.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "batchmean".
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, net1_logits: torch.FloatTensor, net2_logits: torch.FloatTensor) -> torch.FloatTensor:
        net1_dist = F.softmax(input=net1_logits, dim=1)
        net2_dist = F.softmax(input=net2_logits, dim=1)

        avg_dist = (net1_dist + net2_dist) / 2.0

        jsd = 0.0
        jsd += F.kl_div(input=F.log_softmax(net1_logits, dim=1), target=avg_dist, reduction=self.reduction)
        jsd += F.kl_div(input=F.log_softmax(net2_logits, dim=1), target=avg_dist, reduction=self.reduction)

        return jsd / 2.0


def load_words(path: str, run_type: str) -> List[str]:
    if run_type == "prompt":
        list = []
        with open(file=path, mode="r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                list.append(lines[i].strip().split(sep=" ")[0])

    else:
        list = []
        with open(file=path, mode="r") as f:
            lines = f.readlines()
            for i in range(len(lines)):
                list.append(lines[i].strip())

    return list


def clear_words(
    _words1: List[str],
    _words2: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    run_type: str,
) -> Union[List[str], Tuple[List[str], List[str]]]:
    """Remove the input word if the word contains the out-of-vocabulary token.

    Args:
        _words1 (List[str]): Input words to check the out-of-vocabulary.
        _words2 (List[str]): Input words to check the out-of-vocabulary.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
        run_type (str): A status of either generating prompts or debiasing models.

    Returns:
        Union[List[str], Tuple[List[str], List[str]]]: _description_
    """
    if run_type in ["prompt", "stereotype"] and _words2 is None:
        words = []
        for i in range(len(_words1)):
            if tokenizer.convert_tokens_to_ids(_words1[i]) != tokenizer.unk_token_id:
                words.append(_words1[i])

        return words

    else:
        words1 = []
        words2 = []
        for i in range(len(_words1)):
            if (
                tokenizer.convert_tokens_to_ids(_words1) != tokenizer.unk_token_id
                and tokenizer.convert_tokens_to_ids(_words2) != tokenizer.unk_token_id
            ):
                words1.append(_words1[i])
                words2.append(_words2[i])

        return words1, words2


def to_cuda(targ1_tokens: BatchEncoding, targ2_tokens: BatchEncoding) -> Tuple[BatchEncoding, BatchEncoding]:
    for key in targ1_tokens.keys():
        targ1_tokens[key] = torch.Tensor.cuda(targ1_tokens[key])
        targ2_tokens[key] = torch.Tensor.cuda(targ2_tokens[key])

    return targ1_tokens, targ2_tokens


def tokenize_ith_prompts(
    prompts: List[str],
    targ1_word: str,
    targ2_word: str,
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[BatchEncoding, BatchEncoding, np.ndarray, np.ndarray]:
    """Create prompts with i-th target concept word and tokenize them.

    Args:
        prompts (List[str]): A total prompt words.
        targ1_word (str): An i-th target 1 word.
        targ2_word (str): An i-th target 2 word.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    targ1_sents = []
    targ2_sents = []
    # make targ1 and targ2 sentences
    for i in range(len(prompts)):
        targ1_sents.append(targ1_word + " " + prompts[i] + " " + tokenizer.mask_token)
        targ2_sents.append(targ2_word + " " + prompts[i] + " " + tokenizer.mask_token)

    # tokenize targ1 and targ2 sentences
    targ1_tokens = tokenizer(text=targ1_sents, padding=True, truncation=True, return_tensors="pt")
    targ2_tokens = tokenizer(text=targ2_sents, padding=True, truncation=True, return_tensors="pt")
    # del targ1 and targ2 sentences
    del targ1_sents, targ2_sents

    # get mask token index
    targ1_mask_idx = np.where(torch.Tensor.numpy(targ1_tokens["input_ids"]) == tokenizer.mask_token_id)[1]
    targ2_mask_idx = np.where(torch.Tensor.numpy(targ2_tokens["input_ids"]) == tokenizer.mask_token_id)[1]

    return targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx


def tokenize_prompts(
    prompts: List[str],
    targ1_words: List[str],
    targ2_words: List[str],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
) -> Tuple[BatchEncoding, BatchEncoding, np.ndarray, np.ndarray]:
    """Create prompts with target concept word and tokenize them.

    Args:
        prompts (List[str]): A total prompt words.
        targ1_words (List[str]): An i-th target 1 word.
        targ2_words (List[str]): An i-th target 2 word.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
    """
    targ1_sents = []
    targ2_sents = []
    # make targ1 and targ2 sentences
    for i in range(len(prompts)):
        for j in range(len(targ1_words)):
            targ1_sents.append(targ1_words[j] + " " + prompts[i] + " " + tokenizer.mask_token + ".")
            targ2_sents.append(targ2_words[j] + " " + prompts[i] + " " + tokenizer.mask_token + ".")

    # tokenize targ1 and targ2 sentences
    targ1_tokens = tokenizer.__call__(text=targ1_sents, padding=True, truncation=True, return_tensors="pt")
    targ2_tokens = tokenizer.__call__(text=targ2_sents, padding=True, truncation=True, return_tensors="pt")

    # del targ1 and targ2 sentences
    del targ1_sents, targ2_sents

    # get mask token index
    targ1_mask_idx = np.where(torch.Tensor.numpy(targ1_tokens["input_ids"]) == tokenizer.mask_token_id)[1]
    targ2_mask_idx = np.where(torch.Tensor.numpy(targ2_tokens["input_ids"]) == tokenizer.mask_token_id)[1]

    return targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx


def to_cuda(targ1_tokens: BatchEncoding, targ2_tokens: BatchEncoding) -> Tuple[BatchEncoding, BatchEncoding]:
    for key in targ1_tokens.keys():
        targ1_tokens[key] = torch.Tensor.cuda(targ1_tokens[key])
        targ2_tokens[key] = torch.Tensor.cuda(targ2_tokens[key])

    return targ1_tokens, targ2_tokens


def get_batch_inputs(
    batch_idx: int,
    targ1_tokens: BatchEncoding,
    targ2_tokens: BatchEncoding,
    targ1_mask_idx: np.ndarray,
    targ2_mask_idx: np.ndarray,
    train_args: TrainingArguments,
) -> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.LongTensor], np.ndarray, np.ndarray]:
    """Slice all inputs as `batch_size`.

    Args:
        batch_idx (int): An index for batch.
        targ1_tokens (BatchEncoding): Tokens for target 1 concepts.
        targ2_tokens (BatchEncoding): Tokens for target 2 concepts.
        targ1_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens.
        targ2_mask_idx (np.ndarray): Positions for [MASK] token in target 2 concept tokens.
        args (TrainingArguments): A parsed arguments.

    Returns:
        targ1_inputs (Dict[str, torch.LongTensor]): Tokens for target 1 concepts sliced as `batch_size`.
        targ2_inputs (Dict[str, torch.LongTensor]): Tokens for target 2 concepts sliced as `batch_size`.
        targ1_batch_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens sliced as `batch_size`.
        targ2_batch_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens sliced as `batch_size`.
    """
    targ1_inputs = {}
    targ2_inputs = {}

    try:
        for key in targ1_tokens.keys():
            # slice to batch size
            targ1_inputs[key] = targ1_tokens[key][
                train_args.per_device_train_batch_size
                * batch_idx : train_args.per_device_train_batch_size
                * (batch_idx + 1)
            ]
            targ2_inputs[key] = targ2_tokens[key][
                train_args.per_device_train_batch_size
                * batch_idx : train_args.per_device_train_batch_size
                * (batch_idx + 1)
            ]

        targ1_batch_mask_idx = targ1_mask_idx[
            train_args.per_device_train_batch_size
            * batch_idx : train_args.per_device_train_batch_size
            * (batch_idx + 1)
        ]
        targ2_batch_mask_idx = targ2_mask_idx[
            train_args.per_device_train_batch_size
            * batch_idx : train_args.per_device_train_batch_size
            * (batch_idx + 1)
        ]

    except IndexError:
        for key in targ1_tokens.keys():
            # get rest of batches
            targ1_inputs[key] = targ1_tokens[key][train_args.per_device_train_batch_size * (batch_idx + 1) :]
            targ2_inputs[key] = targ2_tokens[key][train_args.per_device_train_batch_size * (batch_idx + 1) :]

        targ1_batch_mask_idx = targ1_mask_idx[train_args.per_device_train_batch_size * (batch_idx + 1) :]
        targ2_batch_mask_idx = targ2_mask_idx[train_args.per_device_train_batch_size * (batch_idx + 1) :]

    return targ1_inputs, targ2_inputs, targ1_batch_mask_idx, targ2_batch_mask_idx


def get_logits(
    fixed_model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    tuning_model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    inputs: Dict[str, torch.LongTensor],
    mask_idx: np.ndarray,
    stereotype_ids: List[int],
    run_type: str,
) -> Union[Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor], torch.FloatTensor]:
    """Get logits corresponding to stereotype words at [MASK] token position.

    Args:
        freezing_model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model for freezing.
        tuning_model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model for fine-tuning.
        inputs (Dict[str, torch.LongTensor]): Tokenized prompt inputs with a [MASK] token.
        mask_idx (np.ndarray): An index of a [MASK] token in tokenized prompt inputs.
        stereotype_ids (List[int]): Pre-defined stereotype ids.
        run_type (str): A status of either generating prompts or debiasing models.
    """
    if run_type == "debias":
        fixed_outputs = fixed_model.forward(**inputs)
        tuning_outputs = tuning_model.forward(**inputs)

        fixed_logits = fixed_outputs.logits[torch.arange(torch.Tensor.size(fixed_outputs.logits)[0]), mask_idx]
        tuning_logits = tuning_outputs.logits[torch.arange(torch.Tensor.size(tuning_outputs.logits)[0]), mask_idx]

        return fixed_logits, tuning_logits

    else:
        outputs = fixed_model.forward(**inputs)
        # extract logits only for stereotype words
        logits = outputs.logits[np.arange(torch.Tensor.size(inputs["input_ids"])[0]), mask_idx][:, stereotype_ids]

        return logits


def get_model_outputs(
    fixed_model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    tuning_model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    inputs: Dict[str, torch.LongTensor],
    mask_idx: np.ndarray,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Get the last hidden states of input sequence and logits on [MASK] token position.

    Args:
        freezing_model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model for freezing.
        tuning_model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model for fine-tuning.
        inputs (Dict[str, torch.LongTensor]): Tokenized prompt inputs with a [MASK] token.
        mask_idx (np.ndarray): An index of a [MASK] token in tokenized prompt inputs.
    """
    fixed_outputs = fixed_model.forward(**inputs, output_hidden_states=True)
    tuning_outputs = tuning_model.forward(**inputs, output_hidden_states=True)

    # get last hidden state
    tuning_hidden = tuning_outputs.hidden_states[-1] / torch.norm(tuning_outputs.hidden_states[-1])

    # get [MASK] logits
    fixed_logits = fixed_outputs.logits[torch.arange(torch.Tensor.size(fixed_outputs.logits)[0]), mask_idx]
    tuning_logits = tuning_outputs.logits[torch.arange(torch.Tensor.size(tuning_outputs.logits)[0]), mask_idx]

    return tuning_hidden, fixed_logits, tuning_logits


def get_cosine_similarity(logits1: torch.FloatTensor, logits2: torch.FloatTensor) -> torch.FloatTensor:
    cos_sim = F.cosine_similarity(logits1, logits2)

    return cos_sim.mean()


def get_jsd_values(
    targ1_tokens: BatchEncoding,
    targ2_tokens: BatchEncoding,
    targ1_mask_idx: np.ndarray,
    targ2_mask_idx: np.ndarray,
    model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    stereotype_ids: List[int],
    jsd_module: JSDivergence,
    train_args: TrainingArguments,
) -> List[np.ndarray]:
    """Calculate JS-Divergence values and accumulate them for all prompts of i-th target concept word.

    Args:
        targ1_tokens (BatchEncoding): Tokens for target 1 concepts.
        targ2_tokens (BatchEncoding): Tokens for target 2 concepts.
        targ1_mask_idx (np.ndarray): Positions for [MASK] token in target 1 concept tokens.
        targ2_mask_idx (np.ndarray): Positions for [MASK] token in target 2 concept tokens.
        model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model.
        stereotype_ids (List[int]): Pre-defined stereotype ids.
        js_div_module (JSDivergence): A JS-Divergence module.
        args (Namespace): A parsed arguments.

    Returns:
        js_div_values (List[np.ndarray]): _description_
    """
    jsd_values = []
    # send all tokens to cuda for dataparallel
    targ1_tokens, targ2_tokens = to_cuda(targ1_tokens=targ1_tokens, targ2_tokens=targ2_tokens)

    for batch_idx in range(
        torch.Tensor.size(targ1_tokens["input_ids"])[0] // train_args.per_device_train_batch_size + 1
    ):
        # slice inputs as batch size
        targ1_inputs, targ2_inputs, targ1_batch_mask_idx, targ2_batch_mask_idx = get_batch_inputs(
            batch_idx=batch_idx,
            targ1_tokens=targ1_tokens,
            targ2_tokens=targ2_tokens,
            targ1_mask_idx=targ1_mask_idx,
            targ2_mask_idx=targ2_mask_idx,
            train_args=train_args,
        )

        # get logits of stereotype words
        targ1_logits = get_model_outputs(
            fixed_model=model,
            bias_model=None,
            tuning_model=None,
            inputs=targ1_inputs,
            mask_idx=targ1_batch_mask_idx,
            stereotype_ids=stereotype_ids,
            run_type="prompt",
        )
        targ2_logits = get_model_outputs(
            fixed_model=model,
            bias_model=None,
            tuning_model=None,
            inputs=targ2_inputs,
            mask_idx=targ2_batch_mask_idx,
            stereotype_ids=stereotype_ids,
            run_type="prompt",
        )

        # get JS-Divergence value for two networks
        jsd_value = jsd_module.forward(net1_logits=targ1_logits, net2_logits=targ2_logits)
        jsd_sum = np.sum(jsd_value.detach().cpu().numpy(), axis=1)
        # accumulate all JS-Divergence values
        jsd_values += list(jsd_sum)

        del targ1_logits, targ2_logits, jsd_value

    return jsd_values


def get_prompt_jsd(
    prompts: List[str],
    targ1_words: List[str],
    targ2_words: List[str],
    model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM],
    tokenizer: Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer],
    stereotype_ids: List[int],
    jsd_module: JSDivergence,
    train_args: TrainingArguments,
) -> np.ndarray:
    """Get JS-Divergence values for all prompts of all target concept words about bias.

    Args:
        prompts (List[str]): Candidate words for prompts.
        targ1_words (List[str]): Words for target 1 concepts.
        targ2_words (List[str]): Words for target 2 concepts.
        model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A pre-trained language model.
        tokenizer (Union[BertTokenizer, RobertaTokenizer, AlbertTokenizer]): A pre-trained tokenizer.
        stereotype_ids (List[int]): Pre-defined stereotype ids.
        js_div_module (JSDivergence): A JS-Divergence module.
        train_args (TrainingArguments): A parsed arguments.

    Returns:
        accum_prompt_js_div_values (np.ndarray): Accumulated JS-Divergence values for all prompts.
    """
    prompt_jsd = []
    # for i in tqdm(iterable=range(len(targ1_words))):
    for i in tqdm(range(len(targ1_words))):
        # create all possible prompts combination for i-th target concept word
        targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx = tokenize_ith_prompts(
            prompts=prompts,
            targ1_word=targ1_words[i],
            targ2_word=targ2_words[i],
            tokenizer=tokenizer,
        )
        # get JS-Divergence values of i-th target concept word
        jsd_values = get_jsd_values(
            targ1_tokens=targ1_tokens,
            targ2_tokens=targ2_tokens,
            targ1_mask_idx=targ1_mask_idx,
            targ2_mask_idx=targ2_mask_idx,
            model=model,
            stereotype_ids=stereotype_ids,
            jsd_module=jsd_module,
            train_args=train_args,
        )
        # accumulate all target concept words
        prompt_jsd.append(jsd_values)
    prompt_jsd = np.array(prompt_jsd)
    accum_prompt_jsd = np.mean(prompt_jsd, axis=0)

    return accum_prompt_jsd


def overwrite_state_dict(
    trained_model: Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM], model_args: ModelArguments
) -> Tuple[
    Union[BertForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification],
    Union[BertModel, RobertaModel, AlbertModel],
]:
    """Extract and transfer only the trained weights of the layer matching the new model.

    Args:
        trained_model (Union[BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM]): A debiased model.
        model_args (ModelArguments): A parsed model arguments.
    """
    if "bert" in model_args.model_name_or_path:
        glue_model_class = BertForSequenceClassification
        seat_model_class = BertModel
    elif "roberta" in model_args.model_name_or_path:
        glue_model_class = RobertaForSequenceClassification
        seat_model_class = RobertaModel
    else:
        glue_model_class = AlbertForSequenceClassification
        seat_model_class = AlbertModel

    # get initialized pre-trained model
    glue_model = glue_model_class.from_pretrained(model_args.model_name_or_path)
    seat_model = seat_model_class.from_pretrained(model_args.model_name_or_path)

    # get initialized pre-trained model weights
    glue_model_dict = glue_model.state_dict()
    seat_model_dict = seat_model.state_dict()
    # filter out unnecessary keys in debiased masked model
    state_dict_for_glue = {k: v for k, v in trained_model.state_dict().items() if k in glue_model_dict}
    if seat_model_class == BertModel:
        state_dict_for_seat = {k[5:]: v for k, v in trained_model.state_dict().items() if k[5:] in seat_model_dict}
    elif seat_model_class == RobertaModel:
        state_dict_for_seat = {k[8:]: v for k, v in trained_model.state_dict().items() if k[8:] in seat_model_dict}
    else:
        state_dict_for_seat = {k[7:]: v for k, v in trained_model.state_dict().items() if k[7:] in seat_model_dict}

    # overwrite entries in the existing initialized state dict
    glue_model_dict.update(state_dict_for_glue)
    seat_model_dict.update(state_dict_for_seat)

    # overwrite updated weights
    glue_model.load_state_dict(glue_model_dict)
    seat_model.load_state_dict(seat_model_dict)

    return glue_model, seat_model
