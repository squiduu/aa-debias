import os
import logging
import torch
import torch.nn as nn
from typing import Union, Optional, Tuple
from logging import Logger
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.albert.configuration_albert import AlbertConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM
from transformers.models.albert.modeling_albert import AlbertForMaskedLM
from transformers.training_args import TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput


def clear_console():
    # default to Ubuntu
    command = "clear"
    # if machine is running on Windows
    if os.name in ["nt", "dos"]:
        command = "cls"
    os.system(command)


def get_glue_logger(training_args: TrainingArguments) -> Logger:
    """Create and set environments for logging.

    Args:
        args (Namespace): A parsed arguments.

    Returns:
        logger (Logger): A logger for checking progress.
    """
    # init logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    fmtr = logging.Formatter(
        fmt="%(asctime)s | %(module)s | %(levelname)s > %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    # handler for console
    console_hdlr = logging.StreamHandler()
    console_hdlr.setFormatter(fmtr)
    logger.addHandler(console_hdlr)
    # handler for .log file
    os.makedirs(training_args.output_dir, exist_ok=True)
    file_hdlr = logging.FileHandler(
        filename=training_args.output_dir + f"glue_{training_args.run_name}.log"
    )
    file_hdlr.setFormatter(fmtr)
    logger.addHandler(file_hdlr)

    # notify to start
    logger.info(f"Run number: {training_args.run_name}")

    return logger


class BertClassifier(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_act = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.seq_cls = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, last_hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        # get [CLS] token
        cls_hidden_state = last_hidden_state[:, 0, :]
        cls_hidden_state = self.linear(cls_hidden_state)
        cls_hidden_state = self.linear_act(cls_hidden_state)
        cls_hidden_state = self.dropout(cls_hidden_state)
        logits = self.seq_cls(cls_hidden_state)

        return logits


class RobertaClassifier(nn.Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_act = nn.Tanh()
        self.seq_cls = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, last_hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        # get <s> token
        cls_hidden_state = last_hidden_state[:, 0, :]
        cls_hidden_state = self.dropout(cls_hidden_state)
        cls_hidden_state = self.linear(cls_hidden_state)
        cls_hidden_state = self.linear_act(cls_hidden_state)
        cls_hidden_state = self.dropout(cls_hidden_state)
        logits = self.seq_cls(cls_hidden_state)

        return logits


class AlbertClassifier(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_act = nn.Tanh()
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.seq_cls = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, last_hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        # get [CLS] token
        cls_hidden_state = last_hidden_state[:, 0]
        cls_hidden_state = self.pooler(cls_hidden_state)
        cls_hidden_state = self.pooler_act(cls_hidden_state)
        cls_hidden_state = self.dropout(cls_hidden_state)
        logits = self.seq_cls(cls_hidden_state)

        return logits


class DebiasedEncoder(PreTrainedModel):
    def __init__(
        self,
        model_name_or_path: str,
        config: Union[BertConfig, RobertaConfig, AlbertConfig],
        ckpt_path: str,
    ):
        super().__init__(config)
        self.model_name_or_path = model_name_or_path
        self.num_labels = config.num_labels
        self.config = config
        self.ckpt_path = ckpt_path

        # load pre-trained masked model
        if self.model_name_or_path == "bert-base-uncased":
            self.debiased_masked_model = BertForMaskedLM.from_pretrained(
                pretrained_model_name_or_path=self.ckpt_path, config=self.config
            )
            self.bert_cls = BertClassifier(self.config)
            # init classifier params
            self._init_classifier_weights(self.bert_cls)

        elif self.model_name_or_path == "roberta-base":
            self.debiased_masked_model = RobertaForMaskedLM.from_pretrained(
                pretrained_model_name_or_path=self.ckpt_path, config=self.config
            )
            self.roberta_cls = RobertaClassifier(self.config)
            # init classifier params
            self._init_classifier_weights(self.roberta_cls)

        else:
            self.debiased_masked_model = AlbertForMaskedLM.from_pretrained(
                pretrained_model_name_or_path=self.ckpt_path, config=self.config
            )
            self.albert_cls = AlbertClassifier(self.config)
            # init classifier params
            self._init_classifier_weights(self.albert_cls)

    def _init_classifier_weights(
        self, classifier: Union[BertClassifier, RobertaClassifier, AlbertClassifier]
    ):
        if classifier == self.bert_cls:
            nn.init.normal_(tensor=self.bert_cls.linear.weight.data, mean=0, std=0.02)
            nn.init.zeros_(tensor=self.bert_cls.linear.bias.data)
            nn.init.normal_(tensor=self.bert_cls.seq_cls.weight.data, mean=0, std=0.02)
            nn.init.zeros_(tensor=self.bert_cls.seq_cls.bias.data)

        elif classifier == self.roberta_cls:
            nn.init.normal_(
                tensor=self.roberta_cls.linear.weight.data, mean=0, std=0.02
            )
            nn.init.zeros_(tensor=self.roberta_cls.linear.bias.data)
            nn.init.normal_(
                tensor=self.roberta_cls.seq_cls.weight.data, mean=0, std=0.02
            )
            nn.init.zeros_(tensor=self.roberta_cls.seq_cls.bias.data)

        elif classifier == self.albert_cls:
            nn.init.normal_(tensor=self.albert_cls.pooler.weight.data, mean=0, std=0.02)
            nn.init.zeros_(tensor=self.albert_cls.pooler.bias.data)
            nn.init.normal_(
                tensor=self.albert_cls.seq_cls.weight.data, mean=0, std=0.02
            )
            nn.init.zeros_(tensor=self.roberta_cls.seq_cls.bias.data)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # get output from model
        if self.model_name_or_path == "bert-base-uncased":
            outputs = self.debiased_masked_model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            last_hidden_state = outputs[0]
            logits = self.bert_cls.forward(last_hidden_state)

        elif self.model_name_or_path == "roberta-base":
            outputs = self.debiased_masked_model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            last_hidden_state = outputs[0]
            logits = self.roberta_cls.forward(last_hidden_state)

        else:
            outputs = self.debiased_masked_model.albert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            last_hidden_state = outputs[0]
            logits = self.albert_cls.forward(last_hidden_state)

        # set loss
        loss = None

        # set problem type
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # set loss function
            if self.config.problem_type == "regression":
                loss_fn = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fn.forward(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fn.forward(logits, labels)

            elif self.config.problem_type == "single_label_classification":
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn.forward(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )

            else:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn.forward(logits, labels)

        # set returns
        if not return_dict:
            output = (logits,) + outputs[2:]

            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
