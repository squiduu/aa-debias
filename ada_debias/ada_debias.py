import sys
import os
import torch
from logging import Logger
from torch.utils.data.dataloader import DataLoader
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import set_seed
from torch.optim.adamw import AdamW
from ada_config import DataArguments, ModelArguments
from ada_utils import (
    clear_console,
    get_cosine_similarity,
    get_ada_logger,
    get_logits,
    overwrite_state_dict,
    prepare_model_and_tokenizer,
    load_words,
    clear_words,
    tokenize_prompts,
    to_cuda,
    JSDivergence,
)


def run_ada_debias(model_args: ModelArguments, data_args: DataArguments, train_args: TrainingArguments, logger: Logger):
    """_summary_

    Args:
        data_args (DataArguments): _description_
        model_args (ModelArguments): _description_
        train_args (TrainingArguments): _description_
        logger (Logger): _description_
    """
    logger.info(f"Model args: {vars(model_args)}")
    logger.info(f"Data args: {vars(data_args)}")
    logger.info(f"Training args: {vars(train_args)}")

    logger.info("Set seed.")
    set_seed(train_args.seed)

    logger.info("Prepare models and tokenizer.")
    fixed_model, tuning_model, tokenizer = prepare_model_and_tokenizer(
        model_name_or_path=model_args.model_name_or_path, run_type=data_args.run_type
    )

    logger.info("Load prompts.")
    prompts = load_words(
        path=data_args.data_dir + f"prompts_{model_args.model_name_or_path}_{data_args.debias_type}",
        run_type=data_args.run_type,
    )

    logger.info(f"Load attribute words for {data_args.debias_type}")
    if data_args.debias_type == "gender":
        _targ1_words = load_words(path=data_args.data_dir + "male.txt", run_type=data_args.run_type)
        _targ2_words = load_words(path=data_args.data_dir + "female.txt", run_type=data_args.run_type)
    elif data_args.debias_type == "race":
        _targ1_words = load_words(path=data_args.data_dir + "af_american.txt", run_type=data_args.run_type)
        _targ2_words = load_words(path=data_args.data_dir + "eu_american.txt", run_type=data_args.run_type)

    logger.info("Remove words that contains OOV tokens.")
    targ1_words, targ2_words = clear_words(
        _words1=_targ1_words, _words2=_targ2_words, tokenizer=tokenizer, run_type=data_args.run_type
    )

    logger.info("Get prompts for fine-tuning.")
    targ1_tokens, targ2_tokens, targ1_mask_idx, targ2_mask_idx = tokenize_prompts(
        prompts=prompts,
        targ1_words=targ1_words,
        targ2_words=targ2_words,
        tokenizer=tokenizer,
    )
    logger.info("Send all tensors to cuda.")
    targ1_tokens, targ2_tokens = to_cuda(targ1_tokens=targ1_tokens, targ2_tokens=targ2_tokens)

    logger.info("Get a dataloader.")
    dataloader = DataLoader(
        dataset=[i for i in range(torch.Tensor.size(targ1_tokens["input_ids"])[0])],
        batch_size=train_args.per_device_train_batch_size,
        shuffle=True,
        num_workers=train_args.dataloader_num_workers,
        pin_memory=True,
    )

    logger.info("Set loss function and optimizers for models.")
    jsd_module = JSDivergence(reduction="batchmean")
    optimizer = AdamW(params=tuning_model.parameters(), lr=train_args.learning_rate)

    logger.info("Start to fine-tune.")
    for epoch in range(1, int(train_args.num_train_epochs) + 1):
        # init loss for an epoch
        epoch_loss = 0.0

        # load batch data
        for iteration, batch_idx in enumerate(dataloader):
            # init batch model inputs
            targ1_inputs = {}
            targ2_inputs = {}

            # get batch inputs with batch index
            for key in targ1_tokens.keys():
                targ1_inputs[key] = targ1_tokens[key][batch_idx]
                targ2_inputs[key] = targ2_tokens[key][batch_idx]
            targ1_batch_mask_idx = targ1_mask_idx[batch_idx]
            targ2_batch_mask_idx = targ2_mask_idx[batch_idx]

            # set gradients as zero
            optimizer.zero_grad()

            # learning step 1: get model logits with respect to target1 and target2
            targ1_fixed_logits, targ1_tuning_logits = get_logits(
                fixed_model=fixed_model,
                tuning_model=tuning_model,
                inputs=targ1_inputs,
                mask_idx=targ1_batch_mask_idx,
                stereotype_ids=None,
                run_type=data_args.run_type,
            )
            targ2_fixed_logits, targ2_tuning_logits = get_logits(
                fixed_model=fixed_model,
                tuning_model=tuning_model,
                inputs=targ2_inputs,
                mask_idx=targ2_batch_mask_idx,
                stereotype_ids=None,
                run_type=data_args.run_type,
            )

            # learning step 2: get JSD for [MASK] token of debias model and bias model
            jsd = jsd_module.forward(net1_logits=targ1_tuning_logits, net2_logits=targ2_tuning_logits)

            # learning step 3: get absolute cosine similarity
            targ1_cos_sim = get_cosine_similarity(logits1=targ1_fixed_logits, logits2=targ1_tuning_logits)
            targ2_cos_sim = get_cosine_similarity(logits1=targ2_fixed_logits, logits2=targ2_tuning_logits)

            # learning step 4: make debias jsd smaller and add regularizations
            distance = torch.mean(
                torch.abs(targ1_fixed_logits - targ1_tuning_logits)
                + torch.abs(targ2_fixed_logits - targ2_tuning_logits)
            )
            cos_sim = (targ1_cos_sim + targ2_cos_sim) / 2.0
            batch_loss = jsd

            # learning step 5: make bias loss smaller
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # record loss
            if (iteration + 1) % 100 == 0:
                logger.info(
                    f"Epoch: {epoch} - Iteration: {iteration + 1}/{len(dataloader)} - JSD: {jsd:.4f} - Distance: {distance:.4f} - CosSim: {cos_sim:.4f}"
                )
            # accumulate batch loss
            epoch_loss += batch_loss

        # for the last iteration
        logger.info(
            f"Epoch: {epoch} - Iteration: {iteration + 1}/{len(dataloader)} - JSD: {jsd:.4f} - Distance: {distance:.4f} - CosSim: {cos_sim:.4f}"
        )
        # after an epoch
        logger.info(f"Epoch: {epoch} - Loss: {epoch_loss / len(dataloader):.4f}")

        if epoch <= train_args.num_train_epochs:
            logger.info("Save debiased model and tokenizer.")
            # get state dict for glue and seat
            glue_model, seat_model = overwrite_state_dict(trained_model=tuning_model, model_args=model_args)
            # save for glue
            glue_model.save_pretrained(
                train_args.output_dir
                + f"{model_args.model_name_or_path}4glue_{train_args.run_name}_{data_args.debias_type}_epoch:{epoch}"
            )
            tokenizer.save_pretrained(
                train_args.output_dir
                + f"{model_args.model_name_or_path}4glue_{train_args.run_name}_{data_args.debias_type}_epoch:{epoch}"
            )
            # save for seat
            seat_model.save_pretrained(
                train_args.output_dir
                + f"{model_args.model_name_or_path}4seat_{train_args.run_name}_{data_args.debias_type}_epoch:{epoch}"
            )
            tokenizer.save_pretrained(
                train_args.output_dir
                + f"{model_args.model_name_or_path}4seat_{train_args.run_name}_{data_args.debias_type}_epoch:{epoch}"
            )


if __name__ == "__main__":
    clear_console()

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, train_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    logger = get_ada_logger(data_args=data_args, train_args=train_args)

    run_ada_debias(model_args=model_args, data_args=data_args, train_args=train_args, logger=logger)
