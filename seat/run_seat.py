from argparse import Namespace
from csv import DictWriter
import json
from logging import Logger
import numpy as np
import os
from seat_utils import (
    clear_console,
    get_seat_logger,
    get_keys_to_sort_tests,
    check_availability,
    get_encoded_vectors,
    save_encoded_vectors,
    save_encodings,
    set_seed,
    load_model_and_tokenizer,
    get_encodings,
)
from seat_config import get_seat_args, TEST_EXT
import weat


def run_seat(args: Namespace, logger: Logger):
    """Parse args for seat to run and which models to evaluate.

    Args:
        args (Namespace): A parsed arguments.
        logger (Logger): A logger for checking process.
    """
    # set seed
    if args.seed >= 0:
        logger.info(f"Seed: {args.seed}")
        set_seed(args)

    # get all tests
    all_tests = sorted(
        [
            entry[: -len(TEST_EXT)]
            for entry in os.listdir(args.data_dir)
            if not entry.startswith(".") and entry.endswith(TEST_EXT)
        ],
        key=get_keys_to_sort_tests,
    )
    logger.info(f"Found tests: {all_tests}")

    # check the available tests
    tests = (
        check_availability(arg_str=args.tests, allowed_set=all_tests, item_type="test")
        if args.tests is not None
        else all_tests
    )
    logger.info(f"Selected tests: {tests}")

    # check the available models
    available_models = (
        check_availability(arg_str=args.model_name, allowed_set=["bert", "roberta", "albert"], item_type="model")
        if args.model_name is not None
        else ["bert", "roberta", "albert"]
    )
    logger.info(f"Selected models: {available_models}")

    results = []
    for model_name in available_models:
        logger.info(f"Start to run the SEAT for {model_name}.")
        # load the model and tokenizer
        model, tokenizer = load_model_and_tokenizer(version=args.version, args=args)

        for test in tests:
            logger.info(f"Start to run {test} for {model_name}.")

            # get encoded file
            enc_path = os.path.join(args.exp_dir, f"{args.version}.h5" if args.version else f"{model_name}_{test}.h5")

            # load encoded vectors or test dataset to encode
            if not args.ignore_cached_encs and os.path.isfile(enc_path):
                encs_targ1, encs_targ2, encs_attr1, encs_attr2 = get_encoded_vectors(enc_path=enc_path)
            else:
                test_data = json.load(fp=open(os.path.join(args.data_dir, f"{test}{TEST_EXT}"), mode="r"))

                # get encodings
                encs_targ1, encs_targ2, encs_attr1, encs_attr2 = get_encodings(
                    data_keys=["targ1", "targ2", "attr1", "attr2"], data=test_data, model=model, tokenizer=tokenizer
                )

            # save encoded vectors in `test_data` with `data` key name
            encoded_data = save_encoded_vectors(
                data=test_data,
                encs_targ1=encs_targ1,
                encs_targ2=encs_targ2,
                encs_attr1=encs_attr1,
                encs_attr2=encs_attr2,
            )
            if args.cache_encs:
                logger.info(f"Save the encodings to {enc_path}")
                save_encodings(encodings=encoded_data, enc_path=enc_path)

            # get WEAT results and save them as a result dict
            effect_size, p_value = weat.run_test(
                encs=encoded_data, num_samples=args.num_samples, use_parametric=args.use_parametric, logger=logger
            )
            results.append(
                dict(
                    version=args.version,
                    test=test,
                    p_value=round(p_value, 4),
                    effect_size=round(effect_size, 2),
                    avg_abs_effect_size=None,
                )
            )

        avg_abs_effect_size = {
            "avg_abs_effect_size": sum([abs(results[i]["effect_size"]) for i in range(len(results))]) / len(results)
        }

        for result in results:
            logger.info("Test: {test}\tp-value: {p_value:.9f}\teffect-size: {effect_size:.2f}".format(**result))
        logger.info(f"Average absolute effect-size: {round(avg_abs_effect_size['avg_abs_effect_size'], 2)}")

    if args.results_path is not None:
        logger.info(f"Save the SEAT results to {args.results_path}")

        results[-1]["avg_abs_effect_size"] = round(avg_abs_effect_size["avg_abs_effect_size"], 2)

        with open(file=args.results_path, mode="w") as res_fp:
            writer = DictWriter(f=res_fp, fieldnames=dict.keys(results[0]), delimiter="\t")
            writer.writeheader()
            for result in results:
                writer.writerow(result)


if __name__ == "__main__":
    clear_console()

    args = get_seat_args()
    logger = get_seat_logger(args)
    run_seat(args=args, logger=logger)
