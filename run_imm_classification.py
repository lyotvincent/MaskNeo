# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path
import re

import datasets
import torch
from datasets import load_dataset
from datasets import concatenate_datasets
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import transformers
import pandas as pd
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from models.modeling_m2m_100 import M2M100ForDisentangledRepresentation, M2M100ForSequenceClassification
from models.modeling_bart import (
    BartModel,
    BartForSequenceClassification,
    BartForTokenAttentionCLS,
    BartForDisentangledRepresentation,
    BartForTokenAttentionSparseCLS,
    BartForTokenAttentionSparseCLSJoint,
    BartForTokenAttentionCLSMultiClass,
    BartForDisentangledRepresentationMultiClass,
    BartForTokenAttentionSparseCLSJointMultiClass
)
from models.modeling_bert import (
    BertForTokenAttentionSparseCLSJoint
)
from transformers import T5Tokenizer, BertTokenizer
from models.modeling_t5 import T5ForTokenAttentionSparseCLSJoint


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--only_evaluation",
        action="store_true",
        help="If passed, only run evaluation on validation file.",
    )
    parser.add_argument(
        "--prediction_result_file", type=str, default=None, help="A csv or a json file containing prediction result."
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="BertForTokenAttentionSparseCLSJoint",
        help="BertForTokenAttentionSparseCLSJoint",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=-1, help="Number of steps for the warmup in the lr scheduler, the recommended setting is 5 to 10 percent of total training steps."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=999,
        help="Number of epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.

    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", args.task_name)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if args.task_name is not None:
        is_regression = args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    # tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)

    if args.model_class == "BartForSequenceClassification":
        model = BartForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.model_class == "BartForTokenAttentionCLS":
        model = BartForTokenAttentionCLS.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.model_class == "BartForDisentangledRepresentation":
        model = BartForDisentangledRepresentation.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.model_class == "BartForTokenAttentionSparseCLS":
        model = BartForTokenAttentionSparseCLS.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.model_class == "BartForTokenAttentionSparseCLSJoint":
        model = BartForTokenAttentionSparseCLSJoint.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    elif args.model_class == "BertForTokenAttentionSparseCLSJoint":
        if args.only_evaluation:
            model = BertForTokenAttentionSparseCLSJoint.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                ignore_mismatched_sizes=args.ignore_mismatched_sizes,
            )
        else:
            model = BertForTokenAttentionSparseCLSJoint(config)
    elif args.model_class == "T5ForTokenAttentionSparseCLSJoint":
        model = T5ForTokenAttentionSparseCLSJoint.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        )


    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        # add an assert statement to avoid potential errors
        assert "sentence1" in non_label_column_names and "sentence2" in non_label_column_names
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            logger.info(
                f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
                "Using it!"
            )
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    # apply to binary classification tasks, don't hesitate to tweak to your use case.
    assert len(label_to_id) == 2
    assert label_to_id[0] == 0 and label_to_id[1] == 1

    logger.info(f"label_to_id: {label_to_id}")
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in model.config.label2id.items()}

    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        
        processed_texts = []
        for i, text_group in enumerate(texts):
            processed_group = []
            for j, text in enumerate(text_group):
                if i == 1:
                    assert sentence2_key is not None
                    text = text[:180]  # The HLA sequence is truncated to the peptide-binding domain (chain A, residues 1 to 180).
                text = text.upper()
                processed_text = " ".join(list(re.sub(r"[UZOB]", "X", text)))
                processed_group.append(processed_text)
            processed_texts.append(processed_group)
        
        if len(processed_texts) == 1:
            result = tokenizer(processed_texts[0], padding=padding, max_length=args.max_length, truncation=True)
        else:
            result = tokenizer(processed_texts[0], processed_texts[1], padding=padding, max_length=args.max_length, truncation=True)

        if "label" in examples:
            if label_to_id is not None:
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if args.task_name == "mnli" else "validation"]

    positive_sample_indices = [i for i, example in enumerate(train_dataset) if example["labels"] == 1]

    positive_count = len(positive_sample_indices)
    negative_count = len(train_dataset) - positive_count
    logger.info(f"the original training set: positive {positive_count}, negative {negative_count}.")

    if (negative_count / positive_count) > 6:
        positive_times = math.ceil((negative_count / positive_count) / 2)
        positive_samples = train_dataset.select(positive_sample_indices * positive_times)

        train_dataset = concatenate_datasets([train_dataset, positive_samples])

        new_positive_count = sum(1 for example in train_dataset if example["labels"] == 1)
        new_negative_count = len(train_dataset) - new_positive_count
        logger.info(f"the oversampled training set: positive {new_positive_count}, negative {new_negative_count}.")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    temp_max_train_steps = args.max_train_steps
    temp_num_train_epochs = args.num_train_epochs
    
    if args.num_warmup_steps < 0:
        args.num_warmup_steps = math.ceil(0.06 * args.max_train_steps)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # assert args.max_train_steps == temp_max_train_steps and args.num_train_epochs == temp_num_train_epochs

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("imm_classification", experiment_config)


    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    if args.only_evaluation:
        model.eval()
        all_inputs = []
        all_references = []
        all_predictions = []
        for step, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader)) ):
            with torch.no_grad():
                outputs = model(**batch)
            if outputs.disentangle_mask is not None:
                zero_ratio = torch.sum(outputs.disentangle_mask == 0.) / outputs.disentangle_mask.numel()
            # predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
            predictions = (torch.sigmoid(outputs.logits.squeeze()) > 0.5).long()
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            all_inputs += tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
            all_references += references.tolist()
            all_predictions += predictions.tolist()

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process:
            if num_labels > 2:
                eval_metric = [accuracy_score(y_true=all_references, y_pred=all_predictions),
                               precision_score(y_true=all_references, y_pred=all_predictions, average="macro", zero_division=1),
                               recall_score(y_true=all_references, y_pred=all_predictions, average="macro", zero_division=1),
                               f1_score(y_true=all_references, y_pred=all_predictions, average="macro")]
            else:
                eval_metric = [accuracy_score(y_true=all_references, y_pred=all_predictions),
                               precision_score(y_true=all_references, y_pred=all_predictions),
                               recall_score(y_true=all_references, y_pred=all_predictions),
                               f1_score(y_true=all_references, y_pred=all_predictions)]
        logger.info(f"evaluation result: {eval_metric}")
        if outputs.disentangle_mask is not None:
            logger.info(f"zero ratio in mask layer: {zero_ratio}")
        # save prediction result
        if args.prediction_result_file is not None and accelerator.is_main_process:
            df = pd.DataFrame({'text': all_inputs, 'reference': all_references, 'predictions': all_predictions})
            # make sure the directory exists
            os.makedirs(os.path.dirname(args.prediction_result_file), exist_ok=True)
            df.to_csv(args.prediction_result_file, index=False)
            eval_metric_file = os.path.splitext(args.prediction_result_file)[0] + "_eval_metric.json"
            with open(eval_metric_file, "w") as f:
                json.dump({
                    "accuracy": eval_metric[0],
                    "precision": eval_metric[1],
                    "recall": eval_metric[2],
                    "f1": eval_metric[3]
                }, f)
        else:
            logger.info("please provide a prediction_result_file to save prediction result")
    else:
        all_results = {}
        metrics = []
        metrics2 = []
        metrics3 = []
        metrics4 = []
        metrics5 = []
        metrics6 = []
        metrics7 = []
        metrics8 = []
        metrics9 = []
        zero_ratios = []
        best_metric = 0
        patience_counter = 0

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue
                outputs = model(**batch)
                #print(outputs.disentangle_mask, "num zero", torch.sum(outputs.disentangle_mask==0.))
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                        if args.output_dir is not None:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                all_ckpt = [d for d in os.listdir(args.output_dir) if d.startswith("step_") and os.path.isdir(os.path.join(args.output_dir, d))]
                                all_ckpt = sorted(all_ckpt, key=lambda x: int(x.split("_")[1]))
                                # remove old checkpoints if there are more than 3
                                if len(all_ckpt) > 3:
                                    num_to_remove = len(all_ckpt) - 3
                                    for ckpt in all_ckpt[:num_to_remove]:
                                        ckpt_path = os.path.join(args.output_dir, ckpt)
                                        shutil.rmtree(ckpt_path)

                if completed_steps >= args.max_train_steps:
                    break

            model.eval()
            all_references = []
            all_predictions = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                if outputs.disentangle_mask is not None:
                    zero_ratio = torch.sum(outputs.disentangle_mask == 0.) / outputs.disentangle_mask.numel()
                    zero_ratio = zero_ratio.item()
                # predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                predictions = (torch.sigmoid(outputs.logits.squeeze()) > 0.5).long()
                predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
                all_references += references.tolist()
                all_predictions += predictions.tolist()

            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                if num_labels > 2:
                    all_eval_metric = [accuracy_score(y_true=all_references, y_pred=all_predictions),
                                       precision_score(y_true=all_references, y_pred=all_predictions, average="macro", zero_division=1),
                                       recall_score(y_true=all_references, y_pred=all_predictions, average="macro", zero_division=1),
                                       f1_score(y_true=all_references, y_pred=all_predictions, average="macro")]
                else:
                    all_eval_metric = [accuracy_score(y_true=all_references, y_pred=all_predictions),
                                       precision_score(y_true=all_references, y_pred=all_predictions),
                                       recall_score(y_true=all_references, y_pred=all_predictions),
                                       f1_score(y_true=all_references, y_pred=all_predictions)]
            else:
                all_eval_metric = [0.0, 0.0, 0.0, 0.0]

            if args.with_tracking:
                accelerator.log(
                    values={
                        "accuracy": all_eval_metric[0],
                        "precision": all_eval_metric[1],
                        "recall": all_eval_metric[2],
                        "f1": all_eval_metric[3],
                        "train_loss": total_loss / len(train_dataloader),
                        },
                        step=completed_steps
                        )
            
            metrics.append(all_eval_metric[0])
            metrics2.append(all_eval_metric[1])
            metrics3.append(all_eval_metric[2])
            metrics4.append(all_eval_metric[3])
            metrics5.append(total_loss.item() / len(train_dataloader))
            metric_tp = sum(1 for pred, ref in zip(all_predictions, all_references) if pred == 1 and ref == 1)
            metric_tn = sum(1 for pred, ref in zip(all_predictions, all_references) if pred == 0 and ref == 0)
            metric_fp = sum(1 for pred, ref in zip(all_predictions, all_references) if pred == 1 and ref == 0)
            metric_fn = sum(1 for pred, ref in zip(all_predictions, all_references) if pred == 0 and ref == 1)
            metrics6.append(metric_tp)
            metrics7.append(metric_tn)
            metrics8.append(metric_fp)
            metrics9.append(metric_fn)
            logger.info(f"epoch {epoch}: {all_eval_metric}")

            if outputs.disentangle_mask is not None:
                zero_ratios.append(zero_ratio)
                model.config.task_specific_params = f"zero_ratio: {zero_ratio}"
                logger.info(f"zero ratio in mask layer: {zero_ratio}")

            if args.output_dir is not None and accelerator.is_main_process:
                all_results["accuracy"] = metrics
                all_results["precision"] = metrics2
                all_results["recall"] = metrics3
                all_results["f1"] = metrics4
                all_results["loss"] = metrics5
                all_results["best_metric"] = [metrics[metrics4.index(max(metrics4))], metrics2[metrics4.index(max(metrics4))], metrics3[metrics4.index(max(metrics4))], max(metrics4), metrics5[metrics4.index(max(metrics4))]]
                all_results["TP"] = metrics6
                all_results["TN"] = metrics7
                all_results["FP"] = metrics8
                all_results["FN"] = metrics9
                all_results["zero_ratios"] = zero_ratios
                with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                    json.dump(all_results, f)

            should_early_stop = False
            if accelerator.is_main_process:
                if all_eval_metric[3] > best_metric:
                    best_metric = all_eval_metric[3]
                    patience_counter = 0
                    if args.output_dir is not None:
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                        )
                        tokenizer.save_pretrained(args.output_dir)
                    accelerator.wait_for_everyone()
                else:
                    patience_counter += 1
                    logger.info(f"No improvement for {patience_counter} epochs (best metric: {best_metric})")
                    
                    if patience_counter >= args.patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs without improvement")
                        should_early_stop = True
            should_early_stop = accelerator.gather(torch.tensor([should_early_stop], device=accelerator.device)).any().item()
            if should_early_stop:
                accelerator.wait_for_everyone()
                break
    
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()