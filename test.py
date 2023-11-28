"""
This file is constructed to test the existing pre-trained model
"""

# the public libs
import os

# the self-defined libs
from data_process import(
    MyDatasets
)

from transformers import(
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from utils import (
    create_own_argument,
    output_dir_generation
)

from metrics import(
    build_compute_metrics_fn
)

def main():
    # analyse the inputed argument
    data_args, training_args = create_own_argument()

    output_dir = output_dir_generation(data_args, training_args)
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(output_dir, 'checkpoint-1614'))

    # load the dataset
    my_datasets = MyDatasets(data_args)

    # tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_id)
    tokenized_dataset = my_datasets.tokenization(tokenizer, training_args)

    # load the pre-trained model


if __name__ == '__main__':
    main()