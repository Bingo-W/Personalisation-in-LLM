# the public libs
import os
import json
import numpy as np

# the self-defined libs
from data_process import(
    LlamaDatasets
)

import transformers
import torch
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

from transformers import(
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoModelForCausalLM,
)

from utils import (
    create_own_argument,
    output_dir_generation
)

from metrics import(
    build_compute_metrics_fn
)

MATRICS_MAPPING = {
    "LaMP_1": 'accuracy',
    "LaMP_2": 'accuracy',
    "LaMP_3": 'mae',
    "LaMP_4": 'rouge1',
    "LaMP_5": 'rouge1',
    "LaMP_6": 'rouge1',
    "LaMP_7": 'rouge1',
}


def main():
    # analyse the inputed argument
    data_args, training_args = create_own_argument()

    output_dir = output_dir_generation(data_args, training_args)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # load the dataset
    my_datasets = LlamaDatasets(data_args)

    # load the tokanizer and the pipeline
    access_token = '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'
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_id, token=access_token)
    pipeline = transformers.pipeline(
        "text-generation",
        model=training_args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=access_token
    )

    tokenized_dataset = my_datasets.tokenization(tokenizer, training_args)

    results = []
    labels = tokenized_dataset['labels']
    for item in tqdm( tokenized_dataset['input'], total=len(tokenized_dataset)):
        
        sequences = pipeline(
            item,
            top_k=10,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=4,
            max_new_tokens=128,
        )

        final_res = sequences[0]['generated_text'].split('Answer:')[-1].strip().split('\n')[0].strip().strip('"')
        results.append(final_res)

    output_data = []
    for output, label in zip(results, labels):
        prediction_info = {
            "true_label": label,
            "predicted": output,
        }
        output_data.append(prediction_info)

    output_file_path = os.path.join(output_dir, "predictions.json")
    with open(output_file_path, "w") as file:
        json.dump(output_data, file, indent=2)

    print(f"Predictions saved to {output_file_path}")




if __name__ == '__main__':
    main()