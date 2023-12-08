"""
This file is constructed to test the existing pre-trained model
"""

# the public libs
import os
import json
import numpy as np

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
    output_dir_generation,
    checkpoint_file_generation
)

from metrics import(
    build_compute_metrics_fn
)

def main():
    # analyse the inputed argument
    data_args, training_args = create_own_argument()

    output_dir = output_dir_generation(data_args, training_args)

    # load the dataset
    my_datasets = MyDatasets(data_args)
    # tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_id)
    tokenized_dataset = my_datasets.tokenization(tokenizer, training_args)

    # load the model and the metrics
    checkpoint_name = os.path.join(output_dir, training_args.model_checkpoint) if training_args.model_checkpoint is not None else checkpoint_file_generation(output_dir)
    print('The model is {}'.format(checkpoint_name))
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_name)
    metrics_fn = build_compute_metrics_fn(data_args.task_name)
    label_pad_toke_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_toke_id,
        pad_to_multiple_of=8
    )


    

    # set the computing argus
    eval_out_dir = os.path.join(output_dir, 'eval')
    running_args = Seq2SeqTrainingArguments(
        output_dir=eval_out_dir,
        per_device_eval_batch_size=6,
        predict_with_generate=True,
        generation_max_length=512,
        generation_num_beams=4,
        fp16=False,
        # optimization details
        # learning_rate=5e-5,
        # weight_decay=10e-4,
        # warmup_ratio=0.05,
        # num_train_epochs=training_args.training_epoch,
        # logging_dir=os.path.join(eval_out_dir, 'logs'),
        # logging_strategy='steps',
        # logging_steps=500,
        # evaluation_strategy='epoch',
        # save_strategy='epoch',
        # save_total_limit=2,
        # load_best_model_at_end=True,
        # metric_for_best_model=MATRICS_MAPPING[data_args.task_name],
        # report_to='tensorboard',
    )

    trainer = Seq2SeqTrainer(
        model = model,
        args = running_args,
        data_collator= data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=lambda x: metrics_fn(x, tokenizer),
    )

    # For evaluation
    if training_args.do_eval:
        trainer.evaluate()

    # For prediction
    if training_args.do_predict:
        predictions = trainer.predict(tokenized_dataset["test"])
    
    model_outputs = predictions.predictions
    label_ids = predictions.label_ids

    model_outputs = np.where(model_outputs != -100, model_outputs, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
    label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    output_data = []

    for output, label_id in zip(decoded_preds, decoded_labels):       

        # Save relevant information to a dictionary
        prediction_info = {
            "true_label": label_id,
            "predicted": output,
        }

        output_data.append(prediction_info)

    # Save the output to a JSON file
    output_file_path = os.path.join(eval_out_dir, "predictions.json")
    with open(output_file_path, "w") as file:
        json.dump(output_data, file, indent=2)

    print(f"Predictions saved to {output_file_path}")

    '''
    There are some useful attribution in the prediction
    .prediction (np.array): the prediction results
    .labe_ids (np.array): the label in the dataset
    .metrics (np.array): the score based on the metrics_fn
    .idxs (np.array): the index
    '''


if __name__ == '__main__':
    main()