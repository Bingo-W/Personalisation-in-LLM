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
    Seq2SeqTrainer,
    AutoModelForCausalLM
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
    
    # load the dataset
    my_datasets = MyDatasets(data_args)
    
    # tokenize the dataset
    if training_args.model_id == 'meta-llama/Llama-2-7b-hf':
        access_token = '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'
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token, padding_side = "left")
        tokenizer.pad_token = '[PAD]'
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token, torch_dtype = "auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(training_args.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(training_args.model_id)
    # load the model, dataloader and metrics
    metrics_fn = build_compute_metrics_fn(data_args.task_name)
    tokenized_dataset = my_datasets.tokenization(tokenizer, training_args)
    label_pad_toke_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_toke_id,
        pad_to_multiple_of=8
    )


    output_dir = output_dir_generation(data_args, training_args)
    print(output_dir)
    # define the training hyper-parameters
    running_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_args.batch_size,
        per_device_eval_batch_size=training_args.batch_size,
        predict_with_generate=True,
        generation_max_length=1024,
        generation_num_beams=4,
        #bf16=True,
        #fp16=training_args.fp16,
        # optimization details
        learning_rate=5e-5,
        weight_decay=10e-4,
        warmup_ratio=0.05,
        num_train_epochs=training_args.training_epoch,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_strategy='steps',
        logging_steps=500,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=MATRICS_MAPPING[data_args.task_name],
        report_to='tensorboard',
    )


    # training code
    trainer = Seq2SeqTrainer(
        model = model,
        args = running_args,
        data_collator= data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=lambda x: metrics_fn(x, tokenizer),
    )

    if training_args.do_train:
        trainer.train()

    if training_args.do_eval:
        eval_results = trainer.evaluate()
        output_file_path = os.path.join(output_dir, "evaluation_results.txt")
        with open(output_file_path, "w") as file:
            for metric_name, metric_value in eval_results.items():
                file.write(f"{metric_name}: {metric_value}\n")

if __name__ == '__main__':
    main()