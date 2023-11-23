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
    
    # load the dataset
    my_datasets = MyDatasets(data_args)
    
    # tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(training_args.model_id)
    tokenized_dataset = my_datasets.tokenization(tokenizer, training_args)

    # load the model, dataloader and metrics
    metrics_fn = build_compute_metrics_fn(data_args.task_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(training_args.model_id)
    label_pad_toke_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_toke_id,
        pad_to_multiple_of=8
    )


    output_dir = output_dir_generation(data_args, training_args)
    # define the training hyper-parameters
    running_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_num_beams=4,
        fp16=True,
        # optimization details
        learning_rate=5e-5,
        weight_decay=10e-4,
        warmup_ratio=0.05,
        num_train_epochs=10,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_strategy='steps',
        logging_steps=500,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
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

    trainer.train()

if __name__ == '__main__':
    main()