import numpy as np
import evaluate

def text_accuarcy(preds:list, labels:list):

    assert (len(preds) == len(labels))
    match_num = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            match_num+=1
    
    return {"accuracy": match_num/len(preds)}

def text_F1(preds:list, labels:list):

    assert (len(preds) == len(labels))

    # transform the text into digital
    keys_label = {string: i + 1 for i, string in enumerate(set(labels))}
    transformed_preds = []
    for pred in preds:
        if pred in keys_label.keys():
            transformed_preds.append(keys_label[pred])
        else:
            transformed_preds.append(-1)
    
    transformed_labels = [keys_label[label] for label in labels]

    # load the F1-score
    f1_metric = evaluate.load("f1")
    results = f1_metric.compute(predictions=transformed_preds, references=transformed_labels, average="macro")
    
    return results

MATRICS_MAPPING = {
    "LaMP_1": [text_accuarcy],
    "LaMP_2": [text_accuarcy, text_F1]
}

def LaMP1_postprocess_test(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

def LaMP1_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # construct the metrics
    metrics = MATRICS_MAPPING['LaMP_1']

    results = {}
    for metric in metrics:
        results.update(metric(decoded_preds, decoded_labels))
    
    return results

def LaMP2_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # construct the metrics
    metrics = MATRICS_MAPPING['LaMP_2']

    results = {}
    for metric in metrics:
        results.update(metric(decoded_preds, decoded_labels))
    
    return results
