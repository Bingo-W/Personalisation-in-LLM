import numpy as np
import evaluate

def text_accuarcy(preds:list, labels:list):

    assert (len(preds) != len(labels))
    match_num = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            match_num+=1
    
    return {"accuracy": match_num/len(preds)}

MATRICS_MAPPING = {
    "LaMP_1": [text_accuarcy]
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