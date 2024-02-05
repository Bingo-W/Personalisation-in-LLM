import numpy as np
import evaluate
import math

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

def text_MAE(preds:list, labels:list):
    assert (len(preds) == len(labels))

    transform_labels = [int(item) if len(item)==1 else int(item)[:-4] for item in labels]

    transform_preds = []
    for index_, item in enumerate(preds):
        try:
            transform_preds.append(int(item))
        except:
            transform_preds.append(transform_labels[index_]-5)
    

    mae_metric = evaluate.load("mae")
    results = mae_metric.compute(predictions=transform_preds, references=transform_labels)

    return results

def text_RMSE(preds:list, labels:list):
    assert (len(preds) == len(labels))

    transform_labels = [int(item) if len(item)==1 else int(item)[:-4] for item in labels]

    transform_preds = []
    for index_, item in enumerate(preds):
        try:
            transform_preds.append(int(item))
        except:
            transform_preds.append(transform_labels[index_]-5)

    mae_metric = evaluate.load("mse")
    results = mae_metric.compute(predictions=transform_preds, references=transform_labels)
    rmse_res = math.sqrt(results['mse'])

    return {'rmse': rmse_res}

def text_rouge(preds:list, labels:list):
    assert (len(preds) == len(labels))

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=preds,
                            references=labels,
                            use_aggregator=True)
    
    return results

MATRICS_MAPPING = {
    "LaMP_1": [text_accuarcy],
    "LaMP_2": [text_accuarcy, text_F1],
    "LaMP_3": [text_MAE, text_RMSE],
    "LaMP_4": [text_rouge],
    "LaMP_5": [text_rouge],
    "LaMP_6": [text_rouge],
    "LaMP_7": [text_rouge],
}

def LaMP1_postprocess_test(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    return preds, labels

def LaMP1_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
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
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # construct the metrics
    metrics = MATRICS_MAPPING['LaMP_2']

    results = {}
    for metric in metrics:
        results.update(metric(decoded_preds, decoded_labels))
    
    return results

def LaMP3_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # construct the metrics
    metrics = MATRICS_MAPPING['LaMP_3']

    results = {}
    for metric in metrics:
        results.update(metric(decoded_preds, decoded_labels))
    
    return results

def LaMP4_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # construct the metrics
    metrics = MATRICS_MAPPING['LaMP_4']

    results = {}
    for metric in metrics:
        results.update(metric(decoded_preds, decoded_labels))
    
    return results

def LaMP5_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # construct the metrics
    metrics = MATRICS_MAPPING['LaMP_5']

    results = {}
    for metric in metrics:
        results.update(metric(decoded_preds, decoded_labels))
    
    return results

def LaMP6_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # construct the metrics
    metrics = MATRICS_MAPPING['LaMP_6']

    results = {}
    for metric in metrics:
        results.update(metric(decoded_preds, decoded_labels))
    
    return results

def LaMP7_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # construct the metrics
    metrics = MATRICS_MAPPING['LaMP_7']

    results = {}
    for metric in metrics:
        results.update(metric(decoded_preds, decoded_labels))
    
    return results

def metrics_for_llama(task_names, preditions, references):
    metrics = MATRICS_MAPPING[task_names]
    results = {}

    for metric in metrics:
        results.update(metric(preditions, references))

    return results