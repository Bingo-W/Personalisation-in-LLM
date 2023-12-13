from collections import OrderedDict
from .bm25 import build_bm25_fn
from .random_selection import build_random_fn

IR_METHOD_MAPPING = OrderedDict(
    [
        ('BM25', build_bm25_fn),
        ('Random', build_random_fn),
        ('Full_Random', build_random_fn),
    ]
)

def build_mix_fn(config):
    
    retrieval_fn = {}

    if config['input_retrieval_id'] in IR_METHOD_MAPPING:
            retrieval_fn['input'] = IR_METHOD_MAPPING[config['input_retrieval_id']](config)
    else:
        return ValueError(
            "The retrieval method does not exist in the produced list."
        )
    
    if config['output_retrieval_id'] in IR_METHOD_MAPPING:
            retrieval_fn['output'] =  IR_METHOD_MAPPING[config['output_retrieval_id']](config)
    else:
        return ValueError(
            "The retrieval method does not exist in the produced list."
        )
    
    return retrieval_fn