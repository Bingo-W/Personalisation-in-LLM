import random
from collections import OrderedDict

def random_in_user(task_input, profile, retrieve_num=1):

    retrieved_profile = random.sample(profile, retrieve_num)

    return retrieved_profile


BM25_MAPPING = OrderedDict(
    [
        ('Random', random_in_user),
        ('Full_Random', random_in_user),
    ]
)

def build_random_fn(config):
    
    return random_in_user