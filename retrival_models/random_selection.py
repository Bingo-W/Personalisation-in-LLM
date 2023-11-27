import random

def random_in_user(task_input, profile, retrieve_num=1):

    retrieved_profile = random.sample(profile, retrieve_num)

    return retrieved_profile

def build_random_fn(config):
    
    return random_in_user