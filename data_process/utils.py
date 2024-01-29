from collections import OrderedDict

def list_merge(list_of_list: list):
    
    res = []
    for one_list in list_of_list:
        res.extend(one_list)
    return res

def merge_user_profile(profile_list_a, profile_list_b, task_name):

    OUTPUT_INDEX = OrderedDict(
        [
            ('LaMP_1', None),
            ('LaMP_2', 'tag'),
            ('LaMP_3', 'score'),
            ('LaMP_4', 'title'),
            ('LaMP_5', 'title'),
            ('LaMP_6', None),
            ('LaMP_7', 'text'),
        ]
    )
    output_index = OUTPUT_INDEX[task_name]

    res_profile = []
    for item_a, item_b in zip(profile_list_a, profile_list_b):
        item_c = []
        for profile_a, profile_b in zip(item_a, item_b):
            profile_c = profile_a.copy()
            profile_c[output_index] = profile_b[output_index]
            item_c.append(profile_c)
        res_profile.append(item_c)

    
    return res_profile

def construct_for_llama2(original_input, task_name):
    MODIFIED_ANCHOR = OrderedDict(
        [
            ('LaMP_1', None),
            ('LaMP_2', 'article: '),
            ('LaMP_3', 'review: '),
            ('LaMP_4', 'article: '),
            ('LaMP_5', 'paper: '),
            ('LaMP_6', None),
            ('LaMP_7', 'before or after it: '),
        ]
    )
    if MODIFIED_ANCHOR[task_name] is not None:
        split_anchor = MODIFIED_ANCHOR[task_name]
        split_input = original_input.split(split_anchor)
        modified_input = split_input[0] + split_anchor + '"' + split_input[1] + '"'
    else:
        modified_input = original_input
        
    return modified_input + '\nAnswer:'