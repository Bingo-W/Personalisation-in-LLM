
def list_merge(list_of_list: list):
    
    res = []
    for one_list in list_of_list:
        res.extend(one_list)
    return res