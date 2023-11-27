import os

def output_dir_generation(data_args, training_args):

    model_name = training_args.model_id.split('/')[1]
    
    # add the model name
    output_dir = os.path.join(training_args.result_dir, model_name)

    # add the retrieval name
    output_dir = os.path.join(output_dir, data_args.retrieval_id)
    
    # add the task name
    output_dir = os.path.join(output_dir, data_args.task_pattern, data_args.task_name)

    # add the retrieval number
    output_dir = os.path.join(output_dir, str(data_args.retrieval_num))

    return output_dir


