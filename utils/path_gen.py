import os

def output_dir_generation(data_args, training_args):

    model_name = training_args.model_id.split('/')[1]
    
    # add the model name
    output_dir = os.path.join(training_args.result_dir, model_name)

    # add the retrieval name
    output_dir = os.path.join(output_dir, data_args.retrieval_id)
    if data_args.retrieval_id =='Mixed':
        output_dir = os.path.join(output_dir, data_args.input_retrieval_id+'_'+data_args.output_retrieval_id)
    
    # add the task name
    output_dir = os.path.join(output_dir, data_args.task_pattern, data_args.task_name)

    # add the retrieval number
    if data_args.retrieval_num < 1 and data_args.retrieval_num>0:
        output_dir = os.path.join(output_dir, str(data_args.retrieval_num))
    else:
        output_dir = os.path.join(output_dir, str(int(data_args.retrieval_num)))

    if data_args.retrieval_id == 'Random' or data_args.retrieval_id == 'Full_Random':
        output_dir = os.path.join(output_dir, str(data_args.retrieval_random_seed))
    elif data_args.retrieval_id =='Mixed':
        if data_args.input_retrieval_id in ['Random', 'Full_Random'] or data_args.output_retrieval_id in ['Random', 'Full_Random']:
            output_dir = os.path.join(output_dir, str(data_args.retrieval_random_seed))
    return output_dir


def checkpoint_file_generation(output_dir):

    if not os.path.exists(output_dir):
        raise ValueError("The result for this task does not exist.")
    all_files = os.listdir(output_dir)
    checkpoints_file = []

    # filter out the non-checkpoint
    for item in all_files:
        if 'checkpoint' in item:
            checkpoints_file.append(item)

    # find the best checkpoint
    best_step = int(checkpoints_file[0].split('-')[-1])
    best_checkpoint = checkpoints_file[0]

    for checkpoins in checkpoints_file:
        if int(checkpoins.split('-')[-1]) < best_step:
            best_checkpoint = checkpoins
            best_step = int(checkpoins.split('-')[-1])
    
    return os.path.join(output_dir, best_checkpoint)
