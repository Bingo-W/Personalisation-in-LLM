from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import Optional

def create_own_argument():
    parser = HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    
    # check the hyperparameter
    if training_args.model_id == 'google/flan-t5-xxl':
        training_args.do_train = False
        training_args.do_eval = True

    return data_args, training_args

@dataclass
class DataArguments:
    '''
    Arguments pertaining to the data

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    '''
    task_name: str = field(
        default='All',
        metadata = {
            'help': 'the name of the task for fine-tuning', 
            'choices': ('LaMP_1', 'LaMP_2', 'LaMP_3', 'LaMP_4', 'LaMP_5', 'LaMP_6', 'LaMP_7')
        }
    )

    task_pattern: str = field(
        default='user-based',
        metadata = {
            'help': 'the task split for LaMP benchmark',
            'choices': ('user-based', 'time-based')
        }
    )

    raw_data_folder_path: str = field(
        default='../../data/LaMP',
        metadata={
            'help': 'the path to the raw data foldeß'
        }
    )

    data_folder_path: str = field(
        default='../personalised_results/data/LaMP',
        metadata={
            'help': 'the path to the modified data folder'
        }
    )

    retrieval_id : str = field(
        default='BM25',
        metadata={
            'help': 'the retrieval method for user profiles',
            'choices': ('BM25', 'Random', 'Full_Random', 'Mixed')
        }
    )

    input_retrieval_id : str = field(
        default='Random',
        metadata={
            'help': 'the retrieval method for the input part of user profiles',
            'choices': ('BM25', 'Random', 'Full_Random', 'Mixed')
        }
    )

    input_retrieval_id : str = field(
        default='Random',
        metadata={
            'help': 'the retrieval method for the input part of user profiles',
            'choices': ('BM25', 'Random', 'Full_Random')
        }
    )

    output_retrieval_id : str = field(
        default='Random',
        metadata={
            'help': 'the retrieval method for the output part of user profiles',
            'choices': ('BM25', 'Random', 'Full_Random')
        }
    )

    retrieval_random_seed: int = field(
        default=1,
        metadata={
            'help': 'the random seed to have a constant and reproduceable result.'
        }
    )

    retrieval_num: int = field(
        default=1,
        metadata={
            'help': 'the number of the retrieved user profiles'
        }
    )

    process_batch_size: int = field(
        default=1000,
        metadata={
            'help': 'the hyparameter for debugging'
        }
    )

    process_num: int = field(
        default=24,
        metadata={
            'help': 'the hyparameter for debugging'
        }
    )

@dataclass
class TrainingArguments:
    '''
    Arguments pertaining to the fine-tuning stage

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    '''
    model_id : str = field(
        default = 'google/flan-t5-base',
        metadata = {
            'help': 'the Huggingface ID of the pre-trained model'
        }
    )

    model_checkpoint : Optional[str] = field(
        default=None,
        metadata={
            'help': 'the name of the checkpoint'
        }
    )

    task_max_length: int = field(
        default=256,
        metadata={
            'help': 'the max length of the task input'
        }
    )
    input_max_length: int = field(
        default=512,
        metadata={
            'help': 'the max length of the input tokens sequence'
        }
    )

    output_max_length: int = field(
        default=512,
        metadata={
            'help': 'the max length of the output tokens sequence'
        }
    )

    result_dir: str = field(
        default='../personalised_results/results'
    )

    training_epoch: int = field(
        default=10,
    )

    do_train: bool = field(
        default=True,
        metadata={
            'help': 'denoting whether it need training'
        }
    )

    do_eval: bool = field(
        default=False,
        metadata={
            'help': 'denoting whether it need evaluation'
        }
    )

    do_predict: bool = field(
        default=False,
        metadata={
            'help': 'denoting whether it need prediction'
        }
    )
