from transformers import HfArgumentParser
from dataclasses import dataclass, field

def create_own_argument():
    parser = HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()
    
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
            'help': 'the path to the raw data folder'
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
            'choices': ('BM25', 'Random')
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