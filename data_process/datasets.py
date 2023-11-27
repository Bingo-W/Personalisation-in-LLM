import os
from datasets import load_dataset
from retrival_models import AutoRetrieval

class MyDatasets():

    def __init__(self, data_args):
        self._task_name = data_args.task_name
        self._task_pattern = data_args.task_pattern
        self._retrieval_id = data_args.retrieval_id
        self._retrieval_num = data_args.retrieval_num
        self._raw_data_folder_path = data_args.raw_data_folder_path
        self._data_folder_path = data_args.data_folder_path
        self._data_args = data_args
        
        self._task_path = os.path.join(self._raw_data_folder_path, self._task_pattern, self._task_name)

        if 'LaMP' in self._task_name:
            self._train_input_filename = 'train_questions.json' if self._task_name != 'LaMP_3' else 'train_questions.jsonl'
            self._test_input_filename = 'dev_questions.json'
            self._train_output_filename = 'train_outputs.json'
            self._test_output_filename = 'dev_outputs.json'
        else:
            pass

        # load the dataset
        # check if it is need to proprecess the dataset
        # if not os.path.exists(self._task_path):
        #     print("You haven't process the dataset with the specific retrieval methods")
        #     preprocess_raw_data = PreprocessRawData(data_args)
        #     del preprocess_raw_data

        assert(os.path.exists(self._task_path))
        self._datasets = self.__load_dataset()
    
    def __load_dataset(self,):
        """
        the function to load the modified datasets
        """

        train_input_path = os.path.join(self._task_path, self._train_input_filename)
        train_output_path = os.path.join(self._task_path, self._train_output_filename)
        test_input_path = os.path.join(self._task_path, self._test_input_filename)
        test_output_path = os.path.join(self._task_path, self._test_output_filename)

        input_datafile = {
            'train': train_input_path,
            'test': test_input_path
        }
        input_dataset = load_dataset("json", data_files=input_datafile)
        
        output_datafile = {
            'train': train_output_path,
            'test': test_output_path
        }
        output_dataset = load_dataset("json", data_files=output_datafile)

        # merge the input and output tegother
        input_dataset['train'] = input_dataset['train'].add_column('golds', output_dataset['train'][0]['golds'])
        input_dataset['test'] = input_dataset['test'].add_column('golds', output_dataset['test'][0]['golds'])


        return input_dataset
    
    def tokenization(self, tokenizer, training_args):
        """
        Tranform the text dataset into the tokenized dataset

        :param tokenizer: the tokenizer of the pre-training model from transformers library
        """
        
        if self._task_name == 'LaMP_1':
            from .lamp_prompt import LaMP1Prompt as PromptClass
        elif self._task_name == 'LaMP_2':
            from .lamp_prompt import LaMP2Prompt as PromptClass
        elif self._task_name == 'LaMP_3':
            from .lamp_prompt import LaMP3Prompt as PromptClass
        elif self._task_name == 'LaMP_4':
            from .lamp_prompt import LaMP4Prompt as PromptClass
        elif self._task_name == 'LaMP_5':
            from .lamp_prompt import LaMP5Prompt as PromptClass
        elif self._task_name == 'LaMP_6':
            #from .lamp_prompt import LaMP6Prompt as PromptClass
            raise ValueError('We cannot obtain the access to the dataset')
        elif self._task_name == 'LaMP_7':
            from .lamp_prompt import LaMP7Prompt as PromptClass
        else:
            raise Exception(f"These is no available preprocess function for the task.")

        task_max_length = training_args.task_max_length
        input_max_length = training_args.input_max_length
        output_max_length = training_args.output_max_length
        ir_config = {'task_name':self._task_name}
        retrieval_fn = AutoRetrieval.get(self._retrieval_id, ir_config)
        prompt_constructor = PromptClass()

        # modified the input according to the predefined prompt
        def preprocess_function(sample, padding='max_length'):
            
            sample['retrieved_profile'] = [
                retrieval_fn(task_input, user_profile, self._retrieval_num) \
                for task_input, user_profile in zip(sample['input'], sample['profile'])
            ]

            modified_input = [
                prompt_constructor.aggregated_prompt(task_input, retrieved_profile, tokenizer, input_max_length, task_max_length)
                for task_input, retrieved_profile in zip(sample['input'], sample['retrieved_profile'])
            ]
            model_inputs = tokenizer(modified_input, max_length=input_max_length, padding=padding, truncation=True)

            gold_labels = [item['output'] for item in sample['golds']]

            labels = tokenizer(gold_labels, max_length=output_max_length, padding=padding, truncation=True)
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs
        

        processed_datasets = self._datasets.map(
            preprocess_function,
            batched=True,
            batch_size=self._data_args.process_batch_size,
            num_proc=self._data_args.process_num,
            remove_columns=self._datasets['train'].column_names,
            desc="Running tokenizer on dataset"
        )

        return processed_datasets   
