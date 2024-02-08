import os
from datasets import load_dataset, concatenate_datasets
from retrival_models import AutoRetrieval
import random
from collections import OrderedDict

from .utils import list_merge, merge_user_profile, construct_for_llama2

AVERAGE_NUM = OrderedDict(
    [("LaMP_1", 91),
    ("LaMP_2", 159),
    ("LaMP_3", 188),
    ("LaMP_4", 287),
    ("LaMP_5", 90),
    ("LaMP_6", 81),
    ("LaMP_7", 18)]
)

class MyDatasets():

    def __init__(self, data_args):
        self._task_name = data_args.task_name
        self._task_pattern = data_args.task_pattern
        self._retrieval_id = data_args.retrieval_id
        self._retrieval_ablation = data_args.retrieval_ablation
        self._retrieval_target = data_args.retrieval_target
        self._input_retrieval_id = data_args.input_retrieval_id
        self._output_retrieval_id = data_args.output_retrieval_id
        self._random_seed = data_args.retrieval_random_seed
        self._retrieval_num = data_args.retrieval_num
        self._retrieval_order = data_args.retrieval_order
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

        if self._retrieval_id == 'Full_Random' or 'Mixed':
            self._user_pro = self.__compute_user_pro()
    
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
        if self._task_name == 'LaMP_2':
            input_dataset['train'] = input_dataset['train'].add_column('golds', output_dataset['train'])
            input_dataset['test'] = input_dataset['test'].add_column('golds', output_dataset['test'])
        else:
            input_dataset['train'] = input_dataset['train'].add_column('golds', output_dataset['train'][0]['golds'])
            input_dataset['test'] = input_dataset['test'].add_column('golds', output_dataset['test'][0]['golds'])


        return input_dataset
    
    def __compute_user_pro(self):
        """
        compute the sampling probability of all users according to their number of the profile
        """
        user_pro = []
        for user in concatenate_datasets([self._datasets['train'], self._datasets['test']]):
            user_pro.append(len(user['profile']))

        total_num = sum(user_pro)
        user_pro = [item/total_num for item in user_pro]

        return user_pro
    
    def __sample_among_users(self, num_user, batch_size, retrieval_num, retrieval_fn):
        random.seed(self._random_seed)
        num_user = int(num_user)
        random_users = random.choices(concatenate_datasets([self._datasets['train'], self._datasets['test']]), weights=self._user_pro, k=num_user)
        random_user_profles = []
        for index_ in range(batch_size):
            user_profiles_pool = []
            begin_index = int(index_*retrieval_num)
            end_index = int((index_+1)*retrieval_num)
            for random_user in random_users[begin_index: end_index]:
                user_profiles_pool.extend(random_user['profile'])
            
            random_user_profles.append(retrieval_fn(None, user_profiles_pool, retrieval_num))

        return random_user_profles

    def tokenization(self, tokenizer, training_args):
        """
        Tranform the text dataset into the tokenized dataset

        :param tokenizer: the tokenizer of the pre-training model from transformers library
        """
        
        
        if self._task_name == 'LaMP_1':
            from .lamp_prompt import LaMP1Prompt as PromptClass
        elif self._task_name == 'LaMP_2':
            if self._retrieval_ablation == 'both':
                from .lamp_prompt import LaMP2Prompt as PromptClass
            elif self._retrieval_ablation == 'only_output' or self._retrieval_ablation == 'decouple':
                from .lamp_prompt_ablation import LaMP2PromptAblation as PromptClass
            elif self._retrieval_ablation == 'only_input':
                from .lamp_prompt_ablation import LaMP2PromptInput as PromptClass
            else:
                raise ValueError('No Implements')
        elif self._task_name == 'LaMP_3':
            if self._retrieval_ablation == 'both':
                from .lamp_prompt import LaMP3Prompt as PromptClass
            elif self._retrieval_ablation == 'only_output'or self._retrieval_ablation == 'decouple':
                from.lamp_prompt_ablation import LaMP3PromptAblation as PromptClass
            elif self._retrieval_ablation == 'only_input':
                from .lamp_prompt_ablation import LaMP3PromptInput as PromptClass
            else:
                raise ValueError('No Implements')
        elif self._task_name == 'LaMP_4':
            if self._retrieval_ablation == 'both':
                from .lamp_prompt import LaMP4Prompt as PromptClass
            elif self._retrieval_ablation == 'only_output' or self._retrieval_ablation == 'decouple':
                from.lamp_prompt_ablation import LaMP4PromptAblation as PromptClass
            elif self._retrieval_ablation == 'only_input':
                from .lamp_prompt_ablation import LaMP4PromptInput as PromptClass
            else:
                raise ValueError('No Implements')
        elif self._task_name == 'LaMP_5':
            if self._retrieval_ablation == 'both':
                from .lamp_prompt import LaMP5Prompt as PromptClass
            elif self._retrieval_ablation == 'only_output' or self._retrieval_ablation == 'decouple':
                from.lamp_prompt_ablation import LaMP5PromptAblation as PromptClass
            elif self._retrieval_ablation == 'only_input':
                from .lamp_prompt_ablation import LaMP5PromptInput as PromptClass
            else:
                raise ValueError('No Implements')
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
        ir_config = {
            'task_name':self._task_name,
            'random_seed': self._random_seed,
            'input_retrieval_id': self._input_retrieval_id,
            'output_retrieval_id': self._output_retrieval_id,
        }
        retrieval_fn = AutoRetrieval.get(self._retrieval_id, ir_config)
        prompt_constructor = PromptClass()

        # modified the input according to the predefined prompt
        def preprocess_function(sample, padding='max_length'):
            
            if self._retrieval_num != 0:
                if self._retrieval_id == 'Full_Random':
                    # for random sample
                    batch_size = len(sample['input'])
                    random_user_profles = self.__sample_among_users(batch_size*self._retrieval_num, batch_size, self._retrieval_num, retrieval_fn)

                    sample['retrieved_profile'] = random_user_profles
                elif self._retrieval_id == 'RanBM25':
                    batch_size = len(sample['input'])
                    sample_retrie_fn = AutoRetrieval.get('Random', ir_config)
                    random_user_profles = self.__sample_among_users(batch_size*int(AVERAGE_NUM[self._task_name]), batch_size, AVERAGE_NUM[self._task_name], sample_retrie_fn)

                    sample['retrieved_profile'] = [
                        retrieval_fn(task_input, random_user_profle, self._retrieval_num, self._retrieval_ablation, self._retrieval_target) \
                        for task_input, random_user_profle in zip(sample['input'], random_user_profles)
                    ]
                elif self._retrieval_id == 'Mixed':
                    # for the mixed user profile
                    batch_size = len(sample['input'])
                    if self._input_retrieval_id == 'Full_Random':
                        profiles_for_input = self.__sample_among_users(batch_size*self._retrieval_num, batch_size, retrieval_fn['input'])
                    else:
                        profiles_for_input = [
                        retrieval_fn['input'](task_input, user_profile, self._retrieval_num) \
                        for task_input, user_profile in zip(sample['input'], sample['profile'])
                    ]
                    if self._output_retrieval_id == 'Full_Random':
                        profiles_for_output = self.__sample_among_users(batch_size*self._retrieval_num, batch_size, retrieval_fn['output'])
                    else:
                        profiles_for_output = [
                        retrieval_fn['output'](task_input, user_profile, self._retrieval_num) \
                        for task_input, user_profile in zip(sample['input'], sample['profile'])
                    ]
                    sample['retrieved_profile'] = merge_user_profile(profiles_for_input, profiles_for_output, self._task_name)    
                    
                elif self._retrieval_id == 'Random':
                    # for personalisation or context-aware personalisation
                    sample['retrieved_profile'] = [
                        retrieval_fn(task_input, user_profile, self._retrieval_num) \
                        for task_input, user_profile in zip(sample['input'], sample['profile'])
                    ]
                else:
                    # for personalisation or context-aware personalisation
                    sample['retrieved_profile'] = [
                        retrieval_fn(task_input, user_profile, self._retrieval_num, self._retrieval_ablation, self._retrieval_target) \
                        for task_input, user_profile in zip(sample['input'], sample['profile'])
                    ]

                modified_input = [
                    prompt_constructor.aggregated_prompt(task_input, retrieved_profile, tokenizer, input_max_length, task_max_length, self._retrieval_order, self._random_seed)
                    for task_input, retrieved_profile in zip(sample['input'], sample['retrieved_profile'])
                ]
            else:
                # for non-personalisation
                modified_input = [
                    item for item in sample['input']
                ]
            
            model_inputs = tokenizer(modified_input, max_length=input_max_length, padding=padding, truncation=True)

            gold_labels = [item['output'] for item in sample['golds']]

            labels = tokenizer(gold_labels, max_length=output_max_length, padding=padding, truncation=True)
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]
            model_inputs['modified_input'] = modified_input

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

class LlamaDatasets(MyDatasets):

    def __init__(self, data_args):
        super(LlamaDatasets,self).__init__(data_args)
        self.__sample_among_users = self._MyDatasets__sample_among_users
    
    def tokenization(self, tokenizer, training_args):
        """
        Tranform the text dataset into the tokenized dataset

        :param tokenizer: the tokenizer of the pre-training model from transformers library
        """
        
        
        if self._task_name == 'LaMP_1':
            from .lamp_prompt import LaMP1Prompt as PromptClass
        elif self._task_name == 'LaMP_2':
            if self._retrieval_ablation == 'both':
                from .lamp_prompt import LaMP2Prompt as PromptClass
            elif self._retrieval_ablation == 'only_output' or self._retrieval_ablation == 'decouple':
                from .lamp_prompt_ablation import LaMP2PromptAblation as PromptClass
            elif self._retrieval_ablation == 'only_input':
                from .lamp_prompt_ablation import LaMP2PromptInput as PromptClass
            else:
                raise ValueError('No Implements')
        elif self._task_name == 'LaMP_3':
            if self._retrieval_ablation == 'both':
                from .lamp_prompt import LaMP3Prompt as PromptClass
            elif self._retrieval_ablation == 'only_output'or self._retrieval_ablation == 'decouple':
                from.lamp_prompt_ablation import LaMP3PromptAblation as PromptClass
            elif self._retrieval_ablation == 'only_input':
                from .lamp_prompt_ablation import LaMP3PromptInput as PromptClass
            else:
                raise ValueError('No Implements')
        elif self._task_name == 'LaMP_4':
            if self._retrieval_ablation == 'both':
                from .lamp_prompt import LaMP4Prompt as PromptClass
            elif self._retrieval_ablation == 'only_output' or self._retrieval_ablation == 'decouple':
                from.lamp_prompt_ablation import LaMP4PromptAblation as PromptClass
            elif self._retrieval_ablation == 'only_input':
                from .lamp_prompt_ablation import LaMP4PromptInput as PromptClass
            else:
                raise ValueError('No Implements')
        elif self._task_name == 'LaMP_5':
            if self._retrieval_ablation == 'both':
                from .lamp_prompt import LaMP5Prompt as PromptClass
            elif self._retrieval_ablation == 'only_output' or self._retrieval_ablation == 'decouple':
                from.lamp_prompt_ablation import LaMP5PromptAblation as PromptClass
            elif self._retrieval_ablation == 'only_input':
                from .lamp_prompt_ablation import LaMP5PromptInput as PromptClass
            else:
                raise ValueError('No Implements')
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
        ir_config = {
            'task_name':self._task_name,
            'random_seed': self._random_seed,
            'input_retrieval_id': self._input_retrieval_id,
            'output_retrieval_id': self._output_retrieval_id,
        }
        retrieval_fn = AutoRetrieval.get(self._retrieval_id, ir_config)
        prompt_constructor = PromptClass()

        # modified the input according to the predefined prompt
        def preprocess_function(sample, padding='max_length'):
            
            if self._retrieval_num != 0:
                if self._retrieval_id == 'Full_Random':
                    # for random sample
                    batch_size = len(sample['input'])
                    random_user_profles = self.__sample_among_users(batch_size*self._retrieval_num, batch_size, self._retrieval_num, retrieval_fn)

                    sample['retrieved_profile'] = random_user_profles
                elif self._retrieval_id == 'RanBM25':
                    batch_size = len(sample['input'])
                    sample_retrie_fn = AutoRetrieval.get('Random', ir_config)
                    random_user_profles = self.__sample_among_users(batch_size*int(AVERAGE_NUM[self._task_name]), batch_size, AVERAGE_NUM[self._task_name], sample_retrie_fn)

                    sample['retrieved_profile'] = [
                        retrieval_fn(task_input, random_user_profle, self._retrieval_num, self._retrieval_ablation, self._retrieval_target) \
                        for task_input, random_user_profle in zip(sample['input'], random_user_profles)
                    ]
                elif self._retrieval_id == 'Mixed':
                    # for the mixed user profile
                    batch_size = len(sample['input'])
                    if self._input_retrieval_id == 'Full_Random':
                        profiles_for_input = self.__sample_among_users(batch_size*self._retrieval_num, batch_size, retrieval_fn['input'])
                    else:
                        profiles_for_input = [
                        retrieval_fn['input'](task_input, user_profile, self._retrieval_num) \
                        for task_input, user_profile in zip(sample['input'], sample['profile'])
                    ]
                    if self._output_retrieval_id == 'Full_Random':
                        profiles_for_output = self.__sample_among_users(batch_size*self._retrieval_num, batch_size, retrieval_fn['output'])
                    else:
                        profiles_for_output = [
                        retrieval_fn['output'](task_input, user_profile, self._retrieval_num) \
                        for task_input, user_profile in zip(sample['input'], sample['profile'])
                    ]
                    sample['retrieved_profile'] = merge_user_profile(profiles_for_input, profiles_for_output, self._task_name)    
                    
                elif self._retrieval_id == 'Random':
                    # for personalisation or context-aware personalisation
                    sample['retrieved_profile'] = [
                        retrieval_fn(task_input, user_profile, self._retrieval_num) \
                        for task_input, user_profile in zip(sample['input'], sample['profile'])
                    ]
                else:
                    # for personalisation or context-aware personalisation
                    sample['retrieved_profile'] = [
                        retrieval_fn(task_input, user_profile, self._retrieval_num, self._retrieval_ablation, self._retrieval_target) \
                        for task_input, user_profile in zip(sample['input'], sample['profile'])
                    ]

                modified_input = [
                    prompt_constructor.aggregated_prompt(task_input, retrieved_profile, tokenizer, input_max_length, task_max_length, self._retrieval_order, self._random_seed)
                    for task_input, retrieved_profile in zip(sample['input'], sample['retrieved_profile'])
                ]
            else:
                # for non-personalisation
                modified_input = sample['input']

            model_inputs = {}
            model_inputs["labels"] = [item['output'] for item in sample['golds']]
            if self._task_name == 'LaMP_1' or self._task_name == 'LaMP_2' or self._task_name == 'LaMP_3':
                model_inputs['input'] = [
                        tokenizer.decode(tokenizer(item, max_length=input_max_length, truncation=True)['input_ids']) for item in modified_input
                ]
            else:
                model_inputs['input'] = [
                        tokenizer.decode(tokenizer(self.llama_prompt(item), max_length=input_max_length, truncation=True)['input_ids']) for item in modified_input
                ]

            return model_inputs
        

        processed_datasets = self._datasets['test'].map(
            preprocess_function,
            batched=True,
            batch_size=self._data_args.process_batch_size,
            num_proc=self._data_args.process_num,
            remove_columns=self._datasets['train'].column_names,
            desc="Running tokenizer on dataset"
        )

        return processed_datasets   
    
    def llama_prompt(self, modified_input):

        # choose the anchor
        anchor = ""
        if self._task_name == 'LaMP_4':
            anchor = 'the following article:'
        elif self._task_name == 'LaMP_5':
            anchor = 'the following abstract of a paper:'
        elif self._task_name == 'LaMP_7':
            anchor = 'the following tweet without any explanation before or after it:'
        
        # precoss the input
        split_text = modified_input.split(anchor)
        res = split_text[0] + " " + anchor + " \""+split_text[1]+"\""

        return res