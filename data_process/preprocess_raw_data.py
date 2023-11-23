import os
from datasets import load_dataset
from .utils import extract_quote

class PreprocessRawData():
    def __init__(self, data_args):
        
        self._retrieval_model = data_args.retrieval_id
        self._task_name = data_args.task_name
        self._task_pattern = data_args.task_pattern
        self._raw_data_folder_path = data_args.raw_data_folder_path
        self._data_folder_path = data_args.data_folder_path

        self._raw_task_path = os.path.join(self._raw_data_folder_path, self._task_pattern, self._task_name)
        self._task_path = os.path.join(self._data_folder_path, self._task_pattern, self._task_name)

        if 'LaMP' in self._task_name:
            self._train_input_filename = 'train_questions.json'  
            self._test_input_filename = 'dev_questions.json'
            self._train_output_filename = 'train_outputs.json'
            self._test_output_filename = 'dev_outputs.json'
        else:
            pass
        
        self.input_dataset = self.__load_raw_dataset()
        
        self.preprocess_data()

        self.save_datasets()

    def __del__(self):
        print('This proprecess class is deleted to save memory.')

    def preprocess_data(self):
        

        def retrieved_userprofile(example, metrics=None):

            # extract the words from the non-template part of the sentence
            query = [item for i in extract_quote(example['input']) for item in i.split()]
            
            # exclude the profile related to the input text
            input_title = extract_quote(example['input'])[0]
            for i, userprofile in enumerate(example['profile']):
                if userprofile['title'] == input_title:
                    example['profile'].pop(i)
                    break

            # extract the user profile
            user_profile_corpus = [item['abstract'].split()+item['title'].split() for item in example['profile']]
            
            # compute the scores
            scores = [metrics(query, item, user_profile_corpus) for item in user_profile_corpus]
            retrieved_index = scores.index(max(scores))
            example['retrieved_profile'] = [example['profile'][retrieved_index]]


            return example
        
        from retrival_models import BM25
        metrics = BM25.bm25_score
        self.modified_dataset = self.input_dataset.map(lambda x: retrieved_userprofile(x,metrics=metrics), num_proc=24, remove_columns=['profile'])

        print('The keys after processing are: {}'.format(self.modified_dataset['train'][0].keys()))

    
    def save_datasets(self):
        
        os.makedirs(self._task_path)
        train_input_path = os.path.join(self._task_path, self._train_input_filename)
        test_input_path = os.path.join(self._task_path, self._test_input_filename)
        # save the input datasets
        for split, dataset in self.modified_dataset.items():
            if split == 'train':
                dataset.to_json(train_input_path)
            elif split == 'test':
                dataset.to_json(test_input_path)
            else:
                raise Exception(f"Un-recognized split name {split}.")
        print('Save the input file successfully.')

        # save the output file
        train_output_path = os.path.join(self._raw_task_path, self._train_output_filename)
        test_output_path = os.path.join(self._raw_task_path, self._test_output_filename)

        os.system('cp {} {}'.format(train_output_path, self._task_path))
        os.system('cp {} {}'.format(test_output_path, self._task_path))
        print('Save the output file successfully.')
    
    def __load_raw_dataset(self):
        self._raw_task_path = os.path.join(self._raw_data_folder_path, self._task_pattern, self._task_name)
        assert (os.path.exists(self._raw_task_path))
        train_input_path = os.path.join(self._raw_task_path, self._train_input_filename)
        test_input_path = os.path.join(self._raw_task_path, self._test_input_filename)
        

        input_data_file = {
            "train": train_input_path,
            "test": test_input_path
        }
        print(input_data_file)
        input_datasets = load_dataset("json", data_files=input_data_file)

        return input_datasets

    