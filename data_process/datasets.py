import os
from datasets import load_dataset
from .preprocess_raw_data import PreprocessRawData

class MyDatasets():

    def __init__(self, data_args):
        self._task_name = data_args.task_name
        self._task_pattern = data_args.task_pattern
        self._data_folder_path = data_args.data_folder_path
        
        self._task_path = os.path.join(self._data_folder_path, self._task_pattern, self._task_name)

        if 'LaMP' in self._task_name:
            self._train_input_filename = 'train_questions.json'  
            self._test_input_filename = 'dev_questions.json'
            self._train_output_filename = 'train_outputs.json'
            self._test_output_filename = 'dev_outputs.json'
        else:
            pass

        # load the dataset
        # check if it is need to proprecess the dataset
        if not os.path.exists(self._task_path):
            print("You haven't process the dataset with the specific retrieval methods")
            preprocess_raw_data = PreprocessRawData(data_args)
            del preprocess_raw_data

        assert(os.path.exists(self._task_path))
        self._input_datasets, self._output_datasets = self.__load_dataset()
    
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
            'train': train_input_path,
            'test': test_input_path
        }
        output_dataset = load_dataset("json", data_files=output_datafile)

        return input_dataset, output_dataset
    
    def tokenization(self, tokenizer):
        pass
