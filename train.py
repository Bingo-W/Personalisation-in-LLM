# the public libs

# the self-defined libs
from data_process import(
    MyDatasets
)

from utils import (
    create_own_argument)



def main():

    # analyse the inputed argument
    data_args, training_args = create_own_argument()
    
    # load the dataset
    my_datasets = MyDatasets(data_args)
    
    # load the model
    

    # ? training code



if __name__ == '__main__':
    main()