export CUDA_VISIBLE_DEVICES="1"

RETRIEVAL_ID='Full_Random'

python train.py \
    --task_name 'LaMP_1' \
    --task_pattern 'user-based' \
    --retrieval_id ${RETRIEVAL_ID} \
    --training_epoch '10'

python train.py \
    --task_name 'LaMP_2' \
    --task_pattern 'user-based' \
    --retrieval_id ${RETRIEVAL_ID} \
    --training_epoch '10'

python train.py \
    --task_name 'LaMP_4' \
    --task_pattern 'user-based' \
    --retrieval_id ${RETRIEVAL_ID} \
    --training_epoch '20'

python train.py \
    --task_name 'LaMP_5' \
    --task_pattern 'user-based' \
    --retrieval_id ${RETRIEVAL_ID} \
    --training_epoch '20'

python train.py \
    --task_name 'LaMP_7' \
    --task_pattern 'user-based' \
    --retrieval_id ${RETRIEVAL_ID} \
    --training_epoch '20'

