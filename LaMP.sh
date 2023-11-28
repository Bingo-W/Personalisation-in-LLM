export CUDA_VISIBLE_DEVICES="0"

for RETRIEVAL_NUM in 4; do
    python train.py \
        --task_name 'LaMP_1' \
        --task_pattern 'user-based' \
        --retrieval_id 'BM25' \
        --retrieval_num ${RETRIEVAL_NUM} \
        --training_epoch '10'

    python train.py \
        --task_name 'LaMP_2' \
        --task_pattern 'user-based' \
        --retrieval_id 'BM25' \
        --retrieval_num ${RETRIEVAL_NUM} \
        --training_epoch '10'

    python train.py \
        --task_name 'LaMP_4' \
        --task_pattern 'user-based' \
        --retrieval_id 'BM25' \
        --retrieval_num ${RETRIEVAL_NUM} \
        --training_epoch '20'

    python train.py \
        --task_name 'LaMP_5' \
        --task_pattern 'user-based' \
        --retrieval_id 'BM25' \
        --retrieval_num ${RETRIEVAL_NUM} \
        --training_epoch '20'

    python train.py \
        --task_name 'LaMP_7' \
        --task_pattern 'user-based' \
        --retrieval_id 'BM25' \
        --retrieval_num ${RETRIEVAL_NUM} \
        --training_epoch '20'

done
