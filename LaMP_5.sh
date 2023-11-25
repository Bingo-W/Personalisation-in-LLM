export CUDA_VISIBLE_DEVICES="0"
python train.py \
    --task_name 'LaMP_5' \
    --task_pattern 'user-based' \
    --training_epoch '20'