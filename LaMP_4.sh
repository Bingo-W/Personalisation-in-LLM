export CUDA_VISIBLE_DEVICES="1"
python train.py \
    --task_name 'LaMP_4' \
    --task_pattern 'user-based' \
    --training_epoch '20'