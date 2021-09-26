export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python main.py --num_train_epochs=30 --num_workers=8 \
    --model_name_or_path bert-base-uncased \
    --accumulate_grad_batches 4 \
    --batch_size 8 \
    --data_dir dataset/dialogue \
    --check_val_every_n_epoch 1 \
    --data_class DIALOGUE \
    --max_seq_length 512 \
    --model_class BertForMaskedLM \
    --litmodel_class DialogueLitModel \
    --task_name normal \
    --lr 3e-5 \
    --log_dir ./logs/dialogue.log \
    --save_path ./saved_models