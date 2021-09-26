export CUDA_VISIBLE_DEVICES=0 
python main.py --max_epochs=30  --num_workers=8 \
    --model_name_or_path bert-large-uncased \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/semeval/k-shot/8-1 \
    --check_val_every_n_epoch 3 \
    --data_class REDataset \
    --max_seq_length 256 \
    --model_class BertForMaskedLM \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --lr 3e-5
    --log_dir ./logs/semeval_k-shot_8-1.log \
    --save_path ./saved_models


export CUDA_VISIBLE_DEVICES=0
python main.py --max_epochs=30  --num_workers=8 \
    --model_name_or_path bert-large-uncased \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/semeval/k-shot/16-1 \
    --check_val_every_n_epoch 3 \
    --data_class REDataset \
    --max_seq_length 256 \
    --model_class BertForMaskedLM \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --lr 3e-5
    --log_dir ./logs/semeval_k-shot_16-1.log \
    --save_path ./saved_models

export CUDA_VISIBLE_DEVICES=0 
python main.py --max_epochs=30  --num_workers=8 \
    --model_name_or_path bert-large-uncased \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/semeval/k-shot/32-1 \
    --check_val_every_n_epoch 3 \
    --data_class REDataset \
    --max_seq_length 256 \
    --model_class BertForMaskedLM \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --lr 3e-5
    --log_dir ./logs/semeval_k-shot_32-1.log \
    --save_path ./saved_models



export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python main.py --max_epochs=5  --num_workers=8 \
    --model_name_or_path bert-large-uncased \
    --accumulate_grad_batches 1 \
    --batch_size 16 \
    --data_dir dataset/semeval \
    --check_val_every_n_epoch 1 \
    --data_class REDataset \
    --max_seq_length 256 \
    --model_class BertForMaskedLM \
    --t_lambda 0.001 \
    --litmodel_class BertLitModel \
    --lr 3e-5
    --log_dir ./logs/semeval.log \
    --save_path ./saved_models