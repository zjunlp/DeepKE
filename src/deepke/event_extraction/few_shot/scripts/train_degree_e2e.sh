export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

DATASET="ace05e"
# DATASET="ace05ep"

python degree/generate_data_degree_e2e.py -c config/config_degree_e2e_$DATASET.json
python degree/train_degree_e2e.py -c config/config_degree_e2e_$DATASET.json
