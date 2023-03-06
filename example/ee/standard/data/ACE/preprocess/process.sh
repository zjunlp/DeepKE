export DYGIEFORMAT_PATH="../raw_data"
export OUTPUT_PATH="../degree"

mkdir -p $OUTPUT_PATH

python preprocessing/process_degree.py -i $DYGIEFORMAT_PATH/train.json -o $OUTPUT_PATH/train.w1.oneie.json -b facebook/bart-large -w 1
python preprocessing/process_degree.py -i $DYGIEFORMAT_PATH/dev.json -o $OUTPUT_PATH/dev.w1.oneie.json -b facebook/bart-large -w 1
python preprocessing/process_degree.py -i $DYGIEFORMAT_PATH/test.json -o $OUTPUT_PATH/test.w1.oneie.json -b facebook/bart-large -w 1

export BASE_PATH="../degree"
export SPLIT_PATH="./low_resource_split"

python preprocessing/split_dataset.py -i $BASE_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_001 -o $BASE_PATH/train.001.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_002 -o $BASE_PATH/train.002.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_003 -o $BASE_PATH/train.003.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_005 -o $BASE_PATH/train.005.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_010 -o $BASE_PATH/train.010.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_020 -o $BASE_PATH/train.020.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_030 -o $BASE_PATH/train.030.w1.oneie.json    
python preprocessing/split_dataset.py -i $BASE_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_050 -o $BASE_PATH/train.050.w1.oneie.json      
python preprocessing/split_dataset.py -i $BASE_PATH/train.w1.oneie.json -s $SPLIT_PATH/doc_list_075 -o $BASE_PATH/train.075.w1.oneie.json


export BERT_CRF_PATH="../"

mkdir -p $BERT_CRF_PATH

python preprocessing/process_bertcrf.py --base_path $BASE_PATH --bertcrf_path $BERT_CRF_PATH