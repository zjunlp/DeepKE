export DYGIEFORMAT_PATH="./raw_data/ace05e_dygieppformat"
export OUTPUT_PATH="./processed/ace05e_bart"

mkdir -p $OUTPUT_PATH

python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/train.json -o $OUTPUT_PATH/train.w1.oneie.json -b facebook/bart-large -w 1
python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/dev.json -o $OUTPUT_PATH/dev.w1.oneie.json -b facebook/bart-large -w 1
python preprocessing/process_ace05e.py -i $DYGIEFORMAT_PATH/test.json -o $OUTPUT_PATH/test.w1.oneie.json -b facebook/bart-large -w 1

export BASE_PATH="./processed/"
export SPLIT_PATH="./resource/low_resource_split/ace05e"

python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_001 -o $BASE_PATH/ace05e_bart/train.001.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_002 -o $BASE_PATH/ace05e_bart/train.002.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_003 -o $BASE_PATH/ace05e_bart/train.003.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_005 -o $BASE_PATH/ace05e_bart/train.005.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_010 -o $BASE_PATH/ace05e_bart/train.010.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_020 -o $BASE_PATH/ace05e_bart/train.020.w1.oneie.json
python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_030 -o $BASE_PATH/ace05e_bart/train.030.w1.oneie.json    
python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_050 -o $BASE_PATH/ace05e_bart/train.050.w1.oneie.json      
python preprocessing/split_dataset.py -i $BASE_PATH/ace05e_bart/train.w1.oneie.json -s $SPLIT_PATH/doc_list_075 -o $BASE_PATH/ace05e_bart/train.075.w1.oneie.json
