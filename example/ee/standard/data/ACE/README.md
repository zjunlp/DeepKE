# Data Process For ACE05 Event

1. Prepare data processed from [DyGIE++](https://github.com/dwadden/dygiepp#ace05-event).

2. Put the processed data into the folder `./raw_data`, and the folder should be like:

   ```text
   raw_data
   ├── dev.json
   ├── test.json
   └── train.json
   ```

3. Process the data with following command:

   ```bash
   cd ./preprocess
   bash process.sh
   ```

4. The final folder of ACE dataset should be like:

   ```text
   ACE
   ├── degree # data for degree
   ├── preprocess # preprocess scripts
   ├── raw_data 
   ├── role # data for eae in bertcrf
   ├── schema # schema for bertcrf
   └── trigger # data for ed in bertcrf
   ```

