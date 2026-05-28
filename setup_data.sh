

# mkdir -p data/laion400m

# kaggle datasets download -d romainbeaumont/laion400m

# mv laion400m.zip data/laion400m/

# unzip data/laion400m/laion400m.zip -d data/laion400m/

# rm data/laion400m/laion400m.zip 

python data/process_subset.py --input-parquet-dir data/laion400m --output-parquet-dir data/laion400m_filtered --target-total 1000000 --rows-per-parquet 250000
