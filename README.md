# Language translation task

This repo is a small environment for experiments with Machine Translation.  
An architecture of seq2seq model is configurable.  

## Pre-run requirements

`pip install -r requirements.txt`

Please, install next spacy dependencies, if you plan to use word-tokenizers
- `python -m spacy download en_core_web_sm`
- `python -m spacy download ru_core_news_sm`

Download glove (if you would like to use it) and place it in a folder with `main.py`
- `wget -O glove.6B.zip http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip`

## Run

`python main.py train --experiment_name NAME --architecture_path configs/rnn2rnn_architecture.yaml --learning_path configs/train_config.yaml`

## TODO:

- configurable conv nets; currently not (only name-based presets)  

## Ignored files 
Because they are large:

- glove.6B.zip
- glove_embeds.pt
- 
