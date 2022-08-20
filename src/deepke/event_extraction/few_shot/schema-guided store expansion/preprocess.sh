#!/usr/bin/env bash


 Stanford CoreNLP
 for d in data/Wiki/*; do
     java -Xmx8g -cp "stanford-corenlp-full-2018-10-05/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -props resources/corenlp.properties -file $d -extension .txt -threads 4 -outputFormat json -outputDirectory $d
 done


# NLTK -- WordNet
 python -m nltk.downloader wordnet


# Polyglot
polyglot download morph2.en


# # Run preprocessing
python preprocess.py
