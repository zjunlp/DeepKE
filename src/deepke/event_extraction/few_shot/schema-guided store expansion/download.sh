#!/usr/bin/env bash


# Stanford CoreNLP
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
rm -f stanford-corenlp-full-2018-10-05.zip


# IMS (It Makes Sense)
wget http://www.comp.nus.edu.sg/~nlp/sw/IMS_v0.9.2.1.tar.gz
tar xvfz IMS_v0.9.2.1.tar.gz

wget http://www.comp.nus.edu.sg/~nlp/sw/lib.tar.gz
tar xvfz lib.tar.gz
mv lib ims_0.9.2.1

wget http://sterling8.d2.comp.nus.edu.sg/sw/models-MUN-SC-wn30.tar.gz
tar xvfz models-MUN-SC-wn30.tar.gz
mv models-MUN-SC-wn30 ims_0.9.2.1

rm -f IMS_v0.9.2.1.tar.gz
rm -f lib.tar.gz
rm -f models-MUN-SC-wn30.tar.gz

mv ims_0.9.2.1/ims.jar ims_0.9.2.1/lib


# WordNet (for IMS)
wget http://wordnetcode.princeton.edu/3.0/WordNet-3.0.tar.gz
tar xvfz WordNet-3.0.tar.gz
mv WordNet-3.0 ims_0.9.2.1/lib
rm -f WordNet-3.0.tar.gz
