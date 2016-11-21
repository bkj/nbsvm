#!/bin/bash

python ../nbsvm/nbsvm.py \
    --liblinear ./liblinear-1.96 \
    --train data/ft-train.txt \
    --test data/ft-test.txt \
    --ngram 12 \
    --outpath ./test-results
