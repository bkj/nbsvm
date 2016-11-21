#!/bin/bash

fasttext supervised \
    -input ../data/ft-train.txt \
    -output delete-me \
    -minCount 10 \
    -minCountLabel 10 \
    -validation ../data/ft-test.txt
