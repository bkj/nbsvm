"""
    nbsvm.py
    
    python nbsvm.py --liblinear /PATH/liblinear-1.96 \
        --ptrain /PATH/data/full-train-pos.txt \
        --ntrain /PATH/data/full-train-neg.txt \
        --ptest /PATH/data/test-pos.txt \
        --ntest /PATH/data/test-neg.txt \
        --ngram 123 \
        --out TEST-SCORE    
"""

import os
import sys
import numpy as np
import argparse
from collections import Counter
import numba

def tokenize(sentence, grams):
    words = sentence.split()
    for gram in grams:
        for i in range(len(words) - gram + 1):
            yield "_*_".join(words[i:i+gram])


def process_files(path, ulab, lookup, rat, outpath, grams):
    print >> sys.stderr, 'creating %s' % outpath
    
    outfile = open(outpath, 'w')
    for i,line in enumerate(open(path).xreadlines()):
        if not i % 10000:
            sys.stderr.write('\r\t Processed %s lines' % i)
            sys.stderr.flush()
        
        lab, val = line.strip().split('\t')
        tokens = tokenize(val, grams)       
         
        indices = [lookup[t] for t in tokens if lookup.get(t, False)]
        indices = sorted(list(set(indices)))
        
        line = [ulab[lab]] + ["%i:%f" % (i + 1, rat[i]) for i in indices]
        outfile.write(" ".join(line) + "\n")
    
    print
    outfile.close()


def compute_ratio(poscounts, negcounts, alpha=1):
    utokens = list(set(poscounts.keys() + negcounts.keys()))
    lookup = dict(zip(utokens, range(len(utokens))))
    
    p = np.ones(len(utokens)) * alpha
    q = np.ones(len(utokens)) * alpha
    
    for t in utokens:
        p[lookup[t]] += poscounts[t]
        q[lookup[t]] += negcounts[t]
        
    p /= abs(p).sum()
    q /= abs(q).sum()
    
    return lookup, np.log(p / q)


def parse_args():
    parser = argparse.ArgumentParser(description='Run NB-SVM on some text files.')
    parser.add_argument('--liblinear', help='path of liblinear install e.g. */liblinear-1.96', required=True)
    parser.add_argument('--train', help='train data', required=True)
    parser.add_argument('--test', help='test data', required=True)
    parser.add_argument('--outpath', help='path and filename for score output', default='./nbsvm-model')
    parser.add_argument('--ngram', help='N-grams considered e.g. 123 is uni+bi+tri-grams', default='123')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    grams = map(int, args.ngram)
    
    print >> sys.stderr, 'counting words:'
    counters = {}
    ulab = set([])
    for i,line in enumerate(open(args.train).xreadlines()):
        if not i % 10000:
            sys.stderr.write('\r\t Counted \t%s lines' % i)
            sys.stderr.flush()
        
        lab, val = line.split('\t')
        
        ulab.add(lab)
        
        if lab not in counters:
            counters[lab] = Counter()
        
        for tok in tokenize(val, grams):
            counters[lab][tok] += 1
    
    if len(ulab) > 2:
        raise Exception('this implementation only supports binary classification!')
    else:
        ulab = dict(zip(ulab, ['-1', '+1']))
    
    print >> sys.stderr, '\n writing files:'
    lookup, rat = compute_ratio(*counters.values())
    process_files(args.train, ulab, lookup, rat, "./train-nbsvm.txt", grams)
    process_files(args.test, ulab, lookup, rat, "./test-nbsvm.txt", grams)
    
    os.system("%s/train -s 0 ./train-nbsvm.txt %s" % (args.liblinear, args.outpath))
    os.system("%s/predict -b 1 ./test-nbsvm.txt %s .nbsvm-tmp" % (args.liblinear, args.outpath))
    # os.system("rm .nbsvm-tmp ./train-nbsvm.txt ./test-nbsvm.txt")
