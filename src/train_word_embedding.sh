#!/bin/bash

DATA_DIR=./data
BIN_DIR=.

TEXT_DATA=$DATA_DIR/en.txt

MAP_DATA=$DATA_DIR/en-wordmap.txt

VECTOR_DATA=$DATA_DIR/results/LMM-A.txt
echo -----------------------------------------------------------------------------------------------------
echo -- Training vectors...
echo $VECTOR_DATA
time $BIN_DIR/lmm-a -train $TEXT_DATA -wordmap $MAP_DATA -output $VECTOR_DATA -cbow 1 -size 200 -window 5 -negative 25 -hs 0 -sample 1e-4 -threads 25 -binary 0 -iter 15 -alpha 0.05
echo -----------------------------------------------------------------------------------------------------

VECTOR_DATA=$DATA_DIR/results/LMM-S.txt
echo -----------------------------------------------------------------------------------------------------
echo -- Training vectors...
echo $VECTOR_DATA
time $BIN_DIR/lmm-s -train $TEXT_DATA -wordmap $MAP_DATA -output $VECTOR_DATA -cbow 1 -size 200 -window 5 -negative 25 -hs 0 -sample 1e-4 -threads 25 -binary 0 -iter 15 -alpha 0.05
echo -----------------------------------------------------------------------------------------------------

VECTOR_DATA=$DATA_DIR/results/LMM-M.txt
echo -----------------------------------------------------------------------------------------------------
echo -- Training vectors...
echo $VECTOR_DATA
time $BIN_DIR/lmm-m -train $TEXT_DATA -wordmap $MAP_DATA -output $VECTOR_DATA -cbow 1 -size 200 -window 5 -negative 25 -hs 0 -sample 1e-4 -threads 25 -binary 0 -iter 15 -alpha 0.05
echo -----------------------------------------------------------------------------------------------------