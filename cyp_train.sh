#!/bin/sh
python train_imgreid_xent.py \
--root data \
-s market1501 \
-t market1501 \
-j 4 \
--height 160 \
--width 64 \
--optim sgd \
--lr 0.03 \
--label-smooth \
--max-epoch 300 \
--stepsize 150 225 \
--train-batch-size 32 \
--test-batch-size 100 \
-a hacnn \
--save-dir log/hacnn-market-xent \
--gpu-devices 0 \
