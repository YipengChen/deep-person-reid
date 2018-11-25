#!/bin/sh
python train_imgreid_xent.py \
--root  ../reid-datasets \
-s cuhk03 \
-t cuhk03 \
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
--save-dir log/hacnn-cuhk03-xent \
--gpu-devices 0


python train_imgreid_xent.py \
--root  ../reid-datasets \
-s cuhk03 \
-t cuhk03 \
-j 4 \
--height 256 \
--width 128 \
--optim amsgrad \
--lr 0.0003 \
--label-smooth \
--max-epoch 60 \
--stepsize 20 40 \
--fixbase-epoch 10 \
--open-layers classifier \
--train-batch-size 32 \
--test-batch-size 100 \
-a mlfn \
--save-dir log/mlfn-cuhk03-xent \
--gpu-devices 0


python train_imgreid_xent.py \
--root  ../reid-datasets \
-s cuhk03 \
-t cuhk03 \
-j 4 \
--height 256 \
--width 128 \
--optim amsgrad \
--label-smooth \
--lr 0.0003 \
--max-epoch 60 \
--stepsize 20 40 \
--fixbase-epoch 10 \
--open-layers classifier fc_fusion \
--train-batch-size 32 \
--test-batch-size 100 \
-a resnet50mid \
--save-dir log/resnet50mid-cuhk03-xent \
--gpu-devices 0