#!/bin/sh
python train_imgreid_xent.py \
--root  ../reid-datasets \
-s cuhk03 \
-t cuhk03 \
--height 160 \
--width 64 \
--test-batch-size 20 \
--evaluate \
-a hacnn \
--load-weights log/hacnn_cuhk03_xent/checkpoint_ep300.pth.tar \
--save-dir log/hacnn_cuhk03_xent \
--use-metric-cuhk03 \
--gpu-devices 0