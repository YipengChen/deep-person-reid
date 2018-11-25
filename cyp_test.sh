#!/bin/sh
python train_imgreid_xent.py \
--root  ../reid-datasets \
-s market1501 \
-t market1501 \
--height 256 \
--width 128 \
--test-batch-size 100 \
--evaluate \
-a hacnn \
--load-weights log/hacnn_market_xent/hacnn_market_xent.pth.tar \
--save-dir log/hacnn_market_xent \
--gpu-devices 0
