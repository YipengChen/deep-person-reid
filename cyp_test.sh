#!/bin/sh
python train_imgreid_xent.py \
--root  ../reid-datasets \
-s dukemtmcreid \
-t dukemtmcreid \
--height 160 \
--width 64 \
--test-batch-size 20 \
--evaluate \
-a hacnn \
--load-weights log/hacnn_duke_xent/hacnn_duke_xent.pth.tar \
--save-dir log/hacnn_duke_xent \
--gpu-devices 0
