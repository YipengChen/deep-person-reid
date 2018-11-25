#!/bin/sh
python train_imgreid_xent.py \
--root  ../reid-datasets \
-s market1501 \
-t market1501 \
--height 160 \
--width 64 \
--test-batch-size 20 \
--evaluate \
-a hacnn \
--load-weights log/hacnn_market_xent/hacnn_market_xent.pth.tar \
--save-dir log/hacnn_market_xent \
--gpu-devices 0

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


python train_imgreid_xent.py \
--root  ../reid-datasets \
-s market1501 \
-t market1501 \
--height 256 \
--width 128 \
--test-batch-size 20 \
--evaluate \
-a mlfn \
--load-weights log/mlfn_market_xent/mlfn_market_xent.pth.tar \
--save-dir log/mlfn_market_xent \
--gpu-devices 0

python train_imgreid_xent.py \
--root  ../reid-datasets \
-s dukemtmcreid \
-t dukemtmcreid \
--height 256 \
--width 128 \
--test-batch-size 20 \
--evaluate \
-a mlfn \
--load-weights log/mlfn_duke_xent/mlfn_duke_xent.pth.tar \
--save-dir log/mlfn_duke_xent \
--gpu-devices 0


python train_imgreid_xent.py \
--root  ../reid-datasets \
-s market1501 \
-t market1501 \
--height 256 \
--width 128 \
--test-batch-size 20 \
--evaluate \
-a resnet50mid \
--load-weights log/resnet50mid_market_xent/resnet50mid_market_xent.pth.tar \
--save-dir log/resnet50mid_market_xent \
--gpu-devices 0

python train_imgreid_xent.py \
--root  ../reid-datasets \
-s dukemtmcreid \
-t dukemtmcreid \
--height 256 \
--width 128 \
--test-batch-size 20 \
--evaluate \
-a resnet50mid \
--load-weights log/resnet50mid_duke_xent/resnet50mid_duke_xent.pth.tar \
--save-dir log/resnet50mid_duke_xent \
--gpu-devices 0