#!/usr/bin/env sh

./build/tools/caffe.bin train \
    --solver=plankton/solver.prototxt #--snapshot=plankton/planknet_train_iter_10000.caffemodel
