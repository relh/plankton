#!/usr/bin/env sh

./build/tools/caffe.bin test \
    -model plankton/train_val.prototxt -weights plankton/planknet_train_iter_10000.caffemodel -gpu 0 -iterations 100

