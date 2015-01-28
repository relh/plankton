#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./build/tools/compute_image_mean plankton/plankton_train_lmdb \
  plankton/planknet_mean.binaryproto

echo "Done."
