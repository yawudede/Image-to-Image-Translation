#!/bin/bash
rm data/edges2shoes -rf
mkdir data/edges2shoes -p
mkdir data/edges2shoes/train1 -p
mkdir data/edges2shoes/train0 -p
mkdir data/edges2shoes/test1 -p
mkdir data/edges2shoes/test0 -p
for f in datasets/edges2shoes/train/*; do convert -quality 100 -crop 50%x100% +repage $f data/edges2shoes/train%d/${f##*/}; done;
for f in datasets/edges2shoes/val/*; do convert -quality 100 -crop 50%x100% +repage $f data/edges2shoes/test%d/${f##*/}; done;
mv data/edges2shoes/train0 data/edges2shoes/trainA
mv data/edges2shoes/train1 data/edges2shoes/trainB
mv data/edges2shoes/test0 data/edges2shoes/testA
mv data/edges2shoes/test1 data/edges2shoes/testB
rm datasets -rf