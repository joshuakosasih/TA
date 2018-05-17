#!/bin/bash
echo "Running T trainable"
for i in True False
do
  for j in True False
  do
    for k in $(seq 0 2)
    do
      python Argv_MainTransfer.py $k $i $j 05-17_13_37_921 >> logTtrain.txt
    done
  done
done
echo "Done!"

