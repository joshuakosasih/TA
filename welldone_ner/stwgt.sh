#!/bin/bash
echo "Running T wgt"
for i in 05-17_13_37_921 
do
  for k in $(seq 0 2)
  do
    python Argv_MainTransfer.py $k True True $i >> logTwgt.txt
  done
done
echo "Done!"

