#!/bin/bash
echo "Running T cut layer"
for i in 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005
do
  for j in 0 2 6 7 11
  do
    for k in $(seq 0 3)
    do
      python Argv_MainTransfer_Cut.py $i $k $j >> logTlayer.txt
    done
  done
done
echo "Done!"

