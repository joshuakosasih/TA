#!/bin/bash
echo "Running NO T cut"
for i in 1.0 0.5 0.2 0.1 0.05 0.02 0.01 0.005
do
  for j in $(seq 0 14)
  do
    python Argv_Main.py $i $j >> logNoT.txt
  done
done
echo "Done!"

