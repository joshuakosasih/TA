#!/bin/bash
echo "Running oad"
for i in adam adagrad
do
  for j in $(seq 0 1)
  do
    python Argv_Main.py . . . . . . . . . . $i . . . . >> logOAd.txt
  done
done
echo "Done!"

