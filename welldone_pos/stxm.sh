#!/bin/bash
echo "Running txm"
for i in $(seq 1 3)
do
  for j in $(seq 0 1)
  do
    python Argv_Main.py False . . . . . . $i . . . . . . . >> logTxM.txt
  done
done
echo "Done!"

