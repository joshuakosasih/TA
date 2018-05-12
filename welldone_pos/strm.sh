#!/bin/bash
echo "Running trm"
for i in 64 126
do
  for j in $(seq 0 1)
  do
    python Argv_Main.py . . . $i . . . . . . . . . . . >> logTrm.txt
  done
done
echo "Done!"

