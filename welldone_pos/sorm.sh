#!/bin/bash
echo "Running orm"
for i in 0.0005 0.002 0.005
do
  for j in $(seq 0 1)
  do
    python Argv_Main_rms.py . . . . . . . . . . $i . . . . >> logORm.txt
  done
done
echo "Done!"

