#!/bin/bash
echo "Running edo"
for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  for j in $(seq 0 1)
  do
    python Argv_Main.py . . . . $i . . . . . . . . . . >> logEDo.txt
  done
done
echo "Done!"

