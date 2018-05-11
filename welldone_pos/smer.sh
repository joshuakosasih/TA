#!/bin/bash
echo "Running mer"
for i in sum mul concat ave
do
  for j in sum mul concat ave
  do
      for k in $(seq 1 6)
      do
        python Argv_Main.py . . . . . $i . . $j $k . . . . . >> logMer.txt
      done
  done
done
echo "Done!"
