#!/bin/bash
echo "Running los"
for i in $(seq 0 9)
do
  python Argv_Main.py . . . . . . . . . . . $i . . . >> logLos.txt
done
echo "Done!"

