#!/bin/bash
echo "Running final"
for i in $(seq 0 4)
do
  python Argv_Main_Final.py . . . . . mul . . ave 1 . . . . . >> logFinal.txt
done
for i in $(seq 0 4)
do
  python Argv_Main_Final.py . . . . . . . . . . . . . . . >> logFinal.txt
done
echo "Done!"

