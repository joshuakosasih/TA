#!/bin/bash
echo "Running default"
for i in $(seq 0 2)
do
  python Argv_Main.py . . . . . . . . . . . . . . . >> logDefault.txt
done
echo "Done!"

