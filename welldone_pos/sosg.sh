#!/bin/bash
echo "Running osg"
python Argv_Main_sgd.py . . . . . . . . . . . . . . . >> logOSg.txt
python Argv_Main_sgd.py . . . . . . . . . . T . . . . >> logOSg.txt
python Argv_Main_sgd.py . . . . . . . . . . . C . . . >> logOSg.txt
python Argv_Main_sgd.py . . . . . . . . . . T C . . . >> logOSg.txt
echo "Done!"

