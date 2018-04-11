#!/bin/bash
echo "Epoch"
python auto_main_epoch.py 10 8 >> logEpoch.txt
python auto_main_epoch.py 10 8 >> logEpoch.txt
python auto_main_epoch.py 20 16 >> logEpoch.txt
python auto_main_epoch.py 20 16 >> logEpoch.txt
echo "done!"
