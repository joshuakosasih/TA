#!/bin/bash
echo "Epoch"
python auto_main_DO.py 0 >> logEpoch.txt
python auto_main_DO.py 0.1 >> logEpoch.txt
python auto_main_DO.py 0.2 >> logEpoch.txt
python auto_main_DO.py 0.3 >> logEpoch.txt
python auto_main_DO.py 0.4 >> logEpoch.txt
python auto_main_DO.py 0.5 >> logEpoch.txt
python auto_main_DO.py 0.6 >> logEpoch.txt
python auto_main_DO.py 0.7 >> logEpoch.txt
python auto_main_DO.py 0.8 >> logEpoch.txt
python auto_main_DO.py 0.9 >> logEpoch.txt
echo "done!"
