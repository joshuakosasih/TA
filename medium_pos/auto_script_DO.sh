#!/bin/bash
echo "Epoch"
python auto_main_DO.py 0 >> logDO.txt
python auto_main_DO.py 0.1 >> logDO.txt
python auto_main_DO.py 0.2 >> logDO.txt
python auto_main_DO.py 0.3 >> logDO.txt
python auto_main_DO.py 0.4 >> logDO.txt
python auto_main_DO.py 0.5 >> logDO.txt
python auto_main_DO.py 0.6 >> logDO.txt
python auto_main_DO.py 0.7 >> logDO.txt
python auto_main_DO.py 0.8 >> logDO.txt
python auto_main_DO.py 0.9 >> logDO.txt
echo "done!"
