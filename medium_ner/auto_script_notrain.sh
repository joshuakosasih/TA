#!/bin/bash
python auto_transfer_main_noweighting_notrainable.py 27 16 True True lastw918 >> logTnotrain.txt
python auto_transfer_main_noweighting_notrainable.py 27 16 True False lastw918 >> logTnotrain.txt
python auto_transfer_main_noweighting_notrainable.py 27 16 False True lastw918 >> logTnotrain.txt
python auto_transfer_main_noweighting_notrainable.py 27 16 False False lastw918 >> logTnotrain.txt
echo "Transfer notrainable done!"
