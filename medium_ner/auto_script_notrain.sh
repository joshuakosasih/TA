#!/bin/bash
python auto_transfer_main_noweighting_notrainable.py 27 16 True True nw921 >> logTnotrain.txt
python auto_transfer_main_noweighting_notrainable.py 27 16 True False nw921 >> logTnotrain.txt
python auto_transfer_main_noweighting_notrainable.py 27 16 False True nw921 >> logTnotrain.txt
python auto_transfer_main_noweighting_notrainable.py 27 16 False False nw921 >> logTnotrain.txt
echo "Transfer notrainable done!"
