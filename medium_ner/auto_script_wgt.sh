#!/bin/bash
python auto_transfer_main_noweighting_wgt.py 27 16 nw921 >> logTwgt.txt
python auto_transfer_main_noweighting_wgt.py 27 16 nw917 >> logTwgt.txt
python auto_transfer_main_noweighting_wgt.py 27 16 nw915 >> logTwgt.txt
python auto_transfer_main_noweighting_wgt.py 27 16 lastw910 >> logTwgt.txt
echo "Transfer notrainable done!"
