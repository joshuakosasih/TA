#!/bin/bash
echo "data 100% - 10%"
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 1 0 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 1 1 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.5 0 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.5 1 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.5 2 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.2 0 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.2 1 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.2 2 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.1 0 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.1 1 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.1 2 >> logCT.txt
echo "data 5% - 1%"
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 0 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 1 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 2 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 3 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 0 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 1 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 2 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 3 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 0 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 1 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 2 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 3 >> logCT.txt
echo "data 0.5%"
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 0 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 1 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 2 >> logCT.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 3 >> logCT.txt
echo "done!"
