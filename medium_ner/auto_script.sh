#!/bin/bash
echo "Transfer data 100% - 10%"
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 1 0 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 1 4 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.5 0 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.5 4 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.5 2 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.2 0 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.2 4 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.2 2 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.1 0 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.1 4 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.1 2 >> logCT_e2.txt
echo "Transfer data 5% - 1%"
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.05 0 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.05 4 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.05 2 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.05 3 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.02 0 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.02 4 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.02 2 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.02 3 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.01 0 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.01 4 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.01 2 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.01 3 >> logCT_e2.txt
echo "Transfer data 0.5%"
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.005 0 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.005 4 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.005 2 >> logCT_e2.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.005 3 >> logCT_e2.txt
echo "Transfer done!"
echo "No transfer data 100% - 10%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 1 0 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 1 4 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.5 0 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.5 4 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.5 2 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.2 0 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.2 4 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.2 2 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.1 0 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.1 4 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.1 2 >> logCnT_e2.txt
echo "No transfer data 5% - 1%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.05 0 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.05 4 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.05 2 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.05 3 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.02 0 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.02 4 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.02 2 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.02 3 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.01 0 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.01 4 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.01 2 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.01 3 >> logCnT_e2.txt
echo "No transfer data 0.5%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.005 0 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.005 4 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.005 2 >> logCnT_e2.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 30 16 0.005 3 >> logCnT_e2.txt
echo "No transfer done!"
