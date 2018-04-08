#!/bin/bash
echo "Transfer data 100% - 10%"
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 1 11 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 1 12 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.5 11 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.5 12 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.5 13 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.2 11 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.2 12 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.2 13 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.1 11 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.1 12 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.1 13 >> logCT_e5.txt
echo "Transfer data 5% - 1%"
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.05 11 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.05 12 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.05 13 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.05 14 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.02 11 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.02 12 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.02 13 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.02 14 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.01 11 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.01 12 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.01 13 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.01 14 >> logCT_e5.txt
echo "Transfer data 0.5%"
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.005 11 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.005 12 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.005 13 >> logCT_e5.txt
python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.005 14 >> logCT_e5.txt
echo "Transfer done!"
echo "No transfer data 100% - 10%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 1 11 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 1 12 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.5 11 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.5 12 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.5 13 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.2 11 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.2 12 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.2 13 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.1 11 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.1 12 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.1 13 >> logCnT_e5.txt
echo "No transfer data 5% - 1%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.05 11 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.05 12 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.05 13 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.05 14 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.02 11 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.02 12 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.02 13 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.02 14 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.01 11 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.01 12 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.01 13 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.01 14 >> logCnT_e5.txt
echo "No transfer data 0.5%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.005 11 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.005 12 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.005 13 >> logCnT_e5.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 28 16 0.005 14 >> logCnT_e5.txt
echo "No transfer done!"
