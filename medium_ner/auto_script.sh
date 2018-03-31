#!/bin/bash
echo "data 100% - 10%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 1 0 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 1 1 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.5 0 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.5 1 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.5 2 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.2 0 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.2 1 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.2 2 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.1 0 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.1 1 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.1 2 >> logCnT.txt
echo "data 5% - 1%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 0 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 1 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 2 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 3 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 0 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 1 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 2 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 3 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 0 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 1 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 2 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 3 >> logCnT.txt
echo "data 0.5%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 0 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 1 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 2 >> logCnT.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 3 >> logCnT.txt
echo "done!"
