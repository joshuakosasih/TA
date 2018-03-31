#!/bin/bash
echo "data 100% - 10%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 1 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 1 1 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.5 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.5 1 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.2 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.2 1 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.1 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.1 1 >> logTnW3.txt
echo "data 5% - 1%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 1 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.05 2 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 1 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.02 2 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 1 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.01 2 >> logTnW3.txt
echo "data 0.5% - 0.1%"
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 1 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.005 2 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.002 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.002 1 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.002 2 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.001 0 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.001 1 >> logTnW3.txt
python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 12 16 0.001 2 >> logTnW3.txt
echo "done!"
