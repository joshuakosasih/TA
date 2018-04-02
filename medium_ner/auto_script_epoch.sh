#!/bin/bash
echo "epoch 16 - 22"
#python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 16 16 >> logTnW3.txt
#python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 17 16 >> logTnW3.txt
#python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 18 16 >> logTnW3.txt
#python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 19 16 >> logTnW3.txt
#python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 20 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 21 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 22 16 >> logTnW3.txt
echo "epoch 23 - 29"
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 23 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 24 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 25 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 26 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 27 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 28 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 29 16 >> logTnW3.txt
echo "done!"
