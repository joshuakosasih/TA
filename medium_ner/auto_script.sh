#!/bin/bash
echo "epoch 2 - 8"
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 2 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 3 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 4 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 5 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 6 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 7 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 8 16 >> logTnW3.txt
echo "epoch 9 - 15"
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 9 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 10 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 11 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 12 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 13 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 14 16 >> logTnW3.txt
python auto_transfer_main_noweighting.py ner_3_train.ner ner_3_test.ner 15 16 >> logTnW3.txt
echo "done!"
