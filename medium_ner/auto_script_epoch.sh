#!/bin/bash
echo "seed 0"
python auto_main_noweighting_dev.py ner_3_train.ner 0 200 16 >> logEnTnW3.txt
echo "seed 1"
python auto_main_noweighting_dev.py ner_3_train.ner 1 200 16 >> logEnTnW3.txt
echo "seed 2"
python auto_main_noweighting_dev.py ner_3_train.ner 2 200 16 >> logEnTnW3.txt
echo "done!"
