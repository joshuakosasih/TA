#!/bin/bash
echo "sgd"
python auto_main_with_char.py sgd >> log.txt
echo "rmsprop"
python auto_main_with_char.py rmsprop >> log.txt
echo "adagrad"
python auto_main_with_char.py adagrad >> log.txt
echo "adadelta"
python auto_main_with_char.py adadelta >> log.txt
echo "adam"
python auto_main_with_char.py adam >> log.txt
echo "adamax"
python auto_main_with_char.py adamax >> log.txt
echo "nadam"
python auto_main_with_char.py nadam >> log.txt
echo "done!"
