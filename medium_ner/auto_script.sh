#!/bin/bash
echo "1  - 0.8"
python auto_main_with_char.py 1 >> log4.txt
python auto_main_with_char.py 0.9 >> log4.txt
python auto_main_with_char.py 0.8 >> log4.txt
echo "0.7 - 0.5"
python auto_main_with_char.py 0.7 >> log4.txt
python auto_main_with_char.py 0.6 >> log4.txt
python auto_main_with_char.py 0.5 >> log4.txt
echo "0.4 - 0.1"
python auto_main_with_char.py 0.4 >> log4.txt
python auto_main_with_char.py 0.3 >> log4.txt
python auto_main_with_char.py 0.2 >> log4.txt
python auto_main_with_char.py 0.1 >> log4.txt
echo "done!"
