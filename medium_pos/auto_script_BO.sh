#!/bin/bash
echo "polyglot"
python auto_main_BO.py polyglot.vec polyglot-char.vec 1 >> logBO.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec 2 >> logBO.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec 3 >> logBO.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec 4 >> logBO.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec 5 >> logBO.txt
echo "done!"
