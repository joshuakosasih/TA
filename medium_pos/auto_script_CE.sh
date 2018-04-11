#!/bin/bash
echo "polyglot"
python auto_main_CE.py polyglot.vec polyglot-char.vec sum >> logCE.txt
python auto_main_CE.py polyglot.vec polyglot-char.vec mul >> logCE.txt
python auto_main_CE.py polyglot.vec polyglot-char.vec concat >> logCE.txt
python auto_main_CE.py polyglot.vec polyglot-char.vec ave >> logCE.txt
echo "done!"
