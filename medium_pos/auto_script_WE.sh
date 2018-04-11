#!/bin/bash
echo "polyglot"
python auto_main_WE.py polyglot.vec polyglot-char.vec sum >> logWE.txt
python auto_main_WE.py polyglot.vec polyglot-char.vec mul >> logWE.txt
python auto_main_WE.py polyglot.vec polyglot-char.vec concat >> logWE.txt
python auto_main_WE.py polyglot.vec polyglot-char.vec ave >> logWE.txt
echo "done!"
