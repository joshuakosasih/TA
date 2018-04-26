#!/bin/bash
echo "add"
python auto_main_BO.py polyglot.vec polyglot-char.vec sum sum 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum mul 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum concat 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum ave 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul sum 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul mul 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul concat 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul ave 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave sum 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave mul 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave concat 1 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave ave 1 >> logBOCW.txt
echo "subtract"
python auto_main_BO.py polyglot.vec polyglot-char.vec sum sum 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum mul 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum concat 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum ave 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul sum 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul mul 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul concat 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul ave 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave sum 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave mul 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave concat 2 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave ave 2 >> logBOCW.txt
echo "multiply"
python auto_main_BO.py polyglot.vec polyglot-char.vec sum sum 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum mul 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum concat 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum ave 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul sum 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul mul 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul concat 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul ave 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave sum 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave mul 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave concat 3 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave ave 3 >> logBOCW.txt
echo "average"
python auto_main_BO.py polyglot.vec polyglot-char.vec sum sum 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum mul 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum concat 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum ave 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul sum 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul mul 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul concat 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul ave 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave sum 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave mul 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave concat 4 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave ave 4 >> logBOCW.txt
echo "maximum"
python auto_main_BO.py polyglot.vec polyglot-char.vec sum sum 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum mul 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum concat 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec sum ave 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul sum 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul mul 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul concat 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec mul ave 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave sum 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave mul 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave concat 5 >> logBOCW.txt
python auto_main_BO.py polyglot.vec polyglot-char.vec ave ave 5 >> logBOCW.txt
echo "done!"
