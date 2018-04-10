#!/bin/bash
echo "Rang fasttext 100"
python auto_main_CE.py rang_fasttext.vec rang_fasttext-char.vec sum >> logCE.txt
python auto_main_CE.py rang_fasttext.vec rang_fasttext-char.vec mul >> logCE.txt
python auto_main_CE.py rang_fasttext.vec rang_fasttext-char.vec concat >> logCE.txt
python auto_main_CE.py rang_fasttext.vec rang_fasttext-char.vec ave >> logCE.txt
echo "Rang word2vec 100"
python auto_main_CE.py rang_word2vec.vec rang_word2vec-char.vec sum >> logCE.txt
python auto_main_CE.py rang_word2vec.vec rang_word2vec-char.vec mul >> logCE.txt
python auto_main_CE.py rang_word2vec.vec rang_word2vec-char.vec concat >> logCE.txt
python auto_main_CE.py rang_word2vec.vec rang_word2vec-char.vec ave >> logCE.txt
echo "polyglot"
python auto_main_CE.py polyglot.vec polyglot-char.vec sum >> logCE.txt
python auto_main_CE.py polyglot.vec polyglot-char.vec mul >> logCE.txt
python auto_main_CE.py polyglot.vec polyglot-char.vec concat >> logCE.txt
python auto_main_CE.py polyglot.vec polyglot-char.vec ave >> logCE.txt
echo "WE_w"
python auto_main_CE.py WE_w.vec WE_w-char.vec sum >> logCE.txt
python auto_main_CE.py WE_w.vec WE_w-char.vec mul >> logCE.txt
python auto_main_CE.py WE_w.vec WE_w-char.vec concat >> logCE.txt
python auto_main_CE.py WE_w.vec WE_w-char.vec ave >> logCE.txt
echo "wiki.id.vec"
python auto_main_CE.py wiki.id.vec wiki.id-char.vec sum >> logCE.txt
python auto_main_CE.py wiki.id.vec wiki.id-char.vec mul >> logCE.txt
python auto_main_CE.py wiki.id.vec wiki.id-char.vec concat >> logCE.txt
python auto_main_CE.py wiki.id.vec wiki.id-char.vec ave >> logCE.txt
echo "done!"
