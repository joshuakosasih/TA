#!/bin/bash
echo "Rang fasttext 100"
python auto_main_WE.py rang_fasttext.vec rang_fasttext-char.vec sum >> logWE.txt
python auto_main_WE.py rang_fasttext.vec rang_fasttext-char.vec mul >> logWE.txt
python auto_main_WE.py rang_fasttext.vec rang_fasttext-char.vec concat >> logWE.txt
python auto_main_WE.py rang_fasttext.vec rang_fasttext-char.vec ave >> logWE.txt
echo "Rang word2vec 100"
python auto_main_WE.py rang_word2vec.vec rang_word2vec-char.vec sum >> logWE.txt
python auto_main_WE.py rang_word2vec.vec rang_word2vec-char.vec mul >> logWE.txt
python auto_main_WE.py rang_word2vec.vec rang_word2vec-char.vec concat >> logWE.txt
python auto_main_WE.py rang_word2vec.vec rang_word2vec-char.vec ave >> logWE.txt
echo "polyglot"
python auto_main_WE.py polyglot.vec polyglot-char.vec sum >> logWE.txt
python auto_main_WE.py polyglot.vec polyglot-char.vec mul >> logWE.txt
python auto_main_WE.py polyglot.vec polyglot-char.vec concat >> logWE.txt
python auto_main_WE.py polyglot.vec polyglot-char.vec ave >> logWE.txt
echo "WE_w"
python auto_main_WE.py WE_w.vec WE_w-char.vec sum >> logWE.txt
python auto_main_WE.py WE_w.vec WE_w-char.vec mul >> logWE.txt
python auto_main_WE.py WE_w.vec WE_w-char.vec concat >> logWE.txt
python auto_main_WE.py WE_w.vec WE_w-char.vec ave >> logWE.txt
echo "wiki.id.vec"
python auto_main_WE.py wiki.id.vec wiki.id-char.vec sum >> logWE.txt
python auto_main_WE.py wiki.id.vec wiki.id-char.vec mul >> logWE.txt
python auto_main_WE.py wiki.id.vec wiki.id-char.vec concat >> logWE.txt
python auto_main_WE.py wiki.id.vec wiki.id-char.vec ave >> logWE.txt
echo "done!"
