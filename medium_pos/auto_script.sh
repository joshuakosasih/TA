#!/bin/bash
echo "Rang fasttext 100"
python auto_main_with_char.py rang_fasttext.vec rang_fasttext-char.vec 1 >> logExM.txt
python auto_main_with_char.py rang_fasttext.vec rang_fasttext-char.vec 2 >> logExM.txt
python auto_main_with_char.py rang_fasttext.vec rang_fasttext-char.vec 3 >> logExM.txt
echo "Rang word2vec 100"
python auto_main_with_char.py rang_word2vec.vec rang_word2vec-char.vec 1 >> logExM.txt
python auto_main_with_char.py rang_word2vec.vec rang_word2vec-char.vec 2 >> logExM.txt
python auto_main_with_char.py rang_word2vec.vec rang_word2vec-char.vec 3 >> logExM.txt
echo "polyglot"
python auto_main_with_char.py polyglot.vec polyglot-char.vec 1 >> logExM.txt
python auto_main_with_char.py polyglot.vec polyglot-char.vec 2 >> logExM.txt
python auto_main_with_char.py polyglot.vec polyglot-char.vec 3 >> logExM.txt
echo "WE_w"
python auto_main_with_char.py WE_w.vec WE_w-char.vec 1 >> logExM.txt
python auto_main_with_char.py WE_w.vec WE_w-char.vec 2 >> logExM.txt
python auto_main_with_char.py WE_w.vec WE_w-char.vec 3 >> logExM.txt
echo "wiki.id.vec"
python auto_main_with_char.py wiki.id.vec wiki.id-char.vec 1 >> logExM.txt
python auto_main_with_char.py wiki.id.vec wiki.id-char.vec 2 >> logExM.txt
python auto_main_with_char.py wiki.id.vec wiki.id-char.vec 3 >> logExM.txt
echo "done!"
