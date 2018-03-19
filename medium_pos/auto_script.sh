#!/bin/bash
echo "polyglot"
python auto_main_with_char.py polyglot.vec polyglot-char.vec 1 >> log3.txt
python auto_main_with_char.py polyglot.vec polyglot-char.vec 2 >> log3.txt
python auto_main_with_char.py polyglot.vec polyglot-char.vec 3 >> log3.txt
echo "WE_w"
python auto_main_with_char.py WE_w.vec WE_w-char.vec 1 >> log3.txt
python auto_main_with_char.py WE_w.vec WE_w-char.vec 2 >> log3.txt
python auto_main_with_char.py WE_w.vec WE_w-char.vec 3 >> log3.txt
echo "wiki.id.vec"
python auto_main_with_char.py wiki.id.vec wiki.id-char.vec 1 >> log3.txt
python auto_main_with_char.py wiki.id.vec wiki.id-char.vec 2 >> log3.txt
python auto_main_with_char.py wiki.id.vec wiki.id-char.vec 3 >> log3.txt
echo "done!"
