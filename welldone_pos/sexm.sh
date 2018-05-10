#!/bin/bash
echo "Running default"
for i in $(seq 1 3)
do
  for j in $(seq 0 1)
  do
    python Argv_Main.py . polyglot.vec polyglot-char.vec . . . . $i . . . . . . . >> logExM.txt
  done
  for j in $(seq 0 1)
  do
    python Argv_Main.py . rang_fasttext.vec rang_fasttext-char.vec . . . . $i . . . . . . . >> logExM.txt
  done
  for j in $(seq 0 1)
  do
    python Argv_Main.py . rang_word2vec.vec rang_word2vec-char.vec . . . . $i . . . . . . . >> logExM.txt
  done
  for j in $(seq 0 1)
  do
    python Argv_Main.py . wiki.id.vec wiki.id-char.vec . . . . $i . . . . . . . >> logExM.txt
  done
  for j in $(seq 0 1)
  do
    python Argv_Main.py . WE_w.vec WE_w-char.vec . . . . $i . . . . . . . >> logExM.txt
  done
done
echo "Done!"

