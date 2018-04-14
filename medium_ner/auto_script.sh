#!/bin/bash
echo "Transfer data 100%"
for i in $(seq 0 12)
do
  python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 1 $i nw921 >> logCTfull.txt
done
echo "Transfer data 50% - 10%"
for i in $(seq 0 12)
do
  python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.5 $i nw921 >> logCTfull.txt
done
for i in $(seq 0 12)
do
  python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.2 $i nw921 >> logCTfull.txt
done
for i in $(seq 0 12)
do
  python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.1 $i nw921 >> logCTfull.txt
done
echo "Transfer data 5% - 1%"
for i in $(seq 0 12)
do
  python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.05 $i nw921 >> logCTfull.txt
done
for i in $(seq 0 12)
do
  python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.02 $i nw921 >> logCTfull.txt
done
for i in $(seq 0 12)
do
  python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.01 $i nw921 >> logCTfull.txt
done
echo "Transfer data 0.5%"
for i in $(seq 0 12)
do
  python auto_transfer_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.005 $i nw921 >> logCTfull.txt
done
echo "Transfer done!"
echo "Non-Transfer data 100%"
for i in $(seq 0 12)
do
  python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 1 $i nw921 >> logCnTfull.txt
done
echo "Non-Transfer data 50% - 10%"
for i in $(seq 0 12)
do
  python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.5 $i nw921 >> logCnTfull.txt
done
for i in $(seq 0 12)
do
  python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.2 $i nw921 >> logCnTfull.txt
done
for i in $(seq 0 12)
do
  python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.1 $i nw921 >> logCnTfull.txt
done
echo "Non-Transfer data 5% - 1%"
for i in $(seq 0 12)
do
  python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.05 $i nw921 >> logCnTfull.txt
done
for i in $(seq 0 12)
do
  python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.02 $i nw921 >> logCnTfull.txt
done
for i in $(seq 0 12)
do
  python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.01 $i nw921 >> logCnTfull.txt
done
echo "Non-Transfer data 0.5%"
for i in $(seq 0 12)
do
  python auto_main_noweighting_cut.py ner_3_train.ner ner_3_test.ner 27 16 0.005 $i nw921 >> logCnTfull.txt
done
echo "Non-Transfer done!"
