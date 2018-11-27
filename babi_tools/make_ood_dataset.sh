#!/usr/bin/env bash

if [ $# -lt 2 ]; then
  echo "Usage: make_ood_dataset.sh <source folder> <result folder>"
  exit
fi
SOURCE_FOLDER=$1
RESULT_FOLDER=$2

mkdir -p $RESULT_FOLDER
cp $SOURCE_FOLDER/dialog-babi-task6* $RESULT_FOLDER/

# we have the only backoff response to all OOD
printf "1 Sorry I didn't catch that. Could you please repeat?" >> $RESULT_FOLDER/dialog-babi-task6-dstc2-candidates.txt
for dataset in trn dev tst; do
  src_filename="$SOURCE_FOLDER/dialog-babi-task6-dstc2-$dataset.txt"
  dst_filename="$RESULT_FOLDER/dialog-babi-task6-dstc2-$dataset.txt"
  echo "Processing $src_filename"
  python ood_augmentation.py $src_filename $dst_filename
done

rm -rf .tmp
