#!/usr/bin/env bash

filename="download_list.csv"
save_dir="./"

while read line
do
    array=(${line//,/ })
    wget -P $save_dir -c --tries=75 ${array[0]}
done < $filename
