#!/usr/bin/env bash

filename="download_list.txt"
save_dir="./"
if ${1}!="":
  save_dir=${1}
echo "save_dir:${save_dir}"
while read line
do
    array=(${line//,/ })
    wget -P $save_dir -c --tries=75 ${array[0]}
done < $filename
