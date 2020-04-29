
#!/usr/bin/env bash

filename=$1
save_dir=$2

while read line
do
    array=(${line//,/ })
    wget -P $save_dir -c --tries=75 ${array[0]}
done < $filename
