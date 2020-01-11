#!/usr/bin/env bash
# conda list -e > requirements.txt
git add .
time_str=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "commit in ${time_str} by `whoami`"
git push origin master
git push github master
exit