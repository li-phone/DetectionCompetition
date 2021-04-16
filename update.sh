#!/usr/bin/env bash
#pip freeze > requirements.txt
git add .
time_str=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "${1} commit in ${time_str} by `whoami`"
# git push -f gitee master
# git push -f github master
git push --set-upstream github li-phone -f
git push --set-upstream gitee li-phone -f
exit
