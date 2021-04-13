#!/usr/bin/env bash
# pip freeze > requirements.txt

# checkout to gitee li-phone branch
git checkout li-phone
git add .
time_str=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "${1} commit in ${time_str} by `whoami`"
git push origin master

# checkout to github li-phone branch
git checkout li-phone
git add .
git commit -m "${1} commit in ${time_str} by `whoami`"
git push github li-phone
exit
