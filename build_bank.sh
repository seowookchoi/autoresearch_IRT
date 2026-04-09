#!/bin/bash
cd /Users/awesomedasom/Desktop/autoresearch_irt
source venv/bin/activate
export $(cat .env | xargs)

for i in $(seq 1 15); do
  echo "=== Run $i/15 $(date) ===" >> bank_build.log
  python3 main.py --n 5 --easy-n 2 --quiet >> bank_build.log 2>&1
done

echo "=== DONE $(date) ===" >> bank_build.log
