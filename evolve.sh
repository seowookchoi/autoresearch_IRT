#!/bin/bash
cd /Users/awesomedasom/Desktop/autoresearch_irt
source venv/bin/activate
export $(cat .env | xargs)

# log() in autoevolve.py writes to evolve.log directly — no shell redirect needed
python3 autoevolve.py \
  --iterations 20 \
  --test-items 15 \
  --min-delta 0.05 \
  --n-generate 5
