#!/bin/bash
cd /Users/awesomedasom/Desktop/autoresearch_irt
source venv/bin/activate
export $(cat .env | xargs)

claude --dangerously-skip-permissions --max-turns 80 -p "
Working directory: /Users/awesomedasom/Desktop/autoresearch_irt
The venv is already activated and GROQ_API_KEY is set.

Your job is to build up the real FDA warning letter item bank. Do all steps below.

STEP 1 — Import these URLs we found but have not imported yet (run each):
  python3 fda_importer.py --url <url> --max-items 4 --quiet

URLs:
- https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/episciences-inc-631902-08252022
- https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/adamson-analytical-laboratories-inc-614644-08172021
- https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/americo-f-padilla-md-700447-03252025
- https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/richard-w-foltin-phd-690189-11062024

STEP 2 — Use WebSearch to find 10 more FDA warning letter URLs not already imported.
Priority domains (in order):
  1. promo_review: 21 CFR 202.1 fair balance, off-label, brief summary — HIGHEST PRIORITY, 0 retained real items so far
  2. informed_consent: 21 CFR 50.25, 45 CFR 46 re-consent, missing elements
  3. 21cfr11: 21 CFR 11.10, 211.68 audit trails, electronic records

Only use URLs from: fda.gov/inspections-compliance-enforcement-and-criminal-investigations/warning-letters/

STEP 3 — Import each URL found in Step 2:
  python3 fda_importer.py --url <url> --max-items 4 --quiet

STEP 4 — Print final comparison and append to log:
  python3 -c \"from database import Database; from fda_importer import print_comparison; db=Database('compliance_bank.db'); print_comparison(db)\"

Append all output to bank_build_real.log
" >> bank_build_real.log 2>&1
