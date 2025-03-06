# python -m wikiextractor.WikiExtractor frwiki-latest-pages-articles1.xml-p1p306134.bz2 --json

import wikiextractor
import os
import json
import re
import sys

file = 'frwiki-latest-pages-articles1.xml-p1p306134.bz2'

# Extract the content of the file using WikiExtractor
os.system(f'python3 -m wikiextractor.WikiExtractor {file} --json')