#! /bin/bash
python init_database.py
python skin_core_scanner.py --epsilon 0.7 --suppress_output --no_git
python skin_core_scanner.py --epsilon 0.5 --suppress_output --no_git
python skin_core_scanner.py --epsilon 0.1 --suppress_output --no_git
