#! /bin/bash
python init_database.py
python --epsilon 0.7 --suppress_output 
python --epsilon 0.5 --suppress_output
python --epsilon 0.1 --suppress_output
