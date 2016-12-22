# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:53:21 2016

@author: jensv

Create sql run database.
"""

import sqlite3
import os

assert os.path.exists('../output'), ("../output does not exist."
                                     "Please place repo next to"
                                     "'output' directory")
assert not os.path.exists('../output/output.db'), ("database "
                                                   "../output/output.db "
                                                   "already exists.")
connection = sqlite3.connect('../output/output.db')
cursor = connection.cursor()
cursor.execute("CREATE TABLE Runs(datetime TEXT, "
               "points_core INTEGER, "
               "points_transition INTEGER, "
               "points_skin INTEGER, "
               "core_radius REAL, " 
               "transition_width REAL, " 
               "skin_width REAL, k_bar_start REAL, "
               "k_bar_end REAL, k_bar_num INTEGER, "
               "lambda_bar_start REAL, "
               "lambda_bar_end REAL, "
               "lambda_bar_num INTEGER, epsilon REAL, "
               "git_commit TEXT, python_call TEXT)")
cursor.close()
connection.commit()
connection.close()
