###############################################
## Project: Endogenous Amenities and Location Sorting
## Author: Sriram Tolety
###############################################

# ##################################################################################################################################
# This file executes both welfare calculations.
##################################################################################################################################

# Required library imports
import os
import time
import sys
start = time.time()

# Changing working directory to location of current file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

gamma_val = sys.argv[1]
value = str(gamma_val).zfill(3)

use_B_input = str(sys.argv[2])

os.system("python3 welfare.py config.json " + value + " " + use_B_input)  
os.system("python3 welfare_homogeneous.py config.json " + value + " " + use_B_input)  

print('Full Run Took Approximately ', time.time()-start, ' Seconds!')