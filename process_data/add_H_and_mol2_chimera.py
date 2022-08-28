##This script should be run using chimera 1.16 from the terminal

import os
from chimera import runCommand as rc
from chimera import replyobj
 
# change to folder with data files
os.chdir("path/to/data")
 
path_names = []
file_names = []
 
# assemble all path names
for root, dirs, files in os.walk(".", topdown=False):
   for name in files:
  	if name.endswith("_pocket.pdb") and not os.path.exists("mol2files_nocharges/" + name[:-3] + "mol2"):
           path_names.append(os.path.join(root, name))
           file_names.append(name)
 
# loop through the files, opening, processing, saving, and closing each in turn
for i in range(0, len(path_names)):
     	rc("open " + path_names[i])
     	replyobj.status("Processing " + file_names[i])
     	rc("addh") #Add hydrogens
     	mol2_name = "mol2files_nocharges/" + file_names[i][:-3] + "mol2"
     	rc("write format mol2 0 " + mol2_name) #Convert to mol2
        rc("close all")
