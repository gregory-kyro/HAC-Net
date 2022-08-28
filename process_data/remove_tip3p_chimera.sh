"""
Script to run a python script in chimera 1.16 from the terminal.
"""

/Applications/Chimera.app/Contents/MacOS/chimera --nogui path/to/process_chimera.py

# NOTE: In rare cases (roughly 1 in 5,000) chimera may fail to add hydrogens to a pdb file. Create a new dataset excluding these pdb files and
# rerun the script. In our code, this updated csv file is called: 'PDBbind_2020_data_cleaned_final.csv'.

for f in *
do
  echo $f
  sed -i '' 's/H\.t3p/H	/' $f
  sed -i '' 's/O\.t3p/O\.3  /' $f
  done
