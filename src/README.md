Within this folder is the python executable "bondorder-pl.py".
This executable calls the file "bopcalcpl.py", which contains
most of the code for the analysis.

This executable acts on the files "conf-nvt.gro", which is a
GROMACS coordinate file. It consists of water molecules confined
between two parallel graphene plates.

Running the executable using "python" or "ipython" prodcues
three output files called  "boop_pl_1layer.dat", "boop_pl_2layer.dat"
and "boop_pl_3layer.dat". These three files contain the bond
order parameter values and the error on the value for each time
step while treating the confined structure as a mono, bi or tri
layered structure.

Tests for the python analysis module "bopcalcpl.py" are contained
in the files "test_bopcalcpl.py". This can be tested by running 
the commane 'nosetests test_bopcalcpl.py'.
