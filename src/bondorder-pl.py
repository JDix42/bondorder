import numpy as np
import math
import MDAnalysis as MDA
import matplotlib.pyplot as plt
import bopcalcpl as boppl
from MDAnalysis.core.distances import distance_array, calc_angles
from itertools import combinations

layers = [2]

filenamegro = str("conf-nvt")
filenametrr = str("conf-nvt")

u = MDA.Universe(filenamegro + ".gro", filenametrr + ".gro")

GRA = u.selectAtoms("resname GRA")
SOL = u.selectAtoms("resname SOL")
OW = u.selectAtoms("name OW")
HW1 = u.selectAtoms('name HW1')
HW2 = u.selectAtoms('name HW2')

final = len(u.trajectory)

# Value of distance that molecules are ignored within the channel - in A
channel_offset = 10

symm_values = [4, 6]

one_layer = open("boop_pl_1layer.xvg", "w")
two_layer = open("boop_pl_2layer.xvg", "w")
three_layer = open("boop_pl_3layer.xvg", "w")

# Determines the positions and atom indicies of the maximum and minimum
# ends of the graphene channel. This is used as a reference for the rest
# of the simulation.
ts = 1  # Sets the timestep to the first timestep of the simulation

for ts in u.trajectory:  # Loops over all of the timesteps in the .trr file
    OW = u.selectAtoms("name OW")
    rGRA = GRA.positions  # Grabs positions of GRA atoms
    rSOL = SOL.positions  # Grabs posiitons of SOL atoms
    rOW = OW.positions
    rHW1 = HW1.positions
    rHW2 = HW2.positions
    bop1, bop21, bop22, bop31, bop32, bop33 = boppl.bopreader(rGRA, rSOL, rOW, rHW1, rHW2, layers, symm_values,  u.dimensions)
    for l in range(len(layers)):
        if layers[l] == 1:
            one_layer.write(str(0) +
            "  " +
            str(bop1[0, 0]) + "  " + str(bop1[0, 1]) +
            "  " +
            str(bop1[1, 0]) + "  " + str(bop1[1, 1]) +
             "\n")
        if layers[l] == 2:
            print bop21
            print bop22
            mean_error0 = math.sqrt(math.pow(bop21[0, 1],2) +
                                    math.pow(bop22[0, 1],2)) / 2
            mean_error1 = math.sqrt(math.pow(bop21[1, 1],2) +
                                    math.pow(bop22[1, 1],2)) / 2
                                   
            two_layer.write(str(0) +
            "  " +
            str(bop21[0, 0]) + "  " + str(bop21[0, 1]) +
            "  " +
            str(bop21[1, 0]) + "  " + str(bop21[1, 1]) +
            "  " +
            str(bop22[0, 0]) + "  " + str(bop22[0, 1]) +
            "  " +
            str(bop22[1, 0]) + "  " + str(bop22[1, 1]) +
            "  " +
            str(np.mean([bop21[0, 0], bop22[0, 0]])) + "  " + str(mean_error0) +
            "  " +
            str(np.mean([bop21[1, 0], bop22[1, 0]])) + "  " + str(mean_error1) +
            "\n")
        if layers[l] == 3:
            mean_error0 = math.sqrt(math.pow(bop31[0, 1],2) +
                                    math.pow(bop32[0, 1],2) +
                                    math.pow(bop33[0, 1],2)) / 3
            mean_error1 = math.sqrt(math.pow(bop31[1, 1],2) +
                                    math.pow(bop32[1, 1],2) +
                                    math.pow(bop33[1, 1],2)) / 3

            three_layer.write(str(0) +
            "  " +
            str(bop31[0, 0]) + "  " + str(bop31[0, 1]) +
            "  " +
            str(bop31[1, 0]) + "  " + str(bop31[1, 1]) +
            "  " +
            str(bop32[0, 0]) + "  " + str(bop32[0, 1]) +
             "  " +
            str(bop32[1, 0]) + "  " + str(bop32[1, 1]) +
            "  " +
            str(bop33[0, 0]) + "  " + str(bop33[0, 1]) + 
            "  " +
            str(bop33[1, 0]) + "  " + str(bop33[1, 1]) +
            " " +
            str(np.mean([bop31[0, 0], bop32[0, 0], bop33[0, 0]])) + "  " + str(mean_error0) +
            "  " +
            str(np.mean([bop31[1, 0], bop32[1, 0], bop33[1, 0]])) + "  " + str(mean_error1) +
            "\n")

one_layer.close()
two_layer.close()
three_layer.close()
