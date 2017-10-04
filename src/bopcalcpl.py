import numpy as np
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

"""
This files calculates the 4, 5 and 6 order bond orientational parameter
for pairs of atoms calculated using the Voronoi tesellation function in
python
"""
def bopreader(rGRA, rSOL, rOW, rHW1, rHW2, layers, symm_values, dimensions):
    """
    Notes:
    This contains all the useful stuff
    """
 
    # print(ts.time)

    # Calculates the positions of the maximum C atom for each timestep
    maxvl = np.max(rGRA[:, 2])
    # Calculates the position of the minimum C atom for each timestep
    minvl = np.min(rGRA[:, 2])

    if maxvl < minvl:
        raise ValueError("Max value (maxvl) is less than min value (minvl) for GRA sheets")

    # print(maxvalue,minvalue)

    bop1, bop21, bop22, bop31, bop32, bop33 = bopcalculator(rOW * 0.1, maxvl * 0.1, minvl * 0.1, symm_values, dimensions, layers)
    return bop1, bop21, bop22, bop31, bop32, bop33

def twolayer_slicing(pos,dim):
    """
    Separates the atoms in atom group into two layers based on their centre of mass
    in the y dimension
    """
    # This function assumes that all of the atoms have the same mass

    center = np.sum(pos[:,dim]) / np.shape(pos[:,dim])[0]

    above_plane = np.where((pos[:,dim] >= center))
    below_plane = np.where((pos[:,dim] < center))

    return pos[above_plane[0], :], pos[below_plane[0], :]


def threelayer_slicing(pos, dim):
    """
    Separates the atoms in the atom group into three layers based on their maximum
    and minimum y positions
    """
    # this function assumes that all of the atoms have the same mass

    max_val = np.max(pos[:,dim])
    min_val = np.min(y_pos[:,dim])

    range_val = max_val - min_val

    layer_thick = range_val / 3

    first_layer = np.where((pos[:,dim] >= min_val) &
                           (pos[:,dim] < layer_thick + min_val))
    second_layer = np.where((pos[:,dim] >= layer_thick + min_val) &
                            (pos[:,dim] < 2 * layer_thick + min_val))
    third_layer = np.where((pos[:,dim] >= 2 * layer_thick + min_val) &
                           (pos[:,dim] <= max_val))

    return pos[first_layer[0], :], pos[second_layer[0], :], pos[third_layer[0], :]


def atom_shift(zpos, max_value, zbox):
    """
    This function shifts all the atoms by +lz if they have z pos less than max_value
    """
    lower = np.where((zpos[:] < max_value))

    zpos[lower] += zbox

    return zpos

def bopcalculator(con_pos, maxvl, minvl, sym_values, box, layers):
    """
    This function calculates the bond orienentational order parameters
    for the symmetry values in sym_values on the oxygens of the confined water molecules
    """
    onelayer_bop = np.zeros((len(sym_values),2))
    twolayer_bop_1 = np.zeros((len(sym_values),2))
    twolayer_bop_2 = np.zeros((len(sym_values),2))
    threelayer_bop_1 = np.zeros((len(sym_values),2))
    threelayer_bop_2 = np.zeros((len(sym_values),2))
    threelayer_bop_3 = np.zeros((len(sym_values),2))

    max_dist = 0.4

    for l in range(len(layers)):
        if layers[l] == 1:
            onel_edges = [0 , box[0] * 0.1,
                         0, box[1] * 0.1,
                         0, box[2] * 0.1]
            bopcalc(con_pos,
                         sym_values, onelayer_bop, onel_edges, max_dist)

        if layers[l] == 2:
            twl_con_pos = twolayer_slicing(con_pos, 2)

            twol_edges_1 = [0, box[0] * 0.1,
                           0, box[1] * 0.1,
                           0, box[2] * 0.1]
            twol_edges_2 = [0, box[0] * 0.1,
                           0, box[1] * 0.1,
                           0, box[2] * 0.1]

            bopcalc(twl_con_pos[0],
                         sym_values, twolayer_bop_1, twol_edges_1, max_dist)
            bopcalc(twl_con_pos[1],
                         sym_values, twolayer_bop_2, twol_edges_2, max_dist)

        if layers[l] == 3:
            thl_con_pos = threelayer_slicing(con_pos, 2)

            threel_edges_1 = [0, box[0] * 0.1,
                             0, box[1] * 0.1,
                             0, box[2] * 0.1]
            threel_edges_2 = [0, box[0] * 0.1,
                             0, box[1] * 0.1,
                             0, box[2] * 0.1]
            threel_edges_3 = [0, box[0] * 0.1,
                             0, box[1] * 0.1,
                             0, box[2] * 0.1]

            bopcalc(thl_con_pos[0],
                         sym_values, threelayer_bop_1, threel_edges_1, max_dist)
            bopcalc(thl_con_pos[1],
                         sym_values, threelayer_bop_2, threel_edges_2, max_dist)
            bopcalc(thl_con_pos[2],
                         sym_values, threelayer_bop_3, threel_edges_3, max_dist)

    return onelayer_bop, twolayer_bop_1, twolayer_bop_2, threelayer_bop_1, threelayer_bop_2, threelayer_bop_3

def bopcalc(positions, sym_values, bo_val, edges, max_dist):
    """
    Notes: 
    positions contain all of the coordinates fo the water oxygen atoms that
    are at least 1nm away from the enterance of the channel.

    buffpositions are the coordinates respectively for the water oxygen atoms in the
    channel between the edge of the channel and 1nm within the channel.
    These atoms are used to calculate the bond orientational order parameter for the
    atoms in the x, y and z positions but do not contribute themselves to the value.

    sym_values is a list that contains all of the symmetries that need to be
    investigate, these are written as an integer.

    bondorder_values is a list that is written out containing the average bondorder
    values from this time step.

    max_values is a len=5 list containing:
    [0] - maxvalue edge of the channel,
    [1] - minvalue edge of the channel,
    [2] - simulation cell width in x direction,
    [3] - the x position of the atom with the maximum x distance in the simulation,
    [4] - the x poxisiton of the atom with the minimum x distance in the simulation.
    """
    real_values = []
    complex_values = []
    #bondorder_values=[]

    # Determine the maximum value of an index for the ID of a specific atom.
    desired_confined_atoms = len(positions)

    # Add atoms from 1nm within the boundary of the channel to the opposite
    # channel edge. This is to essentially look at the periodic images of the
    # water atoms and making sure that the Voronoi tessellation doesn't have
    # any problems across the bondary.    

    point_rep_x = replicate_1D(positions, edges[1], edges[0], 1.0, abs(edges[1]-edges[0]), 0)
    point_rep_xy = replicate_1D(point_rep_x, edges[3], edges[2], 1.0, abs(edges[3]-edges[2]), 1)

    #print edges[:]

    #plt.scatter(point_rep_xy[:,0], point_rep_xy[:,1], c="blue")
    #plt.scatter(point_rep_x[:,0], point_rep_x[:,1], c="green")
    #plt.scatter(positions[:,0], positions[:,1], c="red")
    #plt.show()

    atomset1, atomset2, radangle_pair, bo_val, ridge_points, distance= preprocess(
        point_rep_xy, desired_confined_atoms, sym_values, bo_val,[0,1], max_dist)
   
    # add distances and indicies to the bondorder matrix
    # To calculate the bondorder paramter the equation
    # psi-n=1/N*mod(sigma(j=1->N)*1/n_j sigma(k=1->n_j) exp(in*theata_jk)).
    # e^in*theata is split in to a sum of a real complex sine and cosine
    # functions using the identity e^ix=cos(x)+isin(x). There
    # mod(e^ix)=sqrt(cos(x)^2+sin(x)^2). This would be 1 but when the terms
    # are summed up that is not the case. N is the total number of molecules,
    # n_j is the number of nearest neighbours and theata_jk is the angle
    # between atom j and k.

    # This loops over all of the different values for the symmetry, n, and
    # calculates the real and imaginary components for each atom pair.
    # Go through all the different symmetries that are of interest
    for symm_sets in range(np.shape(sym_values)[0]):
        # Keeps a track of how many of the neighbours have been added to the
        # bondorder matrix for that symmetry value

        n = sym_values[symm_sets]

        # Add the bond order value to a list of all of the bond orders
        bo, r, c, err_bo, err=psin_calc(n, radangle_pair, atomset1, atomset2, desired_confined_atoms, ridge_points, distance, positions)
        real_values.append([r, err[0]])
        complex_values.append([c, err[1]])
        bo_val[symm_sets, :] = bo, err_bo

    return bo_val

def replicate_1D(points, max_edge, min_edge, edge_region, box_dim, dim):
    """
    This function works for a 1D perioidc system and replicates
    the coordinates in points within "edge_region" from min and
    max edges on the other edge to reproduce the effect of PBC
    """
    if edge_region == 0:
        edge_region = box_dim / 2
    
    maxval_index = np.where(points[:,dim] > max_edge - edge_region)
    minval_index = np.where(points[:,dim] < min_edge + edge_region)
    
    buf1 = points[maxval_index[0][:]]
    buf1[:,dim] -= box_dim
    buf2 = points[minval_index[0][:]]
    buf2[:,dim] += box_dim
    
    points_rep = np.concatenate([points, buf1, buf2])
    
    return points_rep

def select_atomsinrange(points, max_point):
    """
    This function selection all of the atom indicies where the
    value of points is less than the value given in max_point.
    
    This can be an index or a distance. The trick is to be creative
    with the imput points.
    """
    list = np.where((points[:] < max_point))

    return list

def atom_count(pairs_array, total_atoms):
    """
    This function calculates the number of instances that different atoms
    occur in a pairs array and returns a numpy array with these numbers
    """
    counts = np.zeros((total_atoms))
    counts = np.bincount(pairs_array[0:])

    return counts

def component_norm(comp_array, norm_array):
    """
    This function divides the array comp_array by the array norm_array
    """
    comp_norm = np.divide(comp_array[:], norm_array[:])

    return comp_norm

def psi4_calc(desired_confined_atoms, ridge_points, atomset1, realcomp_array, complexcomp_array, distance):
    """
    This function calculates the value of psi4 for the overall boop
    value
    """
    realcomp_4_array = np.zeros((desired_confined_atoms, 4))
    complexcomp_4_array = np.zeros((desired_confined_atoms, 4))
    for atom in range(desired_confined_atoms):
        dist_list = np.where((ridge_points[atomset1[:], 0] == atom))

        if np.shape(dist_list)[1] != 0:
            order_dist_list = np.argsort(distance[dist_list[:]])

            if np.shape(order_dist_list)[0] >= 4:              
                array_list =  dist_list[0][order_dist_list[:4]]
                
                realcomp_4_array[atom, 0:4] = realcomp_array[array_list[:]]
                complexcomp_4_array[atom, 0:4] = complexcomp_array[array_list[:]]

            # Normalise for the number of pairs
                realcomp_4_array[atom, :] /= 4
                complexcomp_4_array[atom, :] /= 4
            else:
                array_list = dist_list[0][order_dist_list[:]]
                
                realcomp_4_array[atom, 0:np.shape(order_dist_list)[0]] = realcomp_array[array_list]
                complexcomp_4_array[atom, 0:np.shape(order_dist_list)[0]] = complexcomp_array[array_list]

                # Normalise for the number of pairs
                realcomp_4_array[atom, :] /= np.shape(order_dist_list)[0]
                complexcomp_4_array[atom, :] /= np.shape(order_dist_list)[0]

            atom_sum_real = np.sum(realcomp_4_array, axis=1)
            atom_sum_complex = np.sum(complexcomp_4_array, axis=1)

    real_part = np.sum(atom_sum_real[:], axis=0)
    complex_part = np.sum(atom_sum_complex[:], axis=0)

    error_sum = np.zeros(2)
    error_sum[0] = array_error(atom_sum_real, real_part / desired_confined_atoms)
    error_sum[1] = array_error(atom_sum_complex, complex_part / desired_confined_atoms)

    bo_err = bo_error( real_part, error_sum[0], complex_part, error_sum[1]) / desired_confined_atoms

    bondordercomp = np.sqrt(real_part ** 2 + complex_part ** 2) / desired_confined_atoms

    return bondordercomp, real_part, complex_part, bo_err, error_sum

def preprocess(points_positions, desired_confined_atoms, sym_values, bondorder_values, axes, cutoff):
    """
    This function contains all of the preprocessing steps for calculating the boop values.
    It takes the coordinates and returns all the atom sets etc. that are need for the sub-
    sequent calculations
    """
    try:
        if np.shape(points_positions)[0] >= np.shape(points_positions)[1]:
            total_sites = np.shape(points_positions)[0]
        else:
            total_sites = np.shape(points_positions)[1]
    except IndexError:
        total_sites = len(points_positions)

    points = np.zeros((total_sites, 2))
    points[:, 0] = points_positions[:, axes[0]]
    points[:, 1] = points_positions[:, axes[1]]

    # Calculates the voronoi patterm
    vor = Voronoi(points)
 
    # calculates a list of all of the nearest neighbours.
    # Each pair is only written once.
    ridge_points = vor.ridge_points

    # Recalculates the pairs but having flipped the list so that
    # each pair is written twice but in the opposite direction
    # the second time.
    ridge_points = dupl_flip(ridge_points)

    # Calculates the distance vectors between the atoms in each pair.
    dist = points_positions[ridge_points[:, 0]] - points_positions[ridge_points[:, 1]]

    # Need xdist and zdist for arctan command later
    # Note: ydist not needed.
    xdist = dist[:,0]
    ydist = dist[:,1]

    # Calculates the actual distance (not the vector) for each atom.
    distance = np.sqrt((dist * dist).sum(axis=-1))

    # Selects all atom pairs where first atom was in orig. range
    # Note: atoms were duplicated to calculate pairs over PBCs.
    wai = select_atomsinrange(ridge_points[:,0], desired_confined_atoms)

    # Makes sure pairs only within a reasonable distance are taken
    # Prevents pairs that are over the cutoff being included.
    pairs_sep = select_atomsinrange(distance[wai[:]], cutoff)

    # Picks first atom in pairs that complies with the above crit.
    # This is used to count the number of neigh. for each atom.
    pairs_element_array = (ridge_points[wai[0][pairs_sep[:]], 0])

    try:
        counts = atom_count(pairs_element_array, desired_confined_atoms)
    # Needed if there are no atoms in confined space.
    except ValueError:
        for n in range(len(sym_values)):
            bondorder_values.append(0)

    # Extra redundency to ensure that no atoms are missing in case a false
    # Value Error is raised when atoms are present.
    try:
        for a in range(np.shape(counts)[0]):
            if counts[a]!=0:
                zero_ndx = a
                raise ValueError
    except ValueError:
        counts_temp = counts
        counts_add = np.zeros((zero_ndx), dtype=np.int64)
        counts = np.concatenate((counts_add, counts_temp))

    # Creates index of pairs where they are within the cut off and
    # don't have a zero neighbour count.
    
    pairs_index = np.where((distance[wai[:]] < cutoff) &
                                 (counts[ridge_points[wai[:], 0]] != 0))


    # Calulates angle between atoms in each pair. These pairs include
    # both [0] -> [1] and [1] -> [0] directions.
    radangle_pair = np.arctan2(ydist, xdist)

    # print(radangle_pair)

    # List of atoms that aren't from PBC replicate or buffer zone
    # have more than 0 neigh. and are within the distance cutoff
    # of each other.
    atomset1 = wai[0][pairs_index[1][:]]

    # Number of neighbours for atoms in the previous regime
    atomset2 = counts[ridge_points[wai[0][pairs_index[1][:]], 0]]

    # Converts the angle from radians to degrees 
    # Note: This was for checking only!
    #radangle = radangle_pair[atomset1] * 180 / math.pi
    
    return atomset1, atomset2, radangle_pair, bondorder_values, ridge_points, distance

def dupl_flip(index):
    """
    This function flips a list of indicies and appends it to the original list. This allows
    you to cycle both forward and backwards through the atom list at the same time. It assumes
    that there are only two columns in the list (as are needed for the pairs list)
    """
    index_shape = np.shape(index)
    
    if index_shape[0] == 2:
        axis = 0
    elif index_shape[1] == 2:
        axis = 1
    else:
        axis = -1

    flip_index = np.zeros((index_shape), dtype = np.int64)

    try:
        if axis == 0:
            flip_index[0, :] = index[1, :]
            flip_index[1, :] = index[0, :]
            index_fin = np.concatenate((index,flip_index), axis = 1)
        elif axis == 1:
            flip_index[:, 0] = index[:, 1]
            flip_index[:, 1] = index[:, 0]
            index_fin = np.concatenate((index,flip_index), axis = 0)

    except IndexError:
        raise IndexError("There are more than two axes in the pairs list")
    
    return index_fin 

def calc_angles(n, radangle_first_pair, atomset1, atomset2):
    """
    This function calculates all of the angles between the different sets of points
    """
    realcomp_array = np.cos(n * radangle_first_pair[[atomset1[:]]])
    complexcomp_array = np.sin(n * radangle_first_pair[[atomset1[:]]])

    realcomp_array_norm = component_norm(realcomp_array[:], atomset2[:])
    complexcomp_array_norm = component_norm(complexcomp_array[:], atomset2[:])
    
    return realcomp_array_norm, complexcomp_array_norm, realcomp_array, complexcomp_array

def psin_calc(n, radangle_pair, atomset1, atomset2, desired_confined_atoms, ridge_points, distance, positions):
    """
    This function calculates the psi value for the the n-symmetry component
    """
    realcomp_array_norm, complexcomp_array_norm, realcomp_array, complexcomp_array = calc_angles(
           n, radangle_pair, atomset1, atomset2)

        # Need to determine and only select the closest four neighbours for the
        # 4-fold symmetry (n=4)
    if n == 4:
        bondordercomp, real_part, complex_part, bo_err, error_sum = psi4_calc(
            desired_confined_atoms, ridge_points, atomset1, realcomp_array, complexcomp_array, distance)
        #print bo_err

    else:
        real_part = np.sum(realcomp_array_norm, axis=0)
        complex_part = np.sum(complexcomp_array_norm, axis=0)
        bondordercomp = np.sqrt(real_part ** 2 + complex_part ** 2) / len(positions)

        error_sum = np.zeros(2)
        error_sum[0] = array_error(realcomp_array_norm, real_part, Num=len(positions))
        error_sum[1] = array_error(complexcomp_array_norm, complex_part, Num=len(positions))

        bo_err = bo_error ( real_part, error_sum[0], complex_part, error_sum[1]) / len(positions)

    #print bondordercomp, bo_err, n, "complete bondorder"

    return bondordercomp, real_part, complex_part, bo_err, error_sum

def array_error(array, mean, **Num):
    """
    This function calculates the standard error of an array of points
    """
    array -= mean
    arr_sq = np.power(array, 2)
    sq_sum = np.sum(arr_sq)
    std_dev = np.sqrt(sq_sum)
    
    if not Num:                            
        N = np.shape(array)[0]
    else:
        k_Num=Num.keys()
        N = Num[k_Num[0]]
    std_error = std_dev / math.sqrt(N)

    return std_error

def bo_error(r, a_r, c, a_c):
    """
    This function calculates the error on the bond order value based
    on the real component, r, and the complex component c and their
    associated errors, a_r and a_c respectively
    """
    #print r, a_r, c, a_c
    
    err_sqd_top = r**2 * a_r**2 + c**2 * a_c**2
    err_sqd_bot = r**2 + c**2

    
    err_sqd = np.divide(err_sqd_top, err_sqd_bot)
    
    a_bo = math.sqrt(err_sqd)

    return a_bo
