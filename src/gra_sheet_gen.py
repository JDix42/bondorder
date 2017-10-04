import numpy as np
import matplotlib.pyplot as plt
import math

def graphene_create():
    """
    This function will return the coordinates of a graphene ring
    in angstroms in a numpy array
    """
    ring = np.zeros((6,3))#Atom 1 is centered at 0,0
    ang = 30 * math.pi /180
    cos_ang = math.cos(ang)
    sin_ang = math.sin(ang)
    ring[1,:] = [0, 1.42,0]
    ring[2,:] = [1.42*cos_ang, 1.42*(1+sin_ang),0]
    ring[3,:] = [2.84*cos_ang, 1.42,0]
    ring[4,:] = [2.84*cos_ang, 0,0]
    ring[5,:] = [1.42*cos_ang, -1.42*sin_ang,0]

    ring_shift = position_center(ring)

    return ring_shift

def replicate_graphene(unitcell, n, m):
    """
    This function replicates the graphene hexagonal structure into
    a cubic 3D cell by n unit in the y direcition and m units in
    the x direction
    """
    rangey = np.max(unitcell[:,1]) - np.min(unitcell[:,1])

    maxx = np.max(unitcell[:,0])    
    avgx = np.average(unitcell[:,0])
    rangex = np.max(unitcell[:,0]) - np.min(unitcell[:,0])
    
    bond = 1.42
    fudge_factor = bond*math.cos(30*math.pi/180)*0.1 #This accounts for numerical rounding error in 
    #distance cut off

    orig_points = np.shape(unitcell)[0]
    if m==0:
        rep_points = 3*n 
    elif n==0 and m==1:
        rep_points = 2*m
    elif n==0 and m>1:
        rep_points = 2 + 4*(m-1)
    else:
        rep_points = 3*n + 2 + n  + (4+2*n)*(m-1)

    tot_points = orig_points + rep_points
    #print tot_points, orig_points, rep_points

    new_points = np.zeros((tot_points, 3))
    new_points[:orig_points, :] = unitcell[:,:]
    
    if n != 0:
        for rep in range(n):
            dup = rep +1
            if dup % 2 == 0 and dup>0:
                shift_pos = even_yshift(unitcell, dup, rangey, 1.42)
 
            elif dup>0:
                shift_pos = odd_yshift(unitcell, dup, rangey, 1.42)
            new_points[orig_points+(rep*3):orig_points+(rep+1)*3,:] = shift_pos[:, :]
    
    if m != 0:
        rep_sum = 0 
        for rep in range(m):
            dup = rep+1
            shift_pos = new_points[:orig_points+3*n, 0] + rangex*dup
            if dup == 1:
                shift_ndx = np.where((shift_pos[:]>=(avgx - fudge_factor + rangex*dup)) & (shift_pos[:]<=(avgx + fudge_factor + rangex*dup)))
                #print np.shape(shift_ndx)

            else:
                shift_ndx = np.where((shift_pos[:]<=(avgx + rangex*dup + fudge_factor)))
            #print np.shape(shift_ndx), shift_ndx
            #print orig_points+3*n+rep_sum, orig_points+3*n + rep_sum + np.shape(shift_ndx)[1]

            new_points[orig_points+3*n+rep_sum:orig_points+3*n + rep_sum + np.shape(shift_ndx)[1], 0] = shift_pos[shift_ndx]
            new_points[orig_points+3*n+rep_sum:orig_points+3*n + rep_sum + np.shape(shift_ndx)[1], 1] = new_points[shift_ndx,1]
            new_points[orig_points+3*n+rep_sum:orig_points+3*n + rep_sum + np.shape(shift_ndx)[1], 2] = new_points[shift_ndx,2]
            
            rep_sum += np.shape(shift_ndx)[1]

            #print np.shape(new_points)
        
    return new_points

    
def odd_yshift(unitcell, rep, rangey, bond):
    """
    This function shifts the unitcell of graphene in the y-direction
    if the shift is for an odd replica
    """
    avgy = np.average(unitcell[:,1])
    
    bothalf_ndx = np.where(unitcell[:,1]<avgy)
    
    shift =  (rangey + bond)* (rep+1)/2

    shift_pos = np.zeros((np.shape(bothalf_ndx)[1],3))

    shift_pos[:,1] = unitcell[bothalf_ndx,1] + (rangey + bond)* (rep+1)/2
    shift_pos[:,0] = unitcell[bothalf_ndx,0]
    shift_pos[:,2] = unitcell[bothalf_ndx,2]

    return shift_pos

def even_yshift(unitcell, rep, rangey, bond):
    """
    This function shifts the unitcell of graphene in the y-direction
    if the shift is for an even replica
    """
    avgy = np.average(unitcell[:,1])
    
    tophalf_ndx = np.where(unitcell[:,1]>avgy)
    
    shift =  (rangey + bond)*rep/2

    shift_pos = np.zeros((np.shape(tophalf_ndx)[1],3))
    
    shift_pos[:,1] = unitcell[tophalf_ndx,1] + (rangey + bond)*rep/2    
    shift_pos[:,0] = unitcell[tophalf_ndx,0]
    shift_pos[:,2] = unitcell[tophalf_ndx,2]

    return shift_pos
    
    

def replicate_posy(unitcell, points,n):
    """
    This function replicates a lattice structure in the up
    direction (+ve y-axis) for n replicas
    """
    
    maxy = np.max(unitcell[:,1])
    miny = np.min(unitcell[:,1])
    
    rangey = maxy-miny
    
    rep_point_ndx = np.where( unitcell[:,1] != miny)
    
    orig_points = np.shape(points)[0]
    rep_points = np.shape(rep_point_ndx)[1]

    
    new_points = np.zeros((orig_points + rep_points*n, 3))

    new_points[:orig_points,:] = points[:,:]
    for rep in range(n):
        shift = rangey * (rep+1)
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,0] = points[rep_point_ndx, 0]
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,1] = points[rep_point_ndx, 1]+shift
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,2] = points[rep_point_ndx, 2]

    return new_points    

def replicate_negy(unitcell, points,n):
    """
    This function replicates a lattice structure in the up
    direction (+ve y-axis) for n replicas
    """
    
    maxy = np.max(unitcell[:,1])
    miny = np.min(unitcell[:,1])
    
    rangey = maxy-miny
    
    rep_point_ndx = np.where( unitcell[:,1] != maxy)
    
    orig_points = np.shape(points)[0]
    rep_points = np.shape(rep_point_ndx)[1]

    
    new_points = np.zeros((orig_points + rep_points*n, 3))

    new_points[:orig_points,:] = points[:,:]
    for rep in range(n):
        shift = rangey * (rep+1)
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,0] = points[rep_point_ndx, 0]
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,1] = points[rep_point_ndx, 1]-shift
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,2] = points[rep_point_ndx, 2]

    return new_points    

def replicate_posx(unitcell, points,n):
    """
    This function replicates a lattice structure of the unitcell
    in the up direction (+ve y-axis) for n replicas and adds it to
    the positions in points
    """
    
    maxx = np.max(unitcell[:,0])
    minx = np.min(unitcell[:,0])
    
    rangex = maxx-minx
    
    rep_point_ndx = np.where( unitcell[:,0] != minx)
    
    orig_points = np.shape(points)[0]
    rep_points = np.shape(rep_point_ndx)[1]

    
    new_points = np.zeros((orig_points + rep_points*n, 3))

    new_points[:orig_points,:] = points[:,:]
    for rep in range(n):
        shift = rangex * (rep+1)
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,0] = points[rep_point_ndx, 0]+shift
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,1] = points[rep_point_ndx, 1]
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,2] = points[rep_point_ndx, 2]

    return new_points

def replicate_negx(unitcell,points,n):
    """
    This function replicates a lattice structure in the up
    direction (+ve y-axis) for n replicas
    """
    
    maxx = np.max(unitcell[:,0])
    minx = np.min(unitcell[:,0])
    
    rangex = maxx-minx
    
    rep_point_ndx = np.where( unitcell[:,0] != maxx)
    
    orig_points = np.shape(points)[0]
    rep_points = np.shape(rep_point_ndx)[1]

    
    new_points = np.zeros((orig_points + rep_points*n, 3))

    new_points[:orig_points,:] = points[:,:]
    for rep in range(n):
        shift = rangex * (rep+1)
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,0] = points[rep_point_ndx, 0]-shift
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,1] = points[rep_point_ndx, 1]
        new_points[orig_points+(rep)*rep_points:orig_points+(rep+1)*rep_points,2] = points[rep_point_ndx, 2]

    return new_points

def position_center(pos):
    """
    This function centers the positions about the central point
    """
    avgx = np.average(pos[:,0])
    avgy = np.average(pos[:,1])

    pos[:,0] -= avgx
    pos[:,1] -= avgy

    return pos

