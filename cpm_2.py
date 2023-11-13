import math
import numpy as np
import random
import scipy.signal as ss
import matplotlib.pyplot as plt
import seaborn as sns

def toplot(X,Y,max_val):
    grid = np.zeros((max_val,max_val))
    for i in range(np.shape(X)[0]):
        grid[X[i],Y[i]] = 1
    sns.heatmap(grid)
    plt.show()

#Define variables and parameters
R = 11 #um, cell radius (half side length -- area = 4*R**2)
a = 2 #um, lattice size
tau = 1 #s, MC time step
d = a #um, pixel depth
T = 9*60*60+1 #s, total time
deltat = 15*60 #s, measurement interval
alpha = 1 #1/um, cell-ECM line energy
lambdaA = 1e-3 #1/um^4, area deviation penalty
Z = 1 #number of trials
A0 = 4*R*R #cell area (um)
L = 2*R//a #initial cell length in pixels
N = A0//(a**2) #average number of cell pixels
kmax = 10*N #max pixel number
grid = np.zeros((kmax,kmax))
grid2 = np.zeros((kmax,kmax))
indicies = np.arange(kmax)
cell_neighbor_filter = np.array([[0,1,0],[1,0,1],[0,1,0]])
void_neighbor_filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
print(indicies)
mp = kmax//2
b = 1 #buffer for sampling area
Nm = T//deltat #number of measurements (excluding t=0)
xcm = np.empty([Nm+1],float) #center of mass x positions
ycm = np.empty([Nm+1],float) #center of mass y positions
#Loop over trials
for z in range(Z):
    print('z = ', z)
    
    # Initialize cell pixel locations as close to center as possible
    Np = N #current number of cell pixels
    grid[mp-L//2:mp+L-L//2,mp-L//2:mp+L-L//2] = 1
    print(mp-L//2,mp+L-L//2)
    

    #Simulate
    t = 0 #s, time
    tn = 0 #next measurement time
    icm = 0 #index for cm vectors
    while t < T:
        if t >= tn:
            #Record (x,y) for center of mass
            xcm[icm] = np.sum(np.sum(grid,axis = 0)*indicies)/np.sum(np.sum(grid,axis = 0))
            ycm[icm] = np.sum(np.sum(grid, axis=1) * indicies) / np.sum(np.sum(grid, axis=1))
            tn += deltat
            icm = icm + 1

        # returns list of potentially "active" pixels
        # active just means the point has at least one neighbor with a state different than its own
        neg_n_filter = ss.convolve2d(grid, void_neighbor_filter)
        border_pixels = np.heaviside(np.abs(neg_n_filter),0)
        border_pixel_indices = np.argwhere(np.heaviside(np.abs(neg_n_filter),0) == 1)
        tt = 0
        while True:
            tt+=1
            # randomly picks a potentially active pixel to activate
            # minus 1 accounts for shift from convolution array size
            random_index = border_pixel_indices[np.random.randint(border_pixel_indices.shape[0]),:]- 1
            
            neighbor_count = ss.convolve2d(grid, cell_neighbor_filter)
            grid2[random_index[0],random_index[1]] += 1
            if tt%1000== 0:
                sns.heatmap(grid2+grid*500)
                plt.show()
        print(neg_n_filter[random_index[0],random_index[1]])
        rand_pix = np.random.randint(1,5)
        if rand_pix*grid[random_index[0],random_index[1]] <= neg_n_filter[random_index[0],random_index[1]] and rand_pix*(1-grid[random_index[0],random_index[1]]) <= neighbor_count[random_index[0],random_index[1]]:
            print(grid[random_index[0],random_index[1]], rand_pix, neg_n_filter[random_index[0],random_index[1]] , neighbor_count[random_index[0],random_index[1]])
        neighbor_count[random_index[0], random_index[1]] += 8
        sns.heatmap(neighbor_count)
        
        plt.show()
        input()
        #Find min/max x and y locations
        xmin = np.min(X)
        xmax = np.max(X)
        ymin = np.min(Y)
        ymax = np.max(Y)

        #Calculate number of samples
        NT = (xmax - xmin + 2*b + 1)*(ymax - ymin + 2*b + 1) #Number of trials in a MC time step
        
        #Loop over samples
        for l in range(NT):
            #Randomly pick a pixel
            r1 = random.randint(0, (xmax - xmin + 2*b)//1)
            r2 = random.randint(0, (ymax - ymin + 2*b)//1)
            x1 = (xmin - b + r1)//1
            y1 = (ymin - b + r2)//1
            
            #Randomly pick a neighbor of that pixel
            r3 = random.randint(0,3)
            if r3 == 0:
                x2 = x1 - 1
                y2 = y1
            if r3 == 1:
                x2 = x1 + 1
                y2 = y1
            if r3 == 2:
                x2 = x1
                y2 = y1 - 1
            if r3 == 3:
                x2 = x1
                y2 = y1 + 1
            
            #Find identities of pixels and neighboors
            sigma1 = 0 #pixel identities
            sigma2 = 0
            for k in range(Np):
                if (X[k] == x1) and (Y[k] == y1):
                    sigma1 = 1
                if (X[k] == x2) and (Y[k] == y2):
                    sigma2 = 1
                    #Save neighbor's index in case it becomes ECM
                    kn = k #index of neighbor
                    
            #If identities are different, try to copy pixel identity to neighbor
            #Case 1: pixel is cell and neighbor is ECM
            if (sigma1 == 1) and (sigma2 == 0):
                #Find perimiter change based on neighbor's neighbors
                nn = 0 #number of neighbor's neighbors
                for k in range(Np):
                    if (X[k] == x2 + 1) and (Y[k] == y2):
                        nn += 1
                    if (X[k] == x2 - 1) and (Y[k] == y2):
                        nn += 1
                    if (X[k] == x2) and (Y[k] == y2 + 1):
                        nn += 1
                    if (X[k] == x2) and (Y[k] == y2 - 1):
                        nn += 1
                if nn == 1:
                    dN = 2
                if nn == 2:
                    dN = 0
                if nn == 3:
                    dN = -2
                if nn == 4:
                    dN = -4
                if nn > 4:
                    print("alert")
                    input()
                if nn == 0:
                    print("alert")
                    input()
                
                #Calculate energy change
                du = alpha*a*dN + lambdaA*a*a*a*a*((Np+1-N)*(Np+1-N) - (Np-N)*(Np-N)) #du = energy change
                
                #Attempt the move
                r4 = random.random()
                if r4 < np.exp(-du):
                    X[Np] = x2
                    Y[Np] = y2
                    Np += 1
                
            #Case 2: pixel is ECM and neighbor is cell
            ### if statements were the same
            ### Making this one sigma1 = 0 and sigma2 = 1 fixes stuff
            if (sigma1 == 0) and (sigma2 == 1):
                #Find perimiter change based on neighbor's neighbors
                nn = 0

                for k in range(Np):
                    if (X[k] == x2 + 1) and (Y[k] == y2):
                        nn += 1
                    if (X[k] == x2 - 1) and (Y[k] == y2):
                        nn += 1
                    if (X[k] == x2) and (Y[k] == y2 + 1):
                        nn += 1
                    if (X[k] == x2) and (Y[k] == y2 - 1):
                        nn += 1
                if nn == 0:
                    dN = -4
                if nn == 1:
                    dN = -2
                if nn == 2:
                    dN = 0
                if nn == 3:
                    dN = 2
                if nn > 3:
                    print("alert")
                    input()

                #Calculate energy change
                du = alpha*a*dN + lambdaA*a*a*a*a*((Np-1-N)*(Np-1-N) - (Np-N)*(Np-N))
                
                #Attempt the move
                r4 = random.random()
                if r4 < np.exp(-du):
                    #Remove the neighbor pixel from the cell list and shift
                    for k in range(kn, Np-1):
                        X[k] = X[k+1]
                        Y[k] = Y[k+1]
                    X[Np-1] = 0
                    Y[Np-1] = 0
                    Np -= 1
        #Update time
        t += tau
    
    #Compute statistics
    #Speed
    vsum = 0
    for i in range(Nm):
        vsum += a*math.sqrt((xcm[i+1]-xcm[i])*(xcm[i+1]-xcm[i]) + (ycm[i+1]-ycm[i])*(ycm[i+1]-ycm[i]))/deltat
    v = vsum/Nm
    
    #CI
    CI = (xcm[Nm]-xcm[0])/ math.sqrt((xcm[Nm]-xcm[0])*(xcm[Nm]-xcm[0]) + (ycm[Nm]-ycm[0])*(ycm[Nm]-ycm[0]))
    
    #CR
    dist = 0
    for i in range(Nm):
        dist += math.sqrt((xcm[i+1]-xcm[i])*(xcm[i+1]-xcm[i]) + (ycm[i+1]-ycm[i])*(ycm[i+1]-ycm[i]))
    CR = math.sqrt((xcm[Nm]-xcm[0])*(xcm[Nm]-xcm[0]) + (ycm[Nm]-ycm[0])*(ycm[Nm]-ycm[0]))/dist
    
    print(v)
    print(CI)
    print(CR)