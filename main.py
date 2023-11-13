import math
import numpy as np
import random
import matplotlib.pyplot as plt

# Define variables and parameters
R = 10  # um, cell radius (half side length -- area = 4*R**2)
a = 2  # um, lattice size
tau = 1  # s, MC time step
d = a  # um, pixel depth
T = 9 * 60 * 60 + 1  # s, total time
deltat = 15 * 60  # s, measurement interval
alpha = 1  # 1/um, cell-ECM line energy
lambdaA = 1e-3  # 1/um^4, area deviation penalty
Z = 1  # number of trials
A0 = 4 * R * R  # cell area (um)
L = 2 * R // a  # initial cell length in pixels
N = A0 // (a ** 2)  # average number of cell pixels
kmax = 10 * N  # max pixel number
X = np.empty([kmax], int)  # cell pixel x locations
Y = np.empty([kmax], int)  # cell pixel y locations
b = 1  # buffer for sampling area
Nm = T // deltat  # number of measurements (excluding t=0)
xcm = np.empty([Nm + 1], float)  # center of mass x positions
ycm = np.empty([Nm + 1], float)  # center of mass y positions

# Loop over trials
for z in range(0, Z, 1):
    print('z = ', z)

    # Initialize cell pixel locations (with buffer zeros at end)
    Np = N  # current number of cell pixels
    for i in range(0, L, 1):
        for j in range(0, L, 1):
            k = j + L * i
            X[k] = i
            Y[k] = j
    for k in range(Np, kmax, 1):
        X[k] = 0
        Y[k] = 0

    # Simulate
    t = 0  # s, time
    tn = 0  # next measurement time
    icm = 0  # index for cm vectors
    while t < T:

if t >= tn:
    # Record (x,y) for center of mass
    xcm[icm] = 0
    ycm[icm] = 0
    for k in range(0, Np, 1):
        xcm[icm] += X[k]
        ycm[icm] += Y[k]
    print('xcm[icm] = ', xcm[icm])
    xcm[icm] = xcm[icm] / Np
    print('ycm[icm] = ', ycm[icm])
    ycm[icm] = ycm[icm] / Np
    tn += deltat
    icm = icm + 1

# Find min/max x and y locations
xmin = X[0]
xmax = X[0]
ymin = Y[0]
ymax = Y[0]
for k in range(0, Np, 1):
    if X[k] < xmin:
        xmin = X[k]
    if X[k] > xmax:
        xmax = X[k]
    if Y[k] < ymin:
        ymin = Y[k]
    if Y[k] > ymax:
        ymax = Y[k]

# Calculate number of samples
NT = (xmax - xmin + 2 * b + 1) * (ymax - ymin + 2 * b + 1)  # Number of trials in a MC time step

# Loop over samples
for l in range(0, NT, 1):
    
    # Randomly pick a pixel
    r1 = random.randint(0, (xmax - xmin + 2 * b) // 1)
    r2 = random.randint(0, (ymax - ymin + 2 * b) // 1)
    x1 = (xmin - b + r1) // 1
    y1 = (ymin - b + r2) // 1

    # Randomly pick a neighbor of that pixel
    r3 = random.randint(0, 3)
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

    # Find identities of pixels and neighboors
    sigma1 = 0  # pixel identities
    sigma2 = 0
    for k in range(0, Np, 1):
        if (X[k] == x1) and (Y[k] == y1):
            sigma1 = 1
        if (X[k] == x2) and (Y[k] == y2):
            sigma2 = 1
            # Save neighbor's index in case it becomes ECM
            kn = k  # index of neighbor

    # If identities are different, try to copy pixel identity to neighbor
    # Case 1: pixel is cell and neighbor is ECM
    if (sigma1 == 1) and (sigma2 == 0):
        # Find perimiter change based on neighbor's neighbors
        nn = 0  # number of neighbor's neighbors
        for k in range(0, Np, 1):
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
            print('nn = ', nn)
        if nn == 0:
            print('nn = ', nn)
        print('case 1 nn = ', nn)

        # Calculate energy change
        du = alpha * a * dN + lambdaA * a * a * a * a * (
                    (Np + 1 - N) * (Np + 1 - N) - (Np - N) * (Np - N))  # du = energy change

        # Attempt the move
        r4 = random.random()
        if r4 < np.exp(-du):
            X[Np] = x2
            Y[Np] = y2
            Np += 1

    # Case 2: pixel is ECM and neighbor is cell
    if (sigma1 == 1) and (sigma2 == 0):
        
        # Find perimiter change based on neighbor's neighbors
        nn = 0
        for k in range(0, Np, 1):
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
            print('nn = ', nn)

        # Calculate energy change
        du = alpha * a * dN + lambdaA * a * a * a * a * ((Np - 1 - N) * (Np - 1 - N) - (Np - N) * (Np - N))

        # Attempt the move
        r4 = random.random()
        if r4 < np.exp(-du):
            # Remove the neighbor pixel from the cell list and shift
            for k in range(kn, Np - 1, 1):
                X[k] = X[k + 1]
                Y[k] = Y[k + 1]
            X[Np - 1] = 0
            Y[Np - 1] = 0
            Np -= 1
# Update time
t += tau

# Compute statistics
# Speed
vsum = 0
for i in range(0, Nm, 1):
    vsum += a * math.sqrt(
        (xcm[i + 1] - xcm[i]) * (xcm[i + 1] - xcm[i]) + (ycm[i + 1] - ycm[i]) * (ycm[i + 1] - ycm[i])) / deltat
v = vsum / Nm

# CI
CI = (xcm[Nm] - xcm[0]) / math.sqrt((xcm[Nm] - xcm[0]) * (xcm[Nm] - xcm[0]) + (ycm[Nm] - ycm[0]) * (ycm[Nm] - ycm[0]))

# CR
dist = 0
for i in range(0, Nm, 1):
    dist += math.sqrt((xcm[i + 1] - xcm[i]) * (xcm[i + 1] - xcm[i]) + (ycm[i + 1] - ycm[i]) * (ycm[i + 1] - ycm[i]))
CR = math.sqrt((xcm[Nm] - xcm[0]) * (xcm[Nm] - xcm[0]) + (ycm[Nm] - ycm[0]) * (ycm[Nm] - ycm[0])) / dist

print(v)
print(CI)
print(CR)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
