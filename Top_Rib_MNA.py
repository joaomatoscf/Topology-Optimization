"""
Created on Tue Mar 10 13:18:15 2020

@author: João Matos
"""

#from __future__ import division
import sys

### Imports for calculations
import numpy as np
from numpy.lib import scimath
import numpy.matlib
import math

### Import for solver
import scipy
from scipy.sparse import diags
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

### Imports for plots
from matplotlib import colors
import matplotlib.pyplot as plt

def main(nelx=None, nely=None, volfrac=None, maxiteration=None):

    penal=3
    tol = 0.01
    E0 = 1
    Emin = 1e-9
    nu = 0.3
    mMax = nelx * nely * volfrac
    d = 3 * nelx / 8
    R = 6 * nely                    # radius
    Rint1 = nely / 4
    Rint2 = nely / 4
    Rint3 = nely / 6
    a = np.floor((np.sqrt((R) ** 2 - (d) ** 2)) - R + nely)  # edges
    b = np.floor((np.sqrt((R) ** 2 - (nelx - d) ** 2)) - R + nely)
    f = int(np.floor(((nelx / 4) + Rint1 + (((nelx / 4) - Rint1 - Rint2) / 2) - (nelx / 80) + 1)))  # white square dimensions
    g = int(np.floor(((nelx / 4) + Rint1 + (((nelx / 4) - Rint1 - Rint2) / 2) + (nelx / 80))))

    ## INITIAL DESIGN
    nX = 16         #number of components along x axis
    nY = 4          #number of components along y axis

    l_ini = np.sqrt(volfrac * nelx * nely / (0.5 * nX * nY)) #Initial length of components
    xp = np.linspace(1 / (nX + 1) * nelx, nX / (nX + 1) * nelx, nX)
    yp = np.linspace(1 / (nY + 1) * nely, nY / (nY + 1) * nely, nY)
    xElts, yElts = np.meshgrid(xp, yp)
    xElts = np.ravel(xElts[np.newaxis].T)
    yElts = np.ravel(yElts[np.newaxis].T)
    n_c = xElts.shape[0]
    x = np.ravel(np.concatenate(([[xElts], [yElts], [np.zeros(n_c)], [l_ini*np.ones(n_c)], [l_ini*np.ones(n_c)]])).T)
    n_var = x.shape
    x = np.reshape(x,n_var)

    emptyelts = []  # Obligatory empty elements
    fullelts = []   # Obligatory full elements
    passivewhite = np.zeros((nely, nelx))

    ely2 = 0
    while ely2 < nely:  ###white external round
        elx2 = 0
        while elx2 < nelx:
            if np.sqrt((ely2+1 - R) ** 2 + (elx2+1 - d) ** 2) > R:
                emptyelts.append(nely * elx2 + ely2)
                passivewhite[ely2, elx2] = 1
            elx2 = elx2 + 1
        ely2 = ely2 + 1

    ely3 = 0
    while ely3 < nely:  ###white external round
        elx3 = 0
        while elx3 < nelx:
            if np.sqrt((ely3+1 - ((3 * nely) / 5)) ** 2 + (elx3+1 - (nelx / 4)) ** 2) <= Rint1:
                emptyelts.append(nely * elx3 + ely3)
                passivewhite[ely3, elx3] = 1
            elx3 = elx3 + 1
        ely3 = ely3 + 1

    ely4 = 0
    while ely4 < nely:  ###white external round
        elx4 = 0
        while elx4 < nelx:
            if np.sqrt((ely4+1 - ((3 * nely) / 5)) ** 2 + (elx4+1 - (nelx / 2)) ** 2) <= Rint2:
                emptyelts.append(nely * elx4 + ely4)
                passivewhite[ely4, elx4] = 1
            elx4 = elx4 + 1
        ely4 = ely4 + 1

    ely5 = 0
    while ely5 < nely:  ###white external round
        elx5 = 0
        while elx5 < nelx:
            if np.sqrt((ely5+1 - ((3 * nely) / 5)) ** 2 + (elx5+1 - (3 * nelx / 4)) ** 2) <= Rint3:
                emptyelts.append(nely * elx5 + ely5)
                passivewhite[ely5, elx5] = 1
            elx5 = elx5 + 1
        ely5 = ely5 + 1


    ely6 = int(nely / 10)
    while ely6 < 9 * nely / 10:  ###white external round
        elx6 = f - 1
        while elx6 < g:
            emptyelts.append(nely * elx6 + ely6)
            passivewhite[ely6, elx6] = 1
            elx6 = elx6 + 1
        ely6 = ely6 + 1



    ### PREPARE FINITE ELEMENT ANALYSIS
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE = lk()

    # FE: Build the index vectors for the for coo matrix format.
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely
            n1 = (nely + 1) * elx + ely
            n2 = (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = np.array(
                [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1])

    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    ###LOADS AND SUPPORTS
    dinx1 = int(2 * (nely - a + 1) - 1) - 1  # above
    diny1 = int(2 * (nely - a + 1)) - 1
    dinx2 = int(2 * ((nely + 1) * (d) + 1) - 1) - 1
    diny2 = int(2 * ((nely + 1) * (d) + 1)) - 1
    dinx3 = int(2 * ((nely + 1) * (2 * d) + nely - a + 1) - 1) - 1
    diny3 = int(2 * ((nely + 1) * (2 * d) + nely - a + 1)) - 1
    dinx4 = int(2 * ((nely + 1) * (nelx) + nely - b + 1) - 1) - 1
    diny4 = int(2 * ((nely + 1) * (nelx) + nely - b + 1)) - 1

    dout1 = int(2 * ((nely + 1) * f + (nely / 10) + 1)) - 1
    dout3 = int(2 * ((nely + 1) * g + (nely / 10) + 1)) - 1

    dinx5 = int(2 * ((nely + 1)) - 1)  - 1 # below
    diny5 = int(2 * ((nely + 1))) - 1
    dinx6 = int(2 * ((nely + 1) * (d + 1)) - 1) - 1
    diny6 = int(2 * ((nely + 1) * (d + 1))) - 1
    dinx7 = int(2 * ((nely + 1) * (2 * d + 1)) - 1) - 1
    diny7 = int(2 * ((nely + 1) * (2 * d + 1))) - 1
    dinx8 = int(2 * ((nely + 1) * (nelx + 1)) - 1) - 1
    diny8 = int(2 * ((nely + 1) * (nelx + 1))) - 1

    dout2 = int(2 * ((nely + 1) * f + (9 * nely) / 10)) - 1
    dout4 = int(2 * ((nely + 1) * g + (9 * nely) / 10)) - 1

    F = np.zeros((ndof, 1))  # Force array Initialization
    F[dinx1,0] = 1  # above
    F[diny1,0] = 1
    F[dinx2,0] = 1
    F[diny2,0] = 1
    F[dinx3,0] = 1
    F[diny3,0] = 1
    F[dinx4,0] = 1
    F[diny4,0] = 1

    F[dout1,0] = 1
    F[dout3,0] = 1

    F[dinx5,0] = -1  # below
    F[diny5,0] = -1
    F[dinx6,0] = -1
    F[diny6,0] = -1
    F[dinx7,0] = -1
    F[diny7,0] = -1
    F[dinx8,0] = -1
    F[diny8,0] = -1
    F[dout2,0] = -1
    F[dout4,0] = -1

    fixeddofs1 = np.arange((2 * (nely - a + 1 + 1) - 1) - 1, (2 * (nely + 1 - 1) - 1))  # Fixed degrees of freedom
    fixeddofs2 = np.arange((2 * (nely - a + 1 + 1) - 1), (2 * (nely + 1 - 1)))
    fixeddofs = np.union1d(fixeddofs1, fixeddofs2)
    alldofs = np.array(range(0,2*(nely+1)*(nelx+1)))
    freedofs = np.setdiff1d(alldofs, fixeddofs)         #Free degrees of freedom
    U = np.zeros((ndof, 1))


    ## Normalized Variables
    Xmin = np.tile(np.concatenate(([0], [0], [-np.pi], [0], [0])), n_c)
    Xmax = np.tile(np.concatenate(([nelx], [nely], [np.pi], [nelx], [nely])), n_c)
    x = (x - Xmin) / (Xmax - Xmin)

    ## Initialize Optimization Process
    change = 1
    mm = 1
    n_var = x.shape[0]
    xval = x
    xold1 = xval
    xold2 = xval
    xmin = np.zeros(n_var)
    xmax = np.ones(n_var)
    low = xmin
    upp = xmax
    C = 1000 * np.ones(mm)
    d_ = np.zeros(mm)
    a0 = 1
    aa = np.zeros(mm)
    iteration = 0
    kkttol = tol
    kktnorm = kkttol + 10
    X_old = np.zeros(x.shape)
    X = Xmin + (Xmax - Xmin) * x
    
    cvec = np.zeros(maxiteration)       #Vector with objective function for every iteration
    cstvec = np.zeros(maxiteration)       #Vector with the volume fraction for every iteration
    plot_rate = 1

    # initialize variables for plot
    tt = np.arange(0, 2 * np.pi, 0.005)
    tt = np.kron((np.ones(n_c)),tt[np.newaxis].T).T
    cc = np.cos(tt)
    ss = np.sin(tt)

    # Initialize plot and plot the initial design
    plt.ion()
    fig = plt.figure()
    # Density Plot
    den = np.zeros(nelx * nely)    
    ttt = np.reshape(den,(nely,nelx))
    ax1 = fig.add_subplot(221)
    ax1.title.set_text('Density Plot')
    im1 = ax1.imshow(-ttt.reshape((nelx, nely)).T, cmap='gray', interpolation='none',
                     norm=colors.Normalize(vmin=-1, vmax=0))
    ax1.axis('off')
    
    # Components Plot
    ax2 = fig.add_subplot(222)
    ax2.title.set_text('Components Plot')
    ax2.axis('off')
    ax2.set_xlim([0, nelx])
    ax2.set_ylim([0, nely])
    ax2.set_aspect(aspect=1)
    
    # Objective Function PLot
    ax3 = fig.add_subplot(223)
    ax3.title.set_text('Objective Function')
    ax3.set_ylabel('c')
    ax3.set_xlabel('Iterations')
    #ax3.set_ylim([0, 1000])
    # Volume Fraction Plot
    ax4 = fig.add_subplot(224, sharex=ax3)
    ax4.title.set_text('Constraint Function')
    ax4.set_ylabel('v')
    ax4.set_xlabel('Iterations')
    #ax4.set_ylim([0, volfrac * 1.1])
    
    fig.show()

    ## OPTIMIZATION LOOP
    while np.logical_and(np.logical_and(change > tol, iteration <= maxiteration) , kktnorm > tol):
        iteration = iteration + 1
        x_old = x

        # Density and Density's gradient
        den = np.zeros((nelx * nely,1))             # Density
        grad_den = np.zeros((nelx * nely, n_var))   # Gradient of the Density

        for ii in np.arange(0, n_c):                # Loop on the components
            n = 5 * (ii)
            ln = np.sqrt(X[n+3] ** 2 + X[n+4] ** 2)
            sn = np.sin(X[n+2])
            cn = np.cos(X[n+2])

            # Find Neighbours
            limx = max(min(math.floor(X[n] - ln), nelx), 1)
            
            limx = [max(min(math.floor(X[n] - ln), nelx), 1), max(min(math.floor(X[n] + ln), nelx), 1)]
            limy = [max(min(math.floor(X[n + 1] - ln), nely), 1), max(min(math.floor(X[n + 1] + ln), nely), 1)]
            xN, yN = np.meshgrid(np.arange(limx[0], limx[1] + 0.1) - 0.5, np.arange(limy[0], limy[1] + 0.1) - 0.5)

            xN = np.ravel(xN.T)
            yN = np.ravel(yN.T)

            neigh = np.concatenate([[((xN - X[n]) * cn + (yN - X[n + 1]) * sn) / X[n + 3]],[(-(xN - X[n]) * sn + (yN - X[n + 1]) * cn) / X[n + 4]]])
            nc =  np.logical_and(np.abs(neigh[0])<1 , np.abs(neigh[1]) < 1)

            ind1 = np.multiply(np.ones((limy[1] - limy[0] + 1, 1)), (np.arange(limx[0], limx[1] + 0.1) - 1) * nely) #+ np.arange(limy[0], limy[1] + 0.1).T * np.ones((1, limx[1] - limx[0] + 1))
            ind2 = np.multiply(np.arange(limy[0], limy[1] + 0.1)[np.newaxis].T , np.ones((1, limx[1] - limx[0] + 1))) #+ np.arange(limy[0], limy[1] + 0.1).T * np.ones((1, limx[1] - limx[0] + 1))

            ind = np.reshape(ind1 + ind2,(limx[1] - limx[0] + 1) * (limy[1]- limy[0] + 1), 1)
            ind = ind[nc] - 1      # Global index of neighbours
            ind = ind.astype(int)

            ## Compute Densities
            v = (neigh.T)[nc].T
            sx = np.sign(v[0])
            sy = np.sign(v[1])
            v = abs(v)

            p1 = v[0] < 0.5
            p2 = v[1] < 0.5

            wx = np.zeros(v.shape[1])
            wy = np.zeros(v.shape[1])
            dfdx = np.zeros(v.shape[1])
            dfdy = np.zeros(v.shape[1])

            wx[p1] = 1 - 6 * v[0,p1] ** 2 + 6 * v[0,p1] ** 3
            wx[~p1] = 2 - 6 * v[0,~p1] + 6 * v[0,~p1] ** 2 - 2 *v[0,~p1] ** 3
            wy[p2] = 1 - 6 * v[1, p2] ** 2 + 6 * v[1, p2] ** 3
            wy[~p2] = 2 - 6 * v[1, ~p2] + 6 * v[1, ~p2] ** 2 - 2 * v[1, ~p2] ** 3

            f = abs(wx * wy)

            dfdx[p1] = -12 * v[0,p1] + 18 * v[0,p1] ** 2
            dfdx[~p1] = -6 + 12 * v[0, ~p1] - 6 * v[0, ~p1] ** 2
            dfdx = dfdx * sx * wy
            dfdy[p2] = -12 * v[1, p2] + 18 * v[1, p2] ** 2
            dfdy[~p2] = -6 + 12 * v[1, ~p2] - 6 * v[1, ~p2] ** 2
            dfdy = dfdy * sy * wx

            dfdx = dfdx / X[n+3]
            dfdy = dfdy / X[n+4]
            den[ind] = den[ind] + f[np.newaxis].T
            grad_den[ind, n] = -cn * dfdx + sn * dfdy
            grad_den[ind, n + 1] = -sn * dfdx - cn * dfdy
            grad_den[ind, n + 2] = (- (xN[nc] - X[n]) * sn + (yN[nc] - X[n + 1]) * cn).T * dfdx - ((xN[nc] - X[n]) * cn + (yN[nc] - X[n + 1]) * sn).T * dfdy
            grad_den[ind, n + 3] = -((xN[nc] - X[n]) * cn + (yN[nc] - X[n + 1]) * sn).T * dfdx / X[n+3]
            grad_den[ind, n + 4] = ((xN[nc] - X[n]) * sn -(yN[nc] - X[n + 1]) * cn).T * dfdy / X[n+4]

        ix = np.where (den < 1)[0]
        nix = np.where (den >= 1)[0]
        grad_den[ix] = np.dot(np.diag(1 + 12 * den[np.where (den < 1)] ** 2 - 28 * den[np.where (den < 1)] ** 3 + 15 * den[np.where (den < 1)] ** 4), grad_den[ix])
        grad_den[nix] = 0
        den = den + 4 * den ** 3 - 7 * den ** 4 + 3 * den ** 5
        den[fullelts] = 1
        grad_den[fullelts] = 0
        den[emptyelts] = 0
        grad_den[emptyelts] = 0
        den = np.min(den,1)
        E = Emin + den.T ** penal * (E0 - Emin)
        bb = np.multiply(np.ravel(KE)[np.newaxis].T, np.ravel(E.T)[np.newaxis])

        ### FE ANALYSIS
        sK = np.ravel(bb.T)
        K = scipy.sparse.coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        K = K[freedofs, :][:, freedofs]
        U[freedofs,0] = scipy.sparse.linalg.spsolve(K, F[freedofs,0])

        # Compliance and it's gradient calculation
        U1 = U[:,0]
        ce1 = np.sum(np.dot(U1[edofMat], KE) * U1[edofMat],1)
        ce = ce1
        c = np.sum(np.sum(E * ce))
        dc = - penal * (E0 - Emin) * np.dot(grad_den.T, den ** (penal - 1) * ce)
        cst = (np.sum(den) - mMax) / mMax * 100
        dcst = np.sum(grad_den, axis = 0) / mMax * 100
        dc = dc * (Xmax - Xmin)
        dcst = dcst * (Xmax - Xmin).T

        change = np.max(np.abs(X_old - X))
        X_old = X
        f0val = c / 10000
        fval = cst
        df0dx = dc / 10000
        dfdx_ = dcst.T
        
        cvec[iteration-1]=c         #records objective function for every iteration
        cstvec[iteration-1]=cst     #records volume for every iteration
        
        ## MMA code optimization
        x, ymma, zmma, lam, xsi, eta, mu, zet, S, low, upp = mmasub(mm, n_var, iteration, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx_, low, upp, a0, aa, C, d_)
        xold2 = xold1
        xold1 = xval
        xval = x

        # Print iteration results
        print("It.: {0} , Obj.: {1:.3f} Const.: {2:.3f}, kktnorm.: {3:.3f}  ch.: {4:.3f}".format(iteration, c, cst, kktnorm,change))

        ## The Residual Vector of the KKT conditions is calculated
        residu, kktnorm, residumax = kktcheck(mm, n_var, x, ymma, zmma, lam, xsi, eta, mu, zet, S, xmin, xmax, df0dx, fval, dfdx_, a0, aa, C, d_)

        # Plot update
        if iteration % plot_rate == 0:

            ## Density plot            
            im1.set_array(-np.reshape(den, (nelx, nely)).T)

            ## Component plot        
            Xc = X[np.arange(0, len(X), 5)]
            Yc = X[np.arange(1, len(X), 5)]
            Tc = X[np.arange(2, len(X), 5)]
            Lc = X[np.arange(3, len(X), 5)]
            hc = X[np.arange(4, len(X), 5)]

            C0 = np.outer(np.ones(len(cc[0])), np.cos(Tc)).T
            S0 = np.outer(np.ones(len(cc[0])), np.sin(Tc)).T
            xxx = np.outer(np.ones(len(cc[0])), Xc).T + cc
            yyy = np.outer(np.ones(len(cc[0])), Yc).T + ss
            xi = C0 * (xxx - np.outer(np.ones(len(cc[0])), Xc).T) + S0 * (yyy - np.outer(np.ones(len(cc[0])), Yc).T)
            Eta = -S0 * (xxx - np.outer(np.ones(len(cc[0])), Xc).T) + C0 * (yyy - np.outer(np.ones(len(cc[0])), Yc).T)
            dd = np.real(norato_bar(xi, Eta, np.outer(np.ones(len(cc[0])), Lc).T, np.outer(np.ones(len(cc[0])), hc).T))
            xn = np.outer(np.ones(len(cc[0])), Xc).T + dd * cc
            yn = np.outer(np.ones(len(cc[0])), nely - Yc).T + dd * ss

            ax2.fill([0, nelx, nelx, 0], [0, 0, nely, nely], "w", edgecolor='black',
                     linewidth=3)
            ax2.fill(xn.T, yn.T, np.ones((xn.T).shape), "b", edgecolor='blue', linewidth=1)
            
            # Objective function plot
            ax3.plot(np.arange(1, iteration), cvec[np.arange(1, iteration)], 'b', marker='o')

            # Volume fraction plot
            ax4.plot(np.arange(1, iteration), cstvec[np.arange(1, iteration)], 'r', marker='o')
            
            fig.canvas.draw()

        X = Xmin + (Xmax - Xmin) * x

    return iteration


def norato_bar(xi, eta, L, h):
    
    d=((L/2* scimath.sqrt(xi**2/(xi**2+eta**2))+scimath.sqrt(h**2/4-L**2/4*eta**2/(xi**2+eta**2)))*(xi**2/(xi**2+eta**2)>=(L**2/(h**2+L**2)))+h/2*scimath.sqrt(1+xi**2/(eta**2+(eta==0))) * (xi**2/(xi**2+eta**2)<(L**2/(h**2+L**2))))*np.logical_or(xi!=0,eta!=0)+scimath.sqrt(2)/2*h*np.logical_and(xi==0,eta==0)
    return d

def lk():
    E=1
    nu=0.3
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
    return KE

def mmasub(m, n, iter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d):
    #    Version September 2007 (and a small change August 2008)
    #    Krister Svanberg <krille@math.kth.se>
    #    Department of Mathematics, SE-10044 Stockholm, Sweden.

    #    This function mmasub performs one MMA-iteration, aimed at
    #    solving the nonlinear programming problem:
    #      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    #    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
    #                xmin_j <= x_j <= xmax_j,    j = 1,...,n
    #                z >= 0,   y_i >= 0,         i = 1,...,m
    # *** INPUT:
    #   m (int)   = The number of general constraints.
    #   n (int)   = The number of variables x_j.
    #  iter (int) = Current iteration number ( =1 the first time mmasub is called).
    #  xval (n,)  = Column vector with the current values of the variables x_j.
    #  xmin (n,)  = Column vector with the lower bounds for the variables x_j.
    #  xmax (n,)  = Column vector with the upper bounds for the variables x_j.
    #  xold1 (n,) = xval, one iteration ago (provided that iter>1).
    #  xold2 (n,) = xval, two iterations ago (provided that iter>2).
    #  f0val(int) = The value of the objective function f_0 at xval.
    #  df0dx (n,) = Column vector with the derivatives of the objective function
    #          f_0 with respect to the variables x_j, calculated at xval.
    #  fval (int) = Column vector with the values of the constraint functions f_i,
    #          calculated at xval.
    #  dfdx (n,)  = (m x n)-matrix with the derivatives of the constraint functions
    #          f_i with respect to the variables x_j, calculated at xval.
    #          dfdx(i,j) = the derivative of f_i with respect to x_j.
    #  low  (n,)  = Column vector with the lower asymptotes from the previous
    #          iteration (provided that iter>1).
    #  upp  (n,)  = Column vector with the upper asymptotes from the previous
    #          iteration (provided that iter>1).
    #  a0  (int)  = The constants a_0 in the term a_0*z.
    #  a   (m,)   = Column vector with the constants a_i in the terms a_i*z.
    #  c   (m,)   = Column vector with the constants c_i in the terms c_i*y_i.
    #  d   (m,)   = Column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
    # *** OUTPUT:
    #  xmma  = Column vector with the optimal values of the variables x_j
    #          in the current MMA subproblem.
    #  ymma  = Column vector with the optimal values of the variables y_i
    #          in the current MMA subproblem.
    #  zmma  = Scalar with the optimal value of the variable z
    #          in the current MMA subproblem.
    #  lam   = Lagrange multipliers for the m general MMA constraints.
    #  xsi   = Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
    #  eta   = Lagrange multipliers for the n constraints x_j - beta_j <= 0.
    #   mu   = Lagrange multipliers for the m constraints -y_i <= 0.
    #  zet   = Lagrange multiplier for the single constraint -z <= 0.
    #   s    = Slack variables for the m general MMA constraints.
    #  low   = Column vector with the lower asymptotes, calculated and used
    #          in the current MMA subproblem.
    #  upp   = Column vector with the upper asymptotes, calculated and used
    #          in the current MMA subproblem.

    epsimin = 10 ** (- 7)
    raa0 = 1e-05
    albefa = 0.1
    asyinit = 0.01
    asyincr = 1.2
    asydecr = 0.4

    # Calculation of the asymptotes low and upp :
    if iter < 2.5:
        move = 1
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        move = 0.5
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = np.ones(n)
        factor[np.where(zzz > 0)] = asyincr
        factor[np.where(zzz < 0)] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - 0.1 * (xmax - xmin)
        lowmax = xval - 0.0001 * (xmax - xmin)
        uppmin = xval + 0.0001 * (xmax - xmin)
        uppmax = xval + 0.1 * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # Calculation of the bounds alfa and beta :
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)
    # Calculations of p0, q0, P, Q and b.
    xmami = xmax - xmin
    xmamieps = np.ones(n) * 1e-05
    xmami = np.maximum(xmami, xmamieps)
    xmamiinv = np.ones(n) / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    uxinv = np.ones(n) / ux1
    xlinv = np.ones(n) / xl1

    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(- df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2

    P = np.maximum(dfdx, 0)
    Q = np.maximum(- dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * np.ones(m) * xmamiinv.T
    P = P + PQ
    Q = Q + PQ
    P = P * scipy.sparse.spdiags(ux2, 0, n, n)
    Q = Q * scipy.sparse.spdiags(xl2, 0, n, n)
    b = np.dot(P[np.newaxis], uxinv[np.newaxis].T)[0] + np.dot(Q[np.newaxis], xlinv[np.newaxis].T)[0] - fval

    ## Solving the subproblem by a primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b,
                                                          c, d)

    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp

def subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d):
    # This function subsolv solves the MMA subproblem:
    # minimize   SUM[ p0j/(uppj-xj) + q0j/(xj-lowj) ] + a0*z +
    #          + SUM[ ci*yi + 0.5*di*(yi)^2 ],
    # subject to SUM[ pij/(uppj-xj) + qij/(xj-lowj) ] - ai*z - yi <= bi,
    #            alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.
    # Input:  m, n, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d.
    # Output: xmma,ymma,zmma, slack variables and Lagrange multiplers.

    epsi = 1
    x = 0.5 * (alfa + beta)
    y = np.ones(m)
    z = 1
    lam = np.ones(m)
    xsi = np.ones(n) / (x - alfa)
    xsi = np.maximum(xsi, np.ones(n))
    eta = np.ones(n) / (beta - x)
    eta = np.maximum(eta, np.ones(n))
    mu = np.maximum(np.ones(m), 0.5 * c)
    zet = 1
    s = np.ones(m)

    it1 = 0
    while epsi > epsimin:
        it1 = it1 + 1

        epsvecn = epsi * np.ones(n)
        epsvecm = epsi * np.ones(m)
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = np.ones(n) / ux1
        xlinv1 = np.ones(n) / xl1
        plam = p0 + P.T * lam
        qlam = q0 + Q.T * lam
        gvec = np.dot(P[np.newaxis], uxinv1[np.newaxis].T)[0] + np.dot(Q[np.newaxis], xlinv1[np.newaxis].T)[0]
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - a.T * lam
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = np.concatenate([rex, rey, rez]).T
        residu2 = np.concatenate([relam, rexsi, reeta, remu, [rezet], res])
        residu = np.concatenate([residu1, residu2])
        residunorm = np.sqrt(np.dot(residu[np.newaxis], residu[np.newaxis].T)[0][
                                 0])  # PROBABLY WRONH; THE VALUE IS TO HIGH WHEN COMPARED TO MATLAB
        residumax = max(abs(residu))

        it2 = 0
        while np.logical_and(residumax > 0.9 * epsi, it2 < 500):
            it2 = it2 + 1

            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = np.ones(n) / ux1
            xlinv1 = np.ones(n) / xl1
            uxinv2 = np.ones(n) / ux2
            xlinv2 = np.ones(n) / xl2
            plam = p0 + P.T * lam
            qlam = q0 + Q.T * lam
            gvec = np.dot(P[np.newaxis], uxinv1[np.newaxis].T)[0] + np.dot(Q[np.newaxis], xlinv1[np.newaxis].T)[0]
            GG = P * scipy.sparse.spdiags(uxinv2, 0, n, n) - Q * scipy.sparse.spdiags(xlinv2, 0, n, n)
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - a.T * lam - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = np.ones(n) / diagx
            diagy = d + mu / y
            diagyinv = np.ones(m) / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv
            if m > n:
                blam = dellam + dely / diagy - np.dot(GG[np.newaxis], (delx / diagx)[np.newaxis].T)[0]
                bb = np.concatenate([blam.T, delz]).T
                Alam = diaglamyi[0] + \
                       np.dot((GG * scipy.sparse.spdiags(diagxinv, 0, n, n))[np.newaxis], GG[np.newaxis].T)[0][0]
                AA = [[Alam, a[0]], [a[0], - zet / z]]
                solut = np.linalg.solve(AA, bb)
                dlam = solut[np.arange(0, m)]
                dz = solut[m]
                dx = - delx / diagx - (GG.T * dlam) / diagx
            else:
                diaglamyiinv = np.ones(m) / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = scipy.sparse.spdiags(diagx, 0, n, n) + GG[np.newaxis].T * scipy.sparse.spdiags(diaglamyiinv, 0, m,
                                                                                                     m) * GG
                azz = zet / z + a.T * (a / diaglamyi)
                axz = - GG.T * (a / diaglamyi)
                bx = delx + GG.T * (dellamyi / diaglamyi)
                bz = delz - a.T * (dellamyi / diaglamyi)
                AA1 = np.c_[Axx, axz]
                AA2 = np.concatenate([axz.T, azz])
                AA = np.r_[AA1, AA2[np.newaxis]]
                bb = np.concatenate([- bx.T, - bz])
                solut = np.linalg.solve(AA, bb)
                dx = solut[np.arange(0, n)]
                dz = solut[n]
                dlam = (np.dot(GG[np.newaxis], dx[np.newaxis].T)[0][0]) / diaglamyi - dz * (
                            a / diaglamyi) + dellamyi / diaglamyi
            dy = - dely / diagy + dlam / diagy
            dxsi = - xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = - eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = - mu + epsvecm / y - (mu * dy) / y
            dzet = - zet + epsi / z - zet * dz / z
            ds = - s + epsvecm / lam - (s * dlam) / lam
            xx = np.concatenate((y.T, [z], lam, xsi.T, eta.T, mu.T, [zet], s.T), axis=0)
            dxx = np.concatenate((dy.T, [dz], dlam.T, dxsi.T, deta.T, dmu.T, [dzet], ds.T), axis=0)
            stepxx = - 1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = - 1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1)
            steg = 1 / stminv
            xold = x
            yold = y
            zold = z
            lamold = lam
            xsiold = xsi
            etaold = eta
            muold = mu
            zetold = zet
            sold = s
            resinew = 2 * residunorm

            it3 = 0
            while np.logical_and(resinew > residunorm, it3 < 50).any():
                it3 = it3 + 1

                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = np.ones(n) / ux1
                xlinv1 = np.ones(n) / xl1
                plam = p0 + P.T * lam
                qlam = q0 + Q.T * lam
                gvec = np.dot(P[np.newaxis], uxinv1[np.newaxis].T)[0] + np.dot(Q[np.newaxis], xlinv1[np.newaxis].T)[0]
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - a.T * lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = np.concatenate([rex.T, rey.T, rez]).T
                residu2 = np.concatenate([relam.T, rexsi.T, reeta.T, remu.T, [rezet], res.T]).T
                residu = np.concatenate([residu1.T, residu2.T]).T
                resinew = np.sqrt(np.dot(residu[np.newaxis], residu[np.newaxis].T)[0][0])
                steg = steg / 2
            residunorm = resinew
            residumax = max(abs(residu))
            # steg = 2 * steg
        epsi = 0.1 * epsi
    xmma = x
    ymma = y
    zmma = z
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma

def kktcheck(m, n, x, y, z, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d):
    #  The left hand sides of the KKT conditions for the following
    #  nonlinear programming problem are calculated.
    #      Minimize  f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    #    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
    #                xmax_j <= x_j <= xmin_j,    j = 1,...,n
    #                z >= 0,   y_i >= 0,         i = 1,...,m
    # *** INPUT:
    #   m    = The number of general constraints.
    #   n    = The number of variables x_j.
    #   x    = Current values of the n variables x_j.
    #   y    = Current values of the m variables y_i.
    #   z    = Current value of the single variable z.
    #  lam   = Lagrange multipliers for the m general constraints.
    #  xsi   = Lagrange multipliers for the n constraints xmin_j - x_j <= 0.
    #  eta   = Lagrange multipliers for the n constraints x_j - xmax_j <= 0.
    #   mu   = Lagrange multipliers for the m constraints -y_i <= 0.
    #  zet   = Lagrange multiplier for the single constraint -z <= 0.
    #   s    = Slack variables for the m general constraints.
    #  xmin  = Lower bounds for the variables x_j.
    #  xmax  = Upper bounds for the variables x_j.
    #  df0dx = Vector with the derivatives of the objective function f_0
    #          with respect to the variables x_j, calculated at x.
    #  fval  = Vector with the values of the constraint functions f_i,
    #          calculated at x.
    #  dfdx  = (m x n)-matrix with the derivatives of the constraint functions
    #          f_i with respect to the variables x_j, calculated at x.
    #          dfdx(i,j) = the derivative of f_i with respect to x_j.
    #   a0   = The constants a_0 in the term a_0*z.
    #   a    = Vector with the constants a_i in the terms a_i*z.
    #   c    = Vector with the constants c_i in the terms c_i*y_i.
    #   d    = Vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
    # *** OUTPUT:
    # residu     = the residual vector for the KKT conditions.
    # residunorm = sqrt(residu'*residu).
    # residumax  = max(abs(residu)).

    rex = df0dx + dfdx.T * lam - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - a.T * lam
    relam = fval - a * z - y + s
    rexsi = xsi * (x - xmin)
    reeta = eta * (xmax - x)
    remu = mu * y
    rezet = zet * z
    res = lam * s
    residu1 = np.concatenate([rex.T, rey.T, rez]).T
    residu2 = np.concatenate([relam.T, rexsi.T, reeta.T, remu.T, [rezet], res.T]).T
    residu = np.concatenate([residu1.T, residu2.T]).T
    residunorm = np.sqrt(np.dot(residu[np.newaxis], residu[np.newaxis].T))[0][0]
    residumax = max(abs(residu))

    return residu, residunorm, residumax


# The real main driver
if __name__ == "__main__":
    # Input parameters
    nelx=160
    nely=40
    volfrac=0.4
    maxiteration=300
    if len(sys.argv)>1: nelx = int(sys.argv[1])
    if len(sys.argv)>2: nely = int(sys.argv[2])
    if len(sys.argv)>3: volfrac = float(sys.argv[3])
    if len(sys.argv)>4: maxiteration = int(sys.argv[4])
    main(nelx,nely,volfrac,maxiteration)
