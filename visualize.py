import numpy as np 
import matplotlib.pyplot as plt 
from copy import copy
from optimize import gradient_descent, gradient_descent_adaptive_step, gradient_descent_momentum, rmsprop, adam


def function_contour(fun, lb, up, delta, levels):
    xs = np.arange(lb[0], up[0], delta)
    ys = np.arange(lb[0], up[1], delta)

    Z = np.zeros((len(xs), len(ys)))

    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            Z[i][j] = fun([x,y])

    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots()
    contour = ax.contour(Y, X, Z, levels=levels)
    ax.set_aspect('equal', adjustable='box')

    return ax

def plot_sequence(fun, seq, ax, label='Não informado'):
    
    p0 = [p[0] for p in seq]
    p1 = [p[1] for p in seq]
    ax.plot(p1, p0, '-d', label=label)


def plot_Htransform(reference, H, ax=None, color='black', w1=0.04, s1=0.3, w2=0.04, s2=0.05):

    arrows = np.array([[reference[0],reference[1],np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))] for angle in range(0,360,10)])

    H_arrows = []

    for ar in arrows:
        a = copy(ar)
        tip = np.array([a[2],a[3]])
        
        t = np.matmul(H(reference),tip)
        H_arrows.append([a[0],a[1],t[0],t[1]])

    H_arrows = np.array(H_arrows)

    ax.quiver(arrows[:,0],arrows[:,1],arrows[:,2],arrows[:,3],units='x', width=w1,scale=s1,color='black')
    ax.quiver(H_arrows[:,0],H_arrows[:,1],H_arrows[:,2],H_arrows[:,3],units='x', width=w2,scale=s2,color=color)

    return ax    

def plot_invHtransform(reference,H,ax=None,color='black',w1=0.04,s1=0.3,w2=0.04,s2=0.05):

    arrows = np.array([[reference[0],reference[1],np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))] for angle in range(0,360,10)])


    invH_arrows = []

    invH_T = np.linalg.inv(H(reference))

    for ar in arrows:
        a = copy(ar) 
        tip = np.array([a[2],a[3]])

        t = np.matmul(invH_T,tip)
        invH_arrows.append([a[0],a[1],t[0],t[1]])

    invH_arrows = np.array(invH_arrows)

    ax.quiver(arrows[:,0],arrows[:,1],arrows[:,2],arrows[:,3],units='x', width=w1,scale=s1,color='black')
    ax.quiver(invH_arrows[:,0],invH_arrows[:,1],invH_arrows[:,2],invH_arrows[:,3],units='x', width=w2,scale=s2,color=color)

    return ax

def plot_grad(reference,grad,H,ax,w1=0.04,s1=0.3,w2=0.04,s2=0.05):

    M = np.linalg.inv(H(reference))
    g = grad(reference)
    t_g = np.matmul(M,g)

    ax.quiver(reference[0],reference[1],-g[0],-g[1],units='x', width=w1,scale=s1,color='blue')
    ax.quiver(reference[0],reference[1],-t_g[0],-t_g[1],units='x', width=w2,scale=s2,color='red')


if __name__ == '__main__':

    f = lambda x: 10*x[0]**2 + 2*x[1]**2
    g = lambda x: np.array([20*x[0], 4*x[1]])
    H = lambda x: [[20,0],
                   [0,4]]
    s1 = 0.3
    s2 = 0.05
    w1 = 0.04
    w2 = 0.04
    
    #f = lambda x: 2*x[0]**2 + x[1]**2 + 2*x[0]*x[1]
    #g = lambda x: np.array([4*x[0] + 2*x[1], 2*x[1] + 2*x[0]])
    #H = lambda x: [[4,2],
    #               [2,2]]
    #s1 = 0.3
    #s2 = 0.05
    #w1 = 0.04
    #w2 = 0.04
    
    #f = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    #g = lambda x: np.array([2*(200*x[0]**3-200*x[0]*x[1] + x[0]), 200*(x[1]-x[0]**2)])
    #H = lambda x: [[1200*x[0]**2-400*x[1] + 2, -400*x[0]],
    #               [-400*x[0],200]]
    #s1 = 5
    #s2 = 0.1
    #w1 = 0.005
    #w2 = 0.005

    #ref = [-2,4]
    #ax = function_contour(f,[-10,-10],[10,10],0.1,50)
    x0 = np.array([-3,-4])

    _, gd_gs, _, _ = gradient_descent_adaptive_step(x0,f,g,step_size=0.09,niter=1000,tol=1e-6, metod='golden_section')
    _, gd_ar, _, _ = gradient_descent_adaptive_step(x0,f,g,step_size=0.09,niter=1000,tol=1e-6, metod='armijo')

    
    ref = [0.4,-0.4]
    ax = function_contour(f,[-.5,-.5],[.5,.5],0.05,100)

    #plot_Htransform(ref,H,ax,color='blue',s1=s1,s2=s2,w1=w1,w2=w2)
    #plot_invHtransform(ref,H,ax,color='red',s1=s1,s2=s2,w1=w1,w2=w2)
    plot_grad(ref,g,H,ax,w1,s1*30,w2,s2*20)
    plot_sequence(f,gd_gs,ax, label='Gradiente descendente com seção aúrea')
    plt.show()