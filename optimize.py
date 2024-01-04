import numpy as np 

def adam(x0,f,grad,niter=500,tol=1e-6,eta=0.9,beta1=0.9,beta2=0.99):

    epsilon = 1e-8
    x = x0
    iter = 1
    xs = [x0]
    ys = [f(x0)]
    s = np.zeros(len(x0))
    v = np.zeros(len(x0))

    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        g = grad(x)
        
        v = beta1*v + (1-beta1)*g 
        s = beta2*s + (1-beta2)*g**2
        vt = v/(1-beta1**iter)
        st = s/(1-beta2**iter)

        d = - eta/(np.sqrt(st+epsilon))*vt

        x = x + d
        
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} grad:{g}')
        iter+=1
        
    return x, xs, ys


def rmsprop(x0,f,grad,niter=500,tol=1e-6,eta=0.9,gamma=0.98):

    epsilon = 1e-8
    x = x0
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    s = np.zeros(len(x0))

    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        g = grad(x)
        s = gamma*s + (1-gamma)*g**2
        x = x - eta/np.sqrt(s+epsilon)*g
        
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} grad:{g}')
        iter+=1
        
    return x, xs, ys

def gradient_descent_momentum(x0,f,grad,niter=500,tol=1e-6,eta=0.9,beta=0.98):

    x = x0
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    v = np.zeros(len(x0))

    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        g = grad(x)
        v = beta*v + g
        x = x - eta * v
        
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} grad:{g}')
        iter+=1
        
    return x, xs, ys

def gradient_descent(x0,f,grad,step_size,niter=500,tol=1e-6,gamma=0.98):

    x = x0
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    ss = [step_size]
    s = step_size
    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = -grad(x)
        x = x + s*d
        ss.append(s)
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{d}')
        iter+=1
        s = step_size
        

    return x, xs, ys,ss

def gradient_descent_adaptive_step(x0,f,grad,step_size,niter=500,tol=1e-6,gamma=0.98, metod='default'):

    x = x0
    iter = 0
    xs = [x0]
    ys = [f(x0)]
    ss = [step_size]
    s = step_size
    while (iter < niter) and (np.linalg.norm(grad(x),2) > tol):
        d = -grad(x)
        
        if metod == 'golden_section':
            s = golden_section(lambda a: f(x + a * d), 0, step_size)
        elif metod == 'default':
            while f(x + s*d) > f(x):
                s = s*gamma

        x = x + s*d
        ss.append(s)
        xs.append(x)
        ys.append(f(x))
        print(f'[{iter}] f: {f(x):.4f} g: {np.linalg.norm(grad(x),2):.4f} d:{d}')
        iter+=1
        s = step_size
        

    return x, xs, ys,ss


def golden_section(f, a, b, tol=1e-6):
    O = (np.sqrt(5) + 1)/2   # Número de ouro

    while (b - a) > tol:
        d = (b - a) / O
        x1 = b - d
        x2 = a + d

        if f(x1) < f(x2):
            b = x2
        else:
            a = x1

    return (a + b) / 2
