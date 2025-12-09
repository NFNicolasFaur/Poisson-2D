#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:14:46 2023

@author: cardon
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import modulocascada as cas

from matplotlib import colors


#from solvers2D  import gauss_seidel, sor, redblack,gaussseidel
#from coeficientes2D import cofD_2D
#from redes2D  import red_nodal_2D_u
#from imprimeydibuja2D import imprime_matriz_2D
import math

def imprime_matriz_2D(Nx, Ny, a, nombre):
    # Printing a 2D matrix

    print(f"--- {Nx} {Ny} {nombre}-----")
    for i in range(0, Ny ):
        for j in range(Nx ):
            print(f"{a[i][j]:.2f} ", end="")
        print()


def kf(k1, k2):
    if k1== 0 or k2 == 0: #evita la division por cero 
        return 0

    else:
        return (2. * k1 * k2) / (k1 + k2)

def cofD_2D(Nx, Ny, dx, dy, ap0, x, y, kn, u, Sc, Sp):
    '''Matriz de coeficientes para la ecuacion de difusion
       calcula para los nodos internos.
       Deben ser modificados para incluir las condiciones de borde'''

    # Coefficients for the control volumes
    ae  = np.zeros((Ny , Nx) ,dtype=float)
    ao  = np.zeros((Ny , Nx ),dtype=float)
    an  = np.zeros((Ny , Nx) ,dtype=float)
    as_ = np.zeros((Ny , Nx ),dtype=float)
    ap  = np.zeros((Ny , Nx ),dtype=float)
    b   = np.zeros((Ny , Nx ),dtype=float)


    for i in range(1, Ny -1):
        for j in range(1, Nx - 1):
            ae[i][j] = (kf(kn[i][j], kn[i][j + 1]) * dy) / (x[j + 1] - x[j])
            ao[i][j] = (kf(kn[i][j], kn[i][j - 1]) * dy) / (x[j] - x[j - 1])
            an[i][j] = (kf(kn[i][j], kn[i + 1][j]) * dx) / (y[i + 1] - y[i])
            as_[i][j] = (kf(kn[i][j], kn[i - 1][j]) * dx) / (y[i] - y[i - 1])

            apaux = ao[i][j] + ae[i][j] + an[i][j] + as_[i][j]
            ap[i][j] = apaux + ap0 - Sp[i][j] * dx * dy
            b[i][j] = ap0 * u[i][j] + Sc[i][j] * dx * dy

    return ae, ao, an, as_, ap, b


def gaussseidel(Nx, Ny, ae, ao, an, a_s, ap, b, u,
                tol=1e-8, maxiter=None,
                criterion='residual',   # 'residual' or 'delta'
                check_every=1,          # cada cuantas iteraciones computar residuo
                verbose=False):
    """
    Gauss-Seidel 2D con criterio de parada configurable.

    Parametros:
      Nx, Ny: numero de celdas internas (uso índices 1..Nx, 1..Ny)
      ae, ao, an, a_s, ap, b, u: arrays numpy de forma (Ny+2, Nx+2)
         (se asume que los "bordes" 0 y N+1 están disponibles como ghost/bc)
      tol: tolerancia relativa (por defecto 1e-8)
      maxiter: maximo de iteraciones (por defecto Nx*Ny)
      criterio: 'residual' (por defecto) o 'delta'
      check_every: cada cuantas iteraciones calcular el residual (reduce coste)
      verbose: imprime informacion de progreso

    Retorna:
      u, info donde info es un dict con 'iter', 'final_norm', 'bnorma'
    """

    if maxiter is None:
        maxiter = Nx * Ny

    # Norma de b (infinity norm) en le interior del dominio
    b_interior = b[1:Ny+1, 1:Nx+1]
    bnorm = np.max(np.abs(b_interior))
    if bnorm == 0:
        bnorm = 1.0

    iter_count = 0
    # si usamos 'delta' guardamos una copia previa
    if criterion == 'delta':
        u_old = u.copy()

    final_norm = None

    while True:
        iter_count += 1

        # Barrido de Gauss-Seidel
        for i in range(1, Ny+1):
            for j in range(1, Nx+1):
                u[i, j] = (
                    ao[i, j] * u[i, j-1] +
                    ae[i, j] * u[i, j+1] +
                    an[i, j] * u[i+1, j] +
                    a_s[i, j] * u[i-1, j] +
                    b[i, j]
                ) / ap[i, j]
        # Chequeos y criterios
        if iter_count==1:
            imprime_matriz_2D(Nx+2, Ny+2, u, "primera iteracion u")

        if criterion == 'delta':
            # ||u - u_old||_inf
            diff = np.max(np.abs(u[1:Ny+1, 1:Nx+1] - u_old[1:Ny+1, 1:Nx+1]))
            final_norm = diff
            if verbose:
                print(f"iter {iter_count}, delta_inf = {diff:.3e}")
            if diff < tol or iter_count >= maxiter:
                break
            # actualizar u_old
            u_old[:] = u

        else:  # criterio 'residual'
            # calculamos residual cada 'check_every' iteraciones
            if iter_count % check_every == 0 or iter_count == 1:
                # construimos residuo r = b - (A u)
                # Para cada interior (i,j): A u = ap*u - ao*u_left - ae*u_right - an*u_up - a_s*u_down
                Au = (ap[1:Ny+1, 1:Nx+1] * u[1:Ny+1, 1:Nx+1]
                      - ao[1:Ny+1, 1:Nx+1] * u[1:Ny+1, 0:Nx]
                      - ae[1:Ny+1, 1:Nx+1] * u[1:Ny+1, 2:Nx+2]
                      - an[1:Ny+1, 1:Nx+1] * u[2:Ny+2, 1:Nx+1]
                      - a_s[1:Ny+1, 1:Nx+1] * u[0:Ny, 1:Nx+1])
                r = b_interior - Au
                rnorm = np.max(np.abs(r))
                final_norm = rnorm
                rel = rnorm / bnorm
                if verbose:
                    print(f"iter {iter_count}, ||r||_inf = {rnorm:.3e}, rel = {rel:.3e}")
                if rel < tol or iter_count >= maxiter:
                    break

        # seguridad: evitar bucle infinito
        if iter_count >= maxiter:
            break

    info = {'iter': iter_count, 'final_norm': float(final_norm), 'bnorma': float(bnorm)}
    if verbose:
        print(f"Gauss-Seidel final: iter={iter_count}, final_norm={final_norm}, bnorm={bnorm}")
    return u, info

def red_nodal_2D_u(Nx, Ny, Lx, Ly):
    # Nx, Ny numero total de nodos
    # Volumenes de control, practica B
    # Nodos centrados entre caras
    # Red uniforme
    # Ojo no todos los dx,dy son iguales!!!!
    x = np.zeros(Nx  )
    y = np.zeros(Ny  )
    dx=Lx/(Nx-2)
    dy=Ly/(Ny-2)
    print(dx,dy)
    x = [0] * (Nx )
    y = [0] * (Ny )

    x[0] = 0
    x[1] = dx / 2
    for i in range(2, Nx-1 ):
        x[i] = x[i - 1] + dx
    x[Nx - 1] = x[Nx-2] + (dx / 2)
    print(x)

    y[0] = 0
    y[1] = dy / 2
    for i in range(2, Ny-1 ):
        y[i] = y[i - 1] + dy
    y[Ny -1 ] = y[Ny-2] + (dy / 2)
    print(y)

    return x,dx, y,dy



def dirichletCB(Nx, Ny, ae, ao, an, as_, ap, b, u):

    # Dirichlet a la izquierda
    j = 1
    for i in range(1, Ny -1):
        b[i][j] = b[i][j] + ao[i][j] * u[i][j - 1]
        #ao[i][j] = 0

    # Dirichlet a la derecha
    j = Nx-2
    for i in range(1, Ny-1):
        b[i][j] = b[i][j] + ae[i][j] * u[i][j + 1]
        #ae[i][j] = 0

    # Dirichlet abajo
    i = 1
    for j in range(1, Nx -1):
        b[i][j] = b[i][j] + as_[i][j] * u[i - 1][j]
        #as_[i][j] = 0

    # Dirichlet arriba
    i = Ny-2
    for j in range(1, Nx -1):
        b[i][j] = b[i][j] + an[i][j] * u[i + 1][j]
        #an[i][j] = 0

    return ae, ao, an, as_, ap, b
 

def bordesDirichlet(u):
    # ===============================================================
    # Condicion de borde de Dirichlet oeste y este
    TB=0
    TL=100
################################################################################################
###================= Regiones adiabáticas ================#############3
    #aqui se copia la temperatura del nodo adyacente hacia la derecha
    #la simulacion es que no hay flujo de calor a traves del borde
    #condiciones para dar la parte adiabatica en la parte de la izqierda superior 
    for i in range(Ny//2 + 1, Ny ):
        for j in range(0, Nx//4):
            u[i][j]= u[i][j+1] 
    #analogo a lo anterior
    #condiciones para dar la parte adiabatica en la parte de la derecha superior de la figura
    for i in range(Ny//2, Ny//2 + 1 ):
        for j in range(Nx//2, 3*Nx//4):
            u[i][j]= u[i-1][j] 
    for j in range(3*Nx//4, Nx-1):
        u[0][j]= 100
    return u

def propiedades(Ny , Nx, rhoo,Cpp,kk ):
    # Matriz de conductividades nodales
    k   = np.zeros(  (Ny , Nx) ,dtype=float)
    rho = np.zeros(  (Ny , Nx) ,dtype=float)
    Cp  = np.zeros(  (Ny , Nx) ,dtype=float)
    # Asigancion de la conductividad uniforme
    for i in range(0, Ny ):
        for j in range(0, Nx ):
            k[i][j] = kk
    #Modificacion de las conductividades en esos "huecos"
    for i in range(Ny//2 , Ny ):
        for j in range(0, Nx//4):
            k[i][j]= 0 ##para simular la figura 
    for i in range(0 , Ny//2 ):
        for j in range(Nx//2, 3*Nx//4):
            k[i][j]= 0 
    for i in range(0, Ny ):
        for j in range(0, Nx ):
            rho[i][j] = rhoo

    for i in range(0, Ny ):
        for j in range(0, Nx ):
            Cp[i][j] = Cpp

    return rho,Cp,k

def fuente(Nx,Ny,Scc,Spp):
    # Matrices del termino de generacion
    # Aproximacion lineal S(T)=Scc + Spp * T
    Sc = np.zeros((Ny , Nx) ,dtype=float)  # Coordenada al origen
    Sp = np.zeros((Ny , Nx ),dtype=float)  # Pendiente
    # Asignacion de fuente uniforme
    radio1= min(Nx, Ny)//8
    radio2= min(Nx, Ny)//8
    for i in range(1, Ny ):
        for j in range(1, Nx ):
            ###if radio1 > (np.sqrt((j-Nx//2)**2 + (i-Ny//2)**2)):
                ###Sc[i][j] = 0
                ###Sp[i][j] = 0

            if radio2 >= (np.sqrt((j-Nx//8)**2 + (i-Ny//4)**2)): #la circunferencia en (Nx//8, Ny//4):
                Sc[i][j] = Scc
                Sp[i][j] = Spp
            else:
                Sc[i][j] = 0
                Sp[i][j] = 0
    return Sc,Sp

def inicialT(Nx, Ny,Tinicial, Tinicialfigura):
    # Condicion inicial uniforme
    u = np.full((Ny , Nx) ,Tinicial,dtype=float)
    ##aquí pongo que la temperatura inicial en esos rectángulos dados por la figura, es un valor del ambiente donde se encuenta
    #####La simulación que hice depende de Tinicialfigura, ya que eso podría ser una temperatura del ambiente donde se encuentre la placa

    for i in range(Ny//2 , Ny ):
        for j in range(0, Nx//4):
            u[i][j]= Tinicialfigura ##para simular la figura 
    for i in range(0 , Ny//2 ):
        for j in range(Nx//2, 3*Nx//4):
            u[i][j]= Tinicialfigura

    return u



def solucion(Nx, Ny, Lx,Ly, dx,dy,dt, kn,  Scc, Spp,rhoo,Cpp, Tinicial, ntemporal):
    # Evolution temporal
    omega=1.3
    ap0 = (Cpp * rhoo * dx * dy) / dt
    x,dx, y,dy = red_nodal_2D_u(Nx, Ny, Lx, Ly)

    print("x :", x)
    print("y :", y)
    print("\n")
    Sc,Sp=fuente(Nx,Ny,Scc,Spp)

    imprime_matriz_2D(Nx, Ny, Sc,"Sc")
    imprime_matriz_2D(Nx, Ny, Sp,"Sp")

    rho,Cp,k=propiedades(Nx,Ny,rhoo,Cpp,kk)
    #imprime_matriz_2D(Nx, Ny, rho,"rho")
    #imprime_matriz_2D(Nx, Ny, Cp,"Cp")
    imprime_matriz_2D(Nx, Ny, k,"k")


    u = inicialT(Nx, Ny,Tinicial, Tinicialfigura)
    imprime_matriz_2D(Nx, Ny, u, "---- condicion inicial u")
    u = bordesDirichlet(u)
    imprime_matriz_2D(Nx, Ny, u, "---- bordes Dirichlet u")

    ae, ao, an, as_, ap, b = cofD_2D(Nx, Ny, dx, dy, ap0, x, y, k, u, Sc, Sp)
    imprime_matriz_2D(Nx, Ny, ae, "ae")
    imprime_matriz_2D(Nx, Ny, ao,"ao")
    imprime_matriz_2D(Nx, Ny, an,"an")
    imprime_matriz_2D(Nx, Ny, as_,"as_")
    imprime_matriz_2D(Nx, Ny, ap,"ap")


    for ki in range(1, ntemporal + 1):
        # No esta implementado el caso temporal!!!
        print("# Temporal step %d\n" % ki)
        u,info=gaussseidel(Nx-2, Ny-2, ae, ao, an, as_, ap, b,
                           u,tol=1e-8,
                           maxiter=None,criterion='residual',   # 'residual' or 'delta'
                           check_every=1,          # cada cuantas iteraciones computar residuo
                           verbose=True)

        #u=gauss_seidel(Nx, Ny, ae, ao, an, as_, ap, b, u)
        #u=sor(Nx, Ny, omega, ae, ao, an, as_, ap, b, u)
        #u=redblack(Nx, Ny,  ae, ao, an, as_, ap, b, u)
    imprime_matriz_2D(Nx, Ny, u,"solucion final")

    return x,y, u

    """
    return x,y,u

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, u, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Solution')
    plt.show()
    """
if __name__ == "__main__":

    """
    Conduccion 2D en una placa rectangular. La fuente corresponde a
    una perdida convectiva.
    """
    # Volumenes de control en diferencias finitas 2D

    Nx = 49  # Numero total de nodos
    Ny = 49
    Lx = 2 ##es un ractángulo, recordar
    Ly = 1            # Longitud del dominio de caculo
    dx = Lx / (Nx-1)
    dy = Ly / (Ny-1)  # tamanio del volumen de control
    dt = 1e10    # Paso de tiempo, grande -> estacionario
    rhoo = 1.0
    Cpp = 1.0
    kk = 1          # Conductividad termica
        # Fuente
    Spp = 50
    Scc = 300
    Tinicial=15 #temperatura de la figura (placa)
    Tinicialfigura= 5 #temperatura en el exterior de la figura, recordar que es adiabático en esa parte
    ntemporal=1 ###No cambiar
    x,y,u = solucion(Nx, Ny, Lx,Ly, dx,dy,dt, kk,  Scc, Spp,rhoo,Cpp, Tinicial, ntemporal)
    imprime_matriz_2D(Nx, Ny, u ,"solucion u")


    cmap = plt.cm.get_cmap("hot")

    image = u
    #plt.imshow(image ,cmap="hot")
    #plt.imshow(image,vmin=0,vmax=100 ,cmap="hot")
    plt.imshow(image,interpolation='bilinear',vmin=0,vmax=100 ,cmap="hot")
    plt.colorbar()
    plt.title("Distribución de temperatura")
    #plt.savefig("temperatura2D.png", dpi=300)

    X, Y = np.meshgrid(x, y)  #y en fila y x en columnas

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')

    ###grafica de la superficie
    surf = ax.plot_surface(X, Y, u, cmap='hot', edgecolor='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Temperatura')
    ax.set_title('Distribución de temperatura en estado estacionario')

    
    fig.colorbar(surf, shrink=0.5, aspect=10) #barra de colores (esto lo saqué de https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html)

    #fig.savefig("temperatura3D.png", dpi=300)


    plt.show()

