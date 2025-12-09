import numpy as np
import sys
from numpy import linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D


large_width = 500
np.set_printoptions(linewidth=large_width)
float_formatter = "{4.1f}".format
np.set_printoptions(formatter={'float_kind':float_formatter},precision=2,suppress=True)
# suppess=true suprime el fromato cientifico
# Para suprimir la impresion resumida
np.set_printoptions(threshold=sys.maxsize)

def matrizCIyB_3(Lx,Ly,columnas,filas,Tx0, TxL, inicializacion):
    """
    crea la discretización de un dominio de cálculo
    rectangular de  Lx x Ly y n x m nodos
    contando los nodos de borde.
    Todos los bordes cero
    sin(x)*cos(y)
    n filas
    m=columnas
    """
    f=np.zeros((filas,columnas))
    a=np.zeros((filas,columnas))
    b=np.zeros((filas,columnas))

#    xs, ys = np.meshgrid(filas, columnas)
    xs=np.linspace(0,Ly,filas)
    ys=np.linspace(0,Lx,columnas)
    a[:,]=np.sin(xs)
    b[:,:]=np.sin(ys).reshape(-1,1)
    f=np.multiply(a,b)
    print("matrizCIyB_3, T", f)
    return f

def matrizCIyB_1(Lx,Ly,columnas,filas,Tx0, TxL, inicializacion):
    """
    crea la discretización de un dominio de cálculo
    rectangular de  Lx x Ly y n x m nodos
    contando los nodos de borde.
    Inicializa según distintos métodos
    Esteblece las siguientes condiciones de borde
    izquierdo: a
    derecho  : b
    bordes   ; T bordes superior e inferior lineal entre Tx0 TxL
    n y m, múltiplos de 2 > 4
    n filas
    m=columnas
    """
    f=np.zeros((filas,columnas))
    dx=Lx/(columnas-1)
    dy=Ly/(filas-1)
#    print(dx,dy)
    # Borde izquierdo
    a=np.linspace(Tx0,TxL,filas)
    f[0:filas,0]=Tx0
    f[0:filas,-1]=TxL

#    print("a",f[0:filas,0],"\nb",f[0:filas,-1])
    a=np.linspace(Tx0,TxL,columnas)
#    print("a", a.shape, "f", f.shape)

#    print(a)
    f[0,:]=a
    f[-1,:]=a
    return f

def matrizCIyB_2(Lx,Ly,columnas,filas,Tx0, TxL, inicializacion):
    """
    crea la discretización de un dominio de cálculo
    rectangular de  Lx x Ly y n x m nodos
    contando los nodos de borde.
    Inicializa según distintos métodos
    Esteblece las siguientes condiciones de borde
    izquierdo: a
    derecho  : b
    bordes   ; T bordes superior e inferior lineal entre Tx0 TxL
    n y m, múltiplos de 2 > 4
    n filas
    m=columnas
    """
    f=np.zeros((filas,columnas))
    dx=Lx/(columnas-1)
    dy=Ly/(filas-1)
#    print(dx,dy)
    # Borde izquierdo
    a=np.linspace(Tx0,TxL,filas)
    f[0:filas,0]=a
    f[0:filas,-1]=-a

#    print("a",f[0:filas,0],"\nb",f[0:filas,-1])
    a=np.linspace(Tx0,TxL,columnas)
#    print("a", a.shape, "f", f.shape)

#    print(a)
    f[0,:]=a
    f[-1,:]=-a
    return f

def JacobiStepP(filas,columnas, dx,dy,T):
    n=filas
    m=columnas
    # requiere definir las matrices
    TT=np.zeros((n,m))
    Te=np.zeros((filas,columnas))
    To=np.zeros((filas,columnas))
    Tu=np.zeros((filas,columnas))
    Td=np.zeros((filas,columnas))

    Te[1:filas-1 , 1:columnas-1]=T[1:filas-1 , 2:columnas]
#    print("matriz corrida a la derecha\n",Te, "\n")
    To[1:filas-1 , 1:columnas-1]=T[1:filas-1 , 0:columnas-2]
#    print("matriz corrida a la izquierda\n",To, "\n")

    Tu[1:filas-1 , 1:columnas-1]=T[2:filas , 1:columnas-1]
#    print("matriz corrida a la arriba\n",Tu, "\n")

    Td[1:filas-1 , 1:columnas-1]=T[0:filas-2 , 1:columnas-1]
#    print("matriz corrida a la abajo\n",Td, "\n")

#    print("Suma\n")
    TT= T + (Te+To+Tu+Td)/4
#    print(f)

    return TT

def JacobiStepPP(T):
    """ Hace un paso de Jacobi
        Trabaja solo con los nodos internos
    n=filas
    m=columnas
    """
    n,m=T.shape
    TT=np.copy(T)
    TT[1:n-1 , 1:m-1 ]=  ( T[1:n-1 , 2:m] + T[1:n-1 , 0:m-2]+ T[2:n , 1:m-1]+T[0:n-2 , 1:m-1]) /4
    return TT

def dibujar(T, titulo):
    filas,columnas=T.shape
    xs, ys = np.meshgrid(filas, columnas)
    plot=plt.imshow(T,vmin=0,vmax=1)#, cmap=plt.cm.gray)

# Draw a color bar
#    plt.colorbar()
    plt.legend()
    plt.show()
   
# Show the plot
    return plot
#print(a)

def prolongacion(T):
    """
    Prolongación de la red suelta a la red fina
    para una red centrada en los vértices
    Briggs 2000, pp 35
    Procedimiento:
    1) Los nodos de la red gruesa se prolonga directamente
     a la red fina. Requiere las dos matrices
    Los demas nodos, incluidos los de borde (¿¿¿???), se interpolan
    a partir de los valores ya prolongados direcamente en la red fina.
    2) Los valores prolongados, ya sobre la red fina, se interpolan (promedian) horizontalmente, de a dos,
    para llenar las posiciones intermedias.     Esto hasta los nodos de borde (¿¿¿???).
    3) Los valores prolongados, ya sobre la red fina se interpolan (promedian) verticalmente, de a dos,
    para llenar las posiciones intermedias.     Esto hasta los nodos de borde (¿¿¿???).
    4) Los valores prolongados sobre la red fina se interpolan (promedian) de a cuatro para
    llenar los nodos centrados de la red fina

    Sobre los bordes, puedo usar la interpolación o puede reconstrior las condiciones de borde sobre
    la red fina.
    n=filas
    m=columnas"""
    n,m=np.shape(T)
#    print(n,m)
    N=2*n-1
    M=2*m-1
    Tfina=np.zeros((N,M))
#    print(Tfina, Tfina.shape)
    # Prolongación directa: dos redes
    Tfina[0:N:2, 0:M:2]=T[0:n , 0:m]
#    print("prolongacion directa \n",Tfina,"\n")
    # Interpolacion horizotal, sobre a red fina
#    print( Tfina[0:N:2, 0:M-1:2] , Tfina[0:N:2, 0:M-1:2].shape )
#    print( Tfina[0:N:2, 2:M:2] , Tfina[0:N:2, 2:M:2].shape )
#    print(Tfina[0:N:2, 1:M-1:2], Tfina[0:N:2, 1:M-1:2].shape)
    Tfina[0:N:2, 1:M-1:2]=  (Tfina[0:N:2, 0:M-1:2] + Tfina[0:N:2, 2:M:2])/2
#    print("Izquierda a\n",Tfina)
    Tfina[1:N-1:2, 0:M:2]=  (Tfina[0:N-1:2, 0:M:2] + Tfina[2:N:2, 0:M:2])/2

#    print("Abajo a\n",Tfina)
#    print("-----------------\n")
#    print("centros\n",Tfina[1:N-1:2, 1:M-1:2],Tfina[1:N-1:2, 1:M-1:2].shape) # a rellenar

#    print("esquina\n",Tfina[0:N-1:2, 0:M-2:2],Tfina[0:N-1:2, 0:M-2:2].shape,"\n")
#    print("derecha\n",Tfina[0:N-1:2, 2:M:2],Tfina[0:N-1:2, 2:M:2].shape,"\n")

#    print("abajoesquina\n", Tfina[2:N:2, 0:M-1:2],Tfina[2:N:2, 0:M-1:2].shape,"\n")
#    print("derecha abajo\n", Tfina[2:N:2, 2:M:2],Tfina[2:N:2, 2:M:2].shape,"\n")
    """
    no es necesario, cada modificacion de la matriz involucra distintos elementos
    a= np.copy(Tfina[0:N-1:2, 0:M-2:2])
    b= np.copy(Tfina[0:N-1:2, 2:M:2])
    c= np.copy(Tfina[2:N:2, 0:M-1:2])
    d= np.copy(Tfina[2:N:2, 2:M:2])
    e=np.copy((a+b+c+d)/4)
    print("e\n",e)
    Tfina[1:N-1:2, 1:M-1:2]=e"""
    Tfina[1:N-1:2, 1:M-1:2]=( Tfina[0:N-1:2, 0:M-2:2] + Tfina[0:N-1:2, 2:M:2] + Tfina[2:N:2, 0:M-1:2]+ Tfina[2:N:2, 2:M:2])/4
#    print("Final a\n",Tfina)
    return Tfina

def solverJacobi(T, epsilon):
    dif =100
    k=0
    while dif > epsilon:
        norma1=la.norm(T,'fro')
        TT=JacobiStepPP(T)
        T=np.copy(TT)
        norma2=la.norm(T,'fro')
        dif=np.abs(norma1-norma2)
        k=k+1
    print("k",k,  "dif",dif)

    return k, T
def solverJacobiN(T, N):
    
    k=0
    for i in range (1,N):
        norma1=la.norm(T,'fro')
        TT=JacobiStepPP(T)
        T=np.copy(TT)
        norma2=la.norm(T,'fro')
        dif=np.abs(norma1-norma2)
        k=k+1
    print("k",k,  "dif",dif)

    return k, T

def cascadic(niveles,T,epsilon):
    """Algoritmo de iteracion anidada
       No es necesario tanta presición para
       Jacopbi
    """
    n,m=T[0].shape
#    print(n,m, range(niveles-1))
    for i in range(1, niveles-1):
        m=m*2-1 # filas
        n=n*2-1
        T[i+1]=np.zeros((n,m))
        T[i+1]=prolongacion(T[i])
        k,T[i+1]=solverJacobi(T[i+1], epsilon)
        #dibujar(T[i+1], "T[i+1]")
    return T
def cascadicN(niveles,T,N):
    """Algoritmo de iteracion anidada
       No es necesario tanta presición para
       Jacopbi
    """
    n,m=T[0].shape
#    print(n,m, range(niveles-1))
    for i in range(1, niveles-1):
        m=m*2-1 # filas
        n=n*2-1
        T[i+1]=np.zeros((n,m))
        T[i+1]=prolongacion(T[i])
        k,T[i+1]=solverJacobiN(T[i+1],N)
        #dibujar(T[i+1], "T[i+1]")
    return T


if __name__ == "__main__":

    inicializacion=1
    epsilon=1.E-1
    Lx=4*np.pi
    Ly=4*np.pi
    columnas=16
    filas=16
    m=columnas
    n=filas
    Tx0=0
    TxL=0
    dx=Lx/columnas # No estoy usando dx, dy
    dy=Ly/filas
    niveles=6
    N=5            # Solo cinco iteraciones
    # crea una arreglo vacio de tipo objeto para alojar matrices de distinto tamaño
    T=np.empty((niveles),dtype=object)
    T[0]=matrizCIyB_3(Lx,Ly,m,n,Tx0, TxL, inicializacion)
    # Hace el calculo sobre la red primaria
    k,T[1]=solverJacobiN(T[0], N)
    # Utiliza el valor de T como CI para un calculo  en la red mas fina
    # descendiendo hasta 6 niveles de refinamiento.
    cascadicN(niveles,T,N)
    """
    
    dibujar(T[-1], "Tmasfina")
    print(T[-1])
    plt.show()"""
    cmap = plt.cm.get_cmap("hot")

    for i in range(niveles):
        image = T[i]
        # Assuming the image is a NumPy array
        plt.imshow(image, vmin=-1,vmax=1, cmap="hot") 
        plt.title(f'Image {i+1}: {T[i].shape}')
        plt.colorbar()
        plt.show()
    """
    plt.plot_surface(x, y, T[0],  
                       rstride = 20, 
                       cstride = 20, 
                       alpha = 0.9,
                       edgecolor = "k",
                       cmap = "hot")
    plt.show()
    """