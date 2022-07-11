# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:42:35 2022

@author: VicBoxMS
"""

def KFCNN(X,y,parametro_gamma):  
  """
  Funcion KFCNN, tiene como objetivo reducir el numero de instancias considerando
  la distancia euclideana en un espacio de caracteristicas donde dichos datos
  sean linealmente separables, lo anterior se hace para posteriormente
  entrenar un clasificador, bajo la idea de tener una perdida minima en el 
  rendimiento.
  La idea es inicilizar el algoritmo KFCNN incluyendo todas las observaciones
  de la clase minoritaria, tal que la  selección o reducción se hace sobre la clase mayoritaria
  -Entrada- 
  X: Matriz de nxp tal que n es el numero de observaciones y p el numero de 
  caracteristicas
  y: Vector de etiquetas de longitud n correspondiente a cada una de las 
  observaciones de X.
  Parametro_gamma: corresponde al parametro que utiliza la función kernel rbf 
  su rango es (0,+ inf )
  -Salida- 
  Retorna un subconjunto consistente reducido S = [S(X),s(y)] en forma de 
  Vector de Caracteristicas - Etiquetas de Clase, lo que tambien se conoce 
  como labeled point, la etiqueta se retorna en la ultima columna.
  """
              #Por defecto utilizaremos el parametro k=1, pero puede ser modificado
  parametro_k=1
  k = parametro_k
              #Realizamos el calculo de las distancias en el espacio kernel
              #Nota: Si no se desea escalar para este momento, simplemente 
              ##K = rbf_kernel(X,gamma=parametro_gamma)
  K = rbf_kernel(X,gamma=parametro_gamma)
  t=np.diag(K).reshape(-1,1)@(np.repeat(1,len(K)).reshape(1,-1))
  dXij = (t+t.T-2*K)
              #dXij, es la matriz utilizada para hacer las comparaciones mas adelante
  
  datosTrain = X
  clasesTrain = y

  nClases = 0 #Contar el numero de clases diferentes que tenemos 
  for i in range(len(clasesTrain)):
    if clasesTrain[i]>nClases:
      nClases = clasesTrain[i];
  nClases+=1

              #Inicializar el vector nearest como -1's
  nearest = np.random.randint(1, size=(len(datosTrain),k))-1


              #Inicializamos al conjunto S como un conjunto vacio
              # y tamS es un contador del nunero de elementos en S
  MAX_VALUE= 1000000000 
  S = np.random.randint(1, size=(len(datosTrain)))+MAX_VALUE
  tamS = 0

              #Inicializamos a dS como las observaciones mas cercanas 
              #a los centroides
  deltaS = []
  for i in range(int(nClases)):
    nCentroid = 0;
    centroid = np.zeros(len(datosTrain[0]))
    for j in range(len(datosTrain)):
      if clasesTrain[j]==i: 
        for l in range(len(datosTrain[j])):
          centroid[l] += datosTrain[j][l];
        nCentroid+=1;
    for j in range(len(centroid)):
      centroid[j] /= nCentroid
    pos = -1;
    minDist = MAX_VALUE
    for j in range(len(datosTrain)):
      if (clasesTrain[j]==i):
          dist = np.linalg.norm(centroid-datosTrain[j])
          if dist<minDist:
            minDist = dist
            pos = j
    if (pos>=0):
      deltaS.append(pos)


  if (nClases < 2):
      print("Todos los datos pertenecen a una unica clase");
      nClases = 1;      
      return np.append(X[int(deltaS[0]):int(deltaS[0]+1)],np.array(y[int(deltaS[0]):int(deltaS[0]+1)]).reshape(-1,1),axis=1)
      
  else:
    clase_minoritaria = np.argmin(np.unique(clasesTrain,return_counts=True)[1])
    for j in range(len(datosTrain)):
      if clase_minoritaria == clasesTrain[j] and not( j in deltaS):
        deltaS.append(j)

            #Una vez inicializado deltaS, procedemos a buscar en cada iteración
            #los elementos de los vectores nearest y rep
  while (len(deltaS)>0):
    for i in range(len(deltaS)):
      S[tamS] = deltaS[i]
      tamS+=1
    S = np.sort(S)  
    rep = np.random.randint(1, size=(len(datosTrain)))-1
    for i in range(len(datosTrain)):
      if not(i in S):
        for j in range(len(deltaS)):
          insert = False
          for l in (l for l in range(len(nearest[i])) if not insert):
            if nearest[i][l]<0:
              nearest[i][l] = deltaS[j]
              insert = True
            else:
            #Linea que es sustituida respecto al codigo original
              if dXij[nearest[i][l],i] >= dXij[deltaS[j],i]:
                for m in range(k-1,l,-1): 
                  nearest[i][m] = nearest[i][m-1]
                nearest[i][l] = deltaS[j]
                insert = True

        votes = np.random.randint(1, size=int(nClases))
        for j in range(len(nearest[i])):
          if nearest[i][j] >= 0:
            votes[int(clasesTrain[nearest[i][j]])]+=1
        max=votes[0]
        pos=0
        for j in range(0,len(votes)):
          if votes[j]>max:
            max = votes[j]
            pos = j
        if clasesTrain[i] != pos:
          for j in range(len(nearest[i])):
            if nearest[i][j] >=0:
              if rep[nearest[i][j]]<0:
                rep[nearest[i][j]]=i
              else:
                #Cambio respecto al codigo original
                if dXij[nearest[i][j], i] <= dXij[nearest[i][j],rep[nearest[i][j]]]:
                  rep[nearest[i][j]] = i

              #Una vez finalizado el calculo de elementos en T respecto a cada 
              #elemento del conjunto S, los candidatos en rep son examinados
              #tal que se agregan a delta S si el i'esimo elemento en rep fue declarado
              #y si ademas dicho elemento aun no ha sido nombrado como candidato por
              #otra observación [evitando de esta manera agregar a dicho candidato 2 veces]
    deltaS = []
    for i in range(tamS): #(rep[S[i]] in S)==False
      if (rep[S[i]]>=0 and not( rep[S[i]] in deltaS) ) :
        deltaS.append(rep[S[i]])
              #A partir de los indices almacenados en el vector S, y considerando 
              #ademas la longitud del mismo, aquellas observaciones en S, se recopilan 
              #tal que podamos obtener el conjunto de etiquetas S final.
  y_final = []
  X_final = []
  for j in range(tamS):
    y_final.append(clasesTrain[S[j]]) 
    X_final.append(datosTrain[S[j]])
  return np.append(X_final,np.array(y_final).reshape(-1,1),axis=1)