# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:44:58 2022

@author: VicBoxMS
"""

def KPCAFCNN(X,y,parametro_gamma):
  """
  Funcion KPCAFCNN para datos no equilibrados, en ese sentido lo unico que 
  cambia es la inicialización del metodo.
  tiene como objetivo reducir el numero de instancias considerando  
  A diferencia de KFCNN, la idea primero es reducir la dimensionalidad no lineal 
  mediante el uso de la matriz de gram, para posteriormente calcular la distancia
  de las observaciones y llevar a cabo FCNN de manera normal.
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
  #No se realiza un escalamiento interno
  #############KPCA
  transformer = KernelPCA(kernel='rbf',n_components=10,gamma=parametro_gamma)
  datosTrain = transformer.fit_transform(X)
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

              #Validacion de numero de clases diferentes
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
              if (np.linalg.linalg.norm(datosTrain[nearest[i][l]]-datosTrain[i]) >= np.linalg.linalg.norm(datosTrain[deltaS[j]]-datosTrain[i])):
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
                if (np.linalg.linalg.norm(datosTrain[nearest[i][j]]-datosTrain[i])<=np.linalg.linalg.norm(datosTrain[nearest[i][j]]-datosTrain[rep[nearest[i][j]]])):
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
    y_final.append(y[S[j]]) 
    X_final.append(X[S[j]])
  return np.append(X_final,np.array(y_final).reshape(-1,1),axis=1)