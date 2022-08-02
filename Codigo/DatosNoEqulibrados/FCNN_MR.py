# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:43:09 2022

@author: VicBoxMS
"""


def calcular_centroides(datos,etiquetas):
  """
  Funcion auxiliar para inicilizar el algoritmo FCNN_MR cuando se tienen 
  datos no equilibrados, la clase minoritaria se incluye en S, tal que la 
  selección o reducción se hace sobre la clase mayoritaria
  """    
  k=1
  datosTrain = datos
  clasesTrain = etiquetas
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

  return np.append(datos[deltaS], etiquetas[deltaS].reshape(-1,1),axis=1)



def mapper(Ti):  
   """ 
   Funcion mapper del metodo FCNN_MR
    Parameters
    ----------
    Ti : i-esima partición del RDD 

    Retorna
    el indice de la observación, la observación-etiqueta y la distancia
    -------
    """    
  Ti = list(Ti)
  ##En cada iteración eliminamos (no considerar) los elementos en S para cada particion
  A  = np.array(Ti)[:,:-1]
  B = np.array(S.value)[:,:-1]
  m = (A[:, None] == B).all(-1).any(1)
  Ti = list(np.array(Ti)[~m])
                                    ##Para cada instancia en la particion ti
                                    ##Buscaremos algun posible enemigo de S
  for index_test, point_test in enumerate(Ti):
    c = point_test[-1]
    dst = euclidean_distances(point_test[None, :-1],S.value[:,:-1]).ravel()
    idx = np.argmin(dst)
    c_estrella = S.value[int(idx)][-1]
    if c !=  c_estrella:
      yield int(idx) , (point_test[:],dst[int(idx)])
                                    ## retornamos el indice de S al inicio y como elemento de la tupla
                                    ## retornamos como primer elemento de la tupla a la instancia + etiqueta,
                                    ## como segundo elemento, retornamos la distancia
                                    
def reducer(x, y):
   """ 
   Funcion reduce del metodo FCNN_MR
    Parameters
    ----------
    x , y: una tupla con dos indices, 0 y 1
    x[0] contiene la instancia+etiqueta
    x[1] contiene la distancia
    Retorna
    La observación con la distancia mas cercana
    -------
    """
    ##(x e y) 
    ##x[0] contiene la instancia+etiqueta
    ##x[1] contiene la distancia
    if x[1] <= y[1]: 
        return x
    else :
        return y

def fcnn_mr_ejecucion(datos_rdd):    
    """
    Función secuencial del algoritmo FCNN en el paradigma de programación
    MapReduce
    
    datos_rdd : datos en forma de rdd
    
    RETORNA:
        S : Subconjunto de observaciones que representa 
        un subconjunto consistente bajo una distancia euclideana
    """
    
    XY = np.array(datos_rdd.collect())
    X_train,y_train = XY[:,:-1],XY[:,-1] 
    inicio = time.time()
    ##Calcular centroides
    deltaS = calcular_centroides(X_train,y_train)
    ###Filtro para eliminar centroides
    S = sc.broadcast(deltaS)
    A = X_train
    B = S.value[:,:-1]
    m = (A[:, None] == B).all(-1).any(1)
    X_sin_centroides = X_train[~m]
    y_sin_centroides =y_train[~m]
    ###Paralelizar los datos
    datos_rdd = sc.parallelize(np.append(X_sin_centroides, y_sin_centroides[:, None], axis = 1),400)  
    dimension = len(datos_rdd.take(1)[0])
    #Para iniciar el while
    valores = 3
    while valores!=[]:
      valores=datos_rdd.mapPartitions(mapper).reduceByKey(reducer).values().collect()
      if valores !=[]:
        #Actualizar el subconjunto S
        valores_filtrados = valores[0][0]
        for i in range(1,len(valores)):  
          valores_filtrados = np.append(valores_filtrados.reshape(-1,dimension), valores[i][0].reshape(-1,dimension),axis=0).copy()
        S_actualizado= np.append(S.value,np.array(valores_filtrados).reshape(-1,dimension),axis=0)
        S = sc.broadcast(S_actualizado)
      else:
        print('El algoritmo finalizó!')
    final = time.time()
    print('tiempo ', final-inicio ,'Tamaño final conjunto S : ' , S_actualizado.shape)
    return S_actualizado
