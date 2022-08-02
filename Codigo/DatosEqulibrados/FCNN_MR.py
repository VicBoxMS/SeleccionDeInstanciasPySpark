# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 18:52:22 2022

@author: VicBoxMS
"""

def calcular_centroides(datos,etiquetas):
  """
  -Entrada- 
  datos: Matriz de nxp donde n es el numero de observaciones y p el numero de 
  caracteristicas
  etiquetas: Vector de etiquetas de longitud n correspondiente a cada una de las 
  observaciones de X.
  -Salida- 
  Retorna un subconjunto de la muestra \DeltaS = [S(X),s(y)]
  el subconjunto corresponde a las observaciones mas cercanas para cada centroide,
  donde el numero de centroides es igaul al numero de clases.
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
  return np.append(datos[deltaS], etiquetas[deltaS].reshape(-1,1),axis=1)



##Funcion mapper para fcnn_mr
def mapper(Ti):  
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

##Funcion reducer para fcnn_mr
def reducer(x, y):
    ##(x e y) es una tupla, 0 y 1
    ##x[0] contiene la instancia+etiqueta
    ##x[1] contiene la distancia
    if x[1] <= y[1]: 
        return x
    else :
        return y

##Otros mappers auxiliares

def mapper_espejo(train_subset):
  train_subset = list(train_subset)
  train_fcnn = np.array(train_subset)
  return [0 , train_fcnn]

def fappend(x,y):
  return (np.append(x,y,axis=0))





def ejecutar_fcnn_mr(datos_rdd, numero_mappers=400):
    """
    Ejemplo de una funcion para utilizar fcnn_mr
    #Consideraciones:
        #Collect no es factible cuando toda la informacion no puede
        #ser almacenada en un solo nodo o en el nodo maestro
    
    -Entrada- 
    datos_rdd: Es un rdd(Resilient Distributed Datasets)
    numero_mappers: representa el numero de mappers en los cuales utilizar para el algoritmo.
      #se sugieren un numero de mappers tal que no se tengan mas de 
      #10,000 observaciones por mapper.
    -Salida- 
    Retorna un subconjunto  S = [S(X),s(y)] consistente.    
    """
    XY = np.array(datos_rdd.collect())
    X_train,y_train = XY[:,:-1],XY[:,-1] 
    inicio = time.time()
    ##Calcular centroides
    deltaS = calcular_centroides(X_train,y_train)
    ###Paralelizar los datos
    datos_rdd = sc.parallelize(np.append(X_train, y_train[:, None], axis = 1),numero_mappers)
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
    print('tiempo en google colab ', final-inicio ,'Tamaño final conjunto S : ' , S_actualizado.shape)
    return S_actualizado