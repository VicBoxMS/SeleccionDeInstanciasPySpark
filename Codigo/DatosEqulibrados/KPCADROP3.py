# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 19:36:08 2022
@author: VicBoxMS
"""


def KPCADROP3(X,y,gamma_rbf=0.01,num_comp_principales=10):
  """
  Funcion KPCADROP3, tiene como objetivo reducir el numero de instancias 
  utilizando la distancia en un espacio de caracteristicas.
  A diferencia de KDROP3, la idea es extraer los componentes principales en un espacio de 
  caracteristicas de mayor dimensión.
  -Entrada- 
  X: Matriz de nxp tal que n es el numero de observaciones y p el numero de 
  caracteristicas
  y: Vector de etiquetas de longitud n correspondiente a cada una de las 
  observaciones de X.
  Parametro_gamma: corresponde al parametro que utiliza la función kernel rbf 
  su rango es (0,+ inf )
  num_comp_principales: indica el numero de componentes a extraer a partir de
  la matriz de gram.
  
  -Salida- 
  Retorna un subconjunto consistente reducido S = [S(X),s(y)] en forma de 
  Vector de Caracteristicas - Etiquetas de Clase, lo que tambien se conoce 
  como labeled point, la etiqueta se retorna en la ultima columna.
  """
  parametro_k=3
  parametro_k_filtro_enn=3
  ##Validacion del numero de clases
  conteo = np.unique(y,return_index=True,return_counts=True)
  clase_menor = np.argmin(conteo[2])
  if len(conteo[0])>1 and conteo[2][clase_menor]<=3:
    return(np.append(X[list(conteo[1])],np.array(y[list(conteo[1])]).reshape(-1,1),axis=1))

  k = 1
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
#  S = np.random.randint(1, size=(len(datosTrain)))+MAX_VALUE
#  tamS = 0

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
      print("Una clase");
      nClases = 1;
      return np.append(X[int(deltaS[0]):int(deltaS[0]+1)],np.array(y[int(deltaS[0]):int(deltaS[0]+1)]).reshape(-1,1),axis=1)
  ##Termina validacion

  def knn_euclideana(xTrain, xTest, k):
      """
      Encuentra los k vecinos mas cercanos de Xtest en Xtrain
      Entrada:
      xTrain = n x p
      xTest = m x p
      k = numero de vecinos a considerar
      Salida:
      dists = distancias entre xTrain y xTest de tamaño Size of n x m
      indices = matriz de kxm con los indices correspondientes a Xtrain
      """
      distances = euclidean_distances(xTrain,xTest)
      distances[distances < 0] = 0
      indices = np.argsort(distances, 0) #get indices of sorted items
      distances = np.sort(distances,0)   #distances sorted in axis 0
      #returning the top-k closest distances.
      return indices[0:k, : ].T , distances[0:k, : ].T  #

  knn= knn_euclideana

  def knn_predictions(xTrain,yTrain,xTest,k):
      """
      Entrada
      xTrain : n x p matriz. n=renglones p=caracteristicas
      yTrain : n x 1 vector. n=renglones de etiquetas de clase
      xTest : m x p matriz. m=renglones
      k : Numero de vecinos mas cercanos
      Salida
      predictions : etiquetas predichas para los m renglones de xTest 
      distancia : euclideana, rbf and poly
      """

      indices, distances = knn_euclideana(xTrain,xTest,k)    
      yTrain = yTrain.flatten()
      rows,columns  = indices.shape
      predictions = list()
      for j in range(rows):
          temp = list()
          for i in range(columns):
              cell = indices[j][i]
              temp.append(yTrain[cell])
          predictions.append(max(temp,key=temp.count)) #this is the key function, brings the mode value
      predictions=np.array(predictions)
      return predictions

  def distancias_enemigomascercano(X_filtrado,y_filtrado):
    """
    Funcion que computa para cada instancia la instancia 
    mas cercana de etiqueta diferente a el
    """
    vecinos,distancias = knn(X_filtrado, X_filtrado, k = X_filtrado.shape[0])
    def indice_enemigo(row,yvalues=y_filtrado):
      for c,i in enumerate(row):
        if yvalues[row[0]]!=yvalues[i]:
          return(c)
          break
    indice_ene=np.apply_along_axis(indice_enemigo,1,vecinos)
    distancia_enemigo=np.zeros(len(distancias))
    for i in range(len(distancias)):
      distancia_enemigo[i]=distancias[i][indice_ene[i]]
    return distancia_enemigo

  def suponer_no_esta(T,T_label,indice,parametro_k):
    """
    La idea es calcular el numero de observaciones correctamente clasificados
    cuando la i-esima instancia no esta en el conjunto de entrenamiento
    """
    T_virt=T.copy()
    T_virt[indice]=np.repeat(1000,T.shape[1])
    pnearest_virt , dd = knn(T_virt, T, parametro_k)
    n_filtrado=X_filtrado.shape[0]
    pasociados_virt=[[] for i in range(n_filtrado)]
    return sum(T_label[pnearest[pasociados[indice]][:,1]]==T_label[pnearest_virt[pasociados[indice]][:,-1]])

  def no_esta(T,T_label,indice,parametro_k):
    T_virt=T.copy()
    T_virt[indice]=np.repeat(10000000,T.shape[1])
    #clasificador.fit(T_virt,T_label)
    #dd,pnearest_virt=clasificador.kneighbors(T)
    pnearest_virt , dd = knn(T_virt, T, parametro_k)
    n_filtrado=X_filtrado.shape[0]
    pasociados_virt=[[] for i in range(n_filtrado)]
    for i in range(n_filtrado):
      for j in pnearest[i]:
          if i!=j:
            pasociados_virt[j].append(i)  
    return T_virt,pnearest_virt,pasociados_virt

################################################################################Comienza algoritmo
  ###########Filtro ENN
  transformador=KernelPCA(n_components=num_comp_principales,kernel='rbf',gamma=gamma_rbf)
  y_estimado=knn_predictions(X,y,X,parametro_k_filtro_enn)
  X_filtrado=transformador.fit_transform(X[y==y_estimado])
  y_filtrado=y[y==y_estimado]
  indices_filtrado=list(range(len(y_filtrado)))
  #############Ordenar
  X_inicial = X[y==y_estimado]
  y_inicial = y[y==y_estimado]
  d=distancias_enemigomascercano(X_filtrado,y_filtrado)
  y_filtrado=y_filtrado[np.argsort(d)[::-1]]  #Primero los mas alejados, al final los puntos fronterizos
  X_filtrado=X_filtrado[np.argsort(d)[::-1]]
  T=X_filtrado
  T_label=y_filtrado
  #T= KernelPCA(kernel='poly',n_components=45,gamma=0.5,degree=3).fit_transform(X_filtrado)
  #T_label=y_filtrado
  #clasificador.fit(X_filtrado,y_filtrado)
  #dd,pnearest=clasificador.kneighbors(T)
  pnearest,dd = knn(T, T, k = parametro_k)
  n_filtrado=X_filtrado.shape[0]
  pasociados=[[] for i in range(n_filtrado)]
  for i in range(n_filtrado):
    for j in pnearest[i]:
        if i!=j:
          pasociados[j].append(i)

  recolector1=[]#tqdm
  for i in (range(len(X_filtrado))):
    if len(pasociados[i])>0:
      with_=sum(y_filtrado[i]==y_filtrado[pasociados[i]])
      without_=suponer_no_esta(T=T,T_label=T_label,indice=i,parametro_k=parametro_k)  
    else:
      without_=0
      with_=30
  #  print(without_-with_>=0)
    if without_-with_>=0:
      recolector1.append(i)
      T,pnearest,pasociados=no_esta(T=T,T_label=T_label,indice=i,parametro_k=parametro_k)    
  id_select=set(range(len(X_filtrado)))-set(recolector1)
  id_select=list(id_select)
  return np.append(X_inicial[id_select] , (y_inicial[id_select]).reshape(-1,1),axis=1)