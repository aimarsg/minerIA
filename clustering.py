import copy
import csv
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt

# VARIABLES GLOBALES #
distancia_entre_instancias = {}
distancia_entre_clusters = {}
lista_clusters = {}


def leer_datos():
    input_file = sys.argv[1]
    df = pd.read_csv(input_file, sep="\t")
    # df = df.drop('User', axis=1)
    # df = df.drop('Label', axis=1)
    # df = df.head(5)
    print(df.head(5))
    return df.values.tolist()


def reducir_dimensionalidad_pca(data):
    # Realizar PCA para reducir la dimensionalidad a 500
    pca = PCA(n_components=5)
    df_reducido = pca.fit_transform(data)

    return df_reducido


def inicializar_distancias(clusters):
    global distancia_entre_clusters
    global distancia_entre_instancias

    distancia_minima = float('inf')
    #clusters_cercanos = (None, None)

    for cluster1 in range(len(clusters)):
        for cluster2 in range(cluster1 + 1, len(clusters)):
            instancia1 = clusters[cluster1][0]
            instancia2 = clusters[cluster2][0]
            distancia = euclidean_distance(instancia1, instancia2)
            distancia_entre_instancias[(cluster1, cluster2)] = distancia
            if distancia < distancia_minima:
                distancia_minima = distancia
                #clusters_cercanos = (cluster1, cluster2)

    distancia_entre_clusters = copy.copy(distancia_entre_instancias)
    #return clusters_cercanos


def fusionar_clusters(clusters_cercanos):
    global lista_clusters

    i1, i2 = clusters_cercanos
    print(f"fusionando clusters... {i1} y {i2}")
    clusters_fusionados = [i1, i2]
    id_nuevo_cluster = len(lista_clusters)
    lista_clusters[id_nuevo_cluster] = clusters_fusionados
    return id_nuevo_cluster


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def calcular_distancia_entre_clusters(idxcluster1, idxcluster2):
    global distancia_entre_instancias
    global distancia_entre_clusters
    global lista_clusters

    cluster1 = lista_clusters[idxcluster1]
    cluster2 = lista_clusters[idxcluster2]

    if not isinstance(cluster1[0], int):
        cluster1 = [idxcluster1]

    if not isinstance(cluster2[0], int):
        cluster2 = [idxcluster2]

    distancia_minima = float('inf')
    dist = 0
    for idxinstancia1 in cluster1:
        for idxinstancia2 in cluster2:
            tupla1 = (idxinstancia1, idxinstancia2)
            tupla2 = (idxinstancia2, idxinstancia1)

            if   (idxcluster1, idxinstancia2) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxcluster1, idxinstancia2)]
            elif (idxinstancia2, idxcluster1) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxinstancia2, idxcluster1)]
            elif (idxinstancia1, idxcluster2) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxinstancia1, idxcluster2)]
            elif (idxcluster2, idxinstancia1) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxcluster2, idxinstancia1)]

            elif tupla1 in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[tupla1]
            elif tupla2 in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[tupla2]

            elif tupla1 in distancia_entre_instancias.keys():
                dist = distancia_entre_instancias[tupla1]
            elif tupla2 in distancia_entre_instancias.keys():
                dist = distancia_entre_instancias[tupla2]

            else:

                """
                arg1 = idxcluster1
                arg2 = idxcluster2
                # comprobar si son clusters, si es asi calcular con las instancias de esos clusters
                if isinstance(lista_clusters[idxinstancia1][0], int):
                    arg1 = idxinstancia1
                if isinstance(lista_clusters[idxinstancia2][0], int):
                    arg2 = idxinstancia2
                dist = calcular_distancia_entre_clusters(arg1,arg2)
                """
                ## PARA DEBUG , QUITAR DESPUES ##
                print(f"Error : tupla1 {tupla1} \n clusters (d) {idxcluster1}: {cluster1} y {idxcluster2} :{cluster2} \n")
                input("continuar...")

            if dist < distancia_minima:
                distancia_minima = dist

    return distancia_minima


def actualizar_distancias(clusters_cercanos, id_nuevo_cluster):
    global distancia_entre_clusters

    i, j = clusters_cercanos
    print(f"actualizando distancias...{[i, j]} \n nuevo cluster: {id_nuevo_cluster}")

    # i y j se han fusionado en id_nuevo_cluster

    if (i, j) in distancia_entre_clusters:
        distancia_entre_clusters.pop((i, j))
    elif (j, i) in distancia_entre_clusters:
        distancia_entre_clusters.pop((j, i))
    else:
        print("error: no se ha encontrado en la lista de distancias entre clusters los dos clusters a fusionar")

    nueva_distancia_entre_clusters = copy.copy(distancia_entre_clusters)

    for cluster in lista_clusters.keys():
        # eliminar y añadir
        if cluster != i and cluster != id_nuevo_cluster:
            if cluster != j:

                if (i, cluster) in distancia_entre_clusters:
                    if id_nuevo_cluster == 55:
                        print(f"antes: {i}, {cluster}// ahora {id_nuevo_cluster}, {cluster}")
                    distancia = calcular_distancia_entre_clusters(id_nuevo_cluster, cluster)
                    nueva_distancia_entre_clusters[(id_nuevo_cluster, cluster)] = distancia
                    if id_nuevo_cluster==55 and cluster==46:
                        print("----------------------")
                        print(nueva_distancia_entre_clusters[(id_nuevo_cluster, cluster)])
                    nueva_distancia_entre_clusters.pop((i, cluster))

                elif (cluster, i) in distancia_entre_clusters:
                    if id_nuevo_cluster == 55:
                        print(f"antes: {cluster}, {i}// ahora {cluster}, {id_nuevo_cluster}")
                    distancia = calcular_distancia_entre_clusters(cluster, id_nuevo_cluster)
                    nueva_distancia_entre_clusters[(cluster, id_nuevo_cluster)] = distancia

                    nueva_distancia_entre_clusters.pop((cluster, i))

        if cluster != j and cluster != id_nuevo_cluster:
            if cluster != i:

                if (j, cluster) in distancia_entre_clusters:
                    nueva_distancia_entre_clusters.pop((j, cluster))
                elif (cluster, j) in distancia_entre_clusters:
                    nueva_distancia_entre_clusters.pop((cluster, j))

    distancia_entre_clusters = nueva_distancia_entre_clusters


def cluster_jerarquico(data, umbral):
    global distancia_entre_clusters
    global distancia_entre_instancias
    global lista_clusters

    j = 0
    for p in data:
        lista_clusters[j] = [p]
        j += 1

    inicializar_distancias(lista_clusters)
    i = 1

    with open(sys.argv[2], 'w') as archivo:

        while j > i:

            print("-------------------ITERACION " + str(i) + "-------------------------")

            clusters_cercanos, distancia_minima = min(distancia_entre_clusters.items(), key=lambda x: x[1])

            id_cluster_nuevo = fusionar_clusters(clusters_cercanos)
            actualizar_distancias(clusters_cercanos, id_cluster_nuevo)

            archivo.write(f"iteración: {i}, distancia: {distancia_minima}, clusters fusionados:  {id_cluster_nuevo }:{lista_clusters[id_cluster_nuevo]}\n")

            if distancia_minima > umbral:
                break
            #input("continuar ... ")
            i += 1

    return lista_clusters


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python clustering.py [fichero de entrada] [fichero de salida] [PCA/no]")
        print("Ejemplo: python clustering.py entrada.csv salida.csv PCA")
        exit(0)
    else:
        datos = leer_datos()
        if sys.argv[3] == 'PCA':
            datos = reducir_dimensionalidad_pca(datos)
            print("Dimensionalidad reducida con PCA:", datos.shape)

        print(f"Numero de instancias: {len(datos)}")
        umbral_clusters = 120  # Ajusta este valor según tus necesidades

        # Obtener clusters jerárquicos
        clusters = cluster_jerarquico(datos, umbral_clusters)
        print(clusters)
        print(len(clusters))
