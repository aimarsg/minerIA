import argparse
import copy
import pickle
import sys
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# VARIABLES GLOBALES #
distancia_entre_instancias = {}
distancia_entre_clusters = {}
lista_clusters = {}
t_distancia = 1 # default: single link
t_mink = 2      # default: distancia euclidea
fich_salida = 'salida.txt' # default


def leer_datos(input_file):
    df = pd.read_csv(input_file, sep=",")
    df = df.drop('User', axis=1)
    df = df.drop('Label', axis=1)
    with open(sys.argv[2], 'w') as archivo:
        archivo.write(f"numero de instancias: {len(df.index)} \n")
    return df.values.tolist()


def reducir_dimensionalidad_pca(data, dim):
    pca = PCA(n_components=dim)
    df_reducido = pca.fit_transform(data)
    return df_reducido


def inicializar_distancias(clusters):
    global distancia_entre_clusters
    global distancia_entre_instancias
    distancia_minima = float('inf')

    for cluster1 in range(len(clusters)):
        for cluster2 in range(cluster1 + 1, len(clusters)):
            instancia1 = clusters[cluster1][0]
            instancia2 = clusters[cluster2][0]
            distancia = distancia_minkowski(instancia1, instancia2)
            distancia_entre_instancias[(cluster1, cluster2)] = distancia
            if distancia < distancia_minima:
                distancia_minima = distancia

    distancia_entre_clusters = copy.copy(distancia_entre_instancias)


def fusionar_clusters(clusters_cercanos):
    global lista_clusters
    i1, i2 = clusters_cercanos
    print(f"fusionando clusters... {i1} y {i2}")
    clusters_fusionados = [i1, i2]
    id_nuevo_cluster = len(lista_clusters)
    lista_clusters[id_nuevo_cluster] = clusters_fusionados
    return id_nuevo_cluster


def distancia_minkowski(point1, point2):
    if np.isnan(point1).any():
        print("point1 contiene NaN.")
    if np.isinf(point1).any():
        print("point1 contiene valores infinitos.")

    # Verificar si point2 contiene NaN o inf
    if np.isnan(point2).any():
        print("point2 contiene NaN.")
    if np.isinf(point2).any():
        print("point2 contiene valores infinitos.")

    if len(point1)!=len(point2):
        print("distinta longitud")
        input("error")
    # calcula la distancia minkowski del valor establecido en el argumetno t_mink

    res = np.power(np.sum((np.array(point1) - np.array(point2)) ** t_mink), (1/5))
    return res


def calcular_distancia_entre_clusters(idxcluster1, idxcluster2):
    global distancia_entre_instancias
    global distancia_entre_clusters
    global lista_clusters

    distancias = []

    cluster1 = lista_clusters[idxcluster1]
    cluster2 = lista_clusters[idxcluster2]

    if not isinstance(cluster1[0], int):
        cluster1 = [idxcluster1]

    if not isinstance(cluster2[0], int):
        cluster2 = [idxcluster2]

    for idxinstancia1 in cluster1:
        for idxinstancia2 in cluster2:
            tupla1 = (idxinstancia1, idxinstancia2)
            tupla2 = (idxinstancia2, idxinstancia1)

            if (idxcluster1, idxinstancia2) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxcluster1, idxinstancia2)]
                distancias.append(dist)
            elif (idxinstancia2, idxcluster1) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxinstancia2, idxcluster1)]
                distancias.append(dist)
            elif (idxinstancia1, idxcluster2) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxinstancia1, idxcluster2)]
                distancias.append(dist)
            elif (idxcluster2, idxinstancia1) in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[(idxcluster2, idxinstancia1)]
                distancias.append(dist)

            elif tupla1 in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[tupla1]
                distancias.append(dist)
            elif tupla2 in distancia_entre_clusters.keys():
                dist = distancia_entre_clusters[tupla2]
                distancias.append(dist)

            elif tupla1 in distancia_entre_instancias.keys():
                dist = distancia_entre_instancias[tupla1]
                distancias.append(dist)
            elif tupla2 in distancia_entre_instancias.keys():
                dist = distancia_entre_instancias[tupla2]
                distancias.append(dist)

            else:
                print(f"Error : tupla1 {tupla1} \n clusters (d) {idxcluster1}: {cluster1} y {idxcluster2} :{cluster2} \n")
                input("continuar...")

    # LOGICA PARA SACAR LA DISTANCIA ENTRE LOS CLUSTERS
    # 1> SINGLE LINK
    # 2> COMPLETE LINK
    # 3> MEAN LINK
    if t_distancia == 1:
        # distancia minima
        # print(min(distancias))
        return min(distancias)

    elif t_distancia == 2:
        # distancia maxima
        # print(max(distancias))
        return max(distancias)

    elif t_distancia == 3:
        # distancia media
        return sum(distancias) / len(distancias)


def actualizar_distancias(clusters_cercanos, id_nuevo_cluster):
    global distancia_entre_clusters

    i, j = clusters_cercanos
    print(f"actualizando distancias...{[i, j]} \n nuevo cluster: {id_nuevo_cluster}")

    if (i, j) in distancia_entre_clusters:
        distancia_entre_clusters.pop((i, j))
    elif (j, i) in distancia_entre_clusters:
        distancia_entre_clusters.pop((j, i))
    else:
        print("error: no se ha encontrado en la lista de distancias entre clusters los dos clusters a fusionar")

    nueva_distancia_entre_clusters = copy.copy(distancia_entre_clusters)

    for cluster in lista_clusters.keys():
        if cluster != i and cluster != id_nuevo_cluster:
            if cluster != j:

                if (i, cluster) in distancia_entre_clusters:
                    distancia = calcular_distancia_entre_clusters(id_nuevo_cluster, cluster)
                    nueva_distancia_entre_clusters[(id_nuevo_cluster, cluster)] = distancia
                    nueva_distancia_entre_clusters.pop((i, cluster))

                elif (cluster, i) in distancia_entre_clusters:
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


def asignar_labels(num_instancias):
    labels = np.arange(num_instancias)
    lista = list(lista_clusters.keys())
    while True and len(lista) > 0:
        elemento = lista.pop()
        if elemento < num_instancias:
            break
        marcar_labels(lista_clusters[elemento], labels, elemento, lista)
    unique_labels = np.unique(labels)
    return labels, len(unique_labels)


def marcar_labels(cluster, labels, label, lista):
    for elemento in cluster:
        idElemento = lista.index(elemento)
        lista.pop(idElemento)
        if not isinstance(lista_clusters[elemento][0], int):  # es decir, no es un cluster
            labels[elemento] = label
        else:
            marcar_labels(lista_clusters[elemento], labels, label, lista)


def cluster_jerarquico(data):
    global distancia_entre_clusters
    global distancia_entre_instancias
    global lista_clusters

    j = 0
    for p in data:
        lista_clusters[j] = [p]
        j += 1

    with open('instancias.pickle', 'wb') as f:
        pickle.dump(lista_clusters, f)

    inicializar_distancias(lista_clusters)
    i = 1

    Z = []  # Matriz de enlace
    silhouette_scores = []

    with open(sys.argv[2], 'a') as archivo:
        while j > i:
            print("-------------------ITERACION " + str(i) + "-------------------------")

            clusters_cercanos, distancia_minima = min(distancia_entre_clusters.items(), key=lambda x: x[1])

            id_cluster_nuevo = fusionar_clusters(clusters_cercanos)
            actualizar_distancias(clusters_cercanos, id_cluster_nuevo)

            archivo.write(
                f"iteración: {i}, distancia: {distancia_minima}, clusters fusionados: {id_cluster_nuevo}:{lista_clusters[id_cluster_nuevo]}, ")

            labels, unique = asignar_labels(num_instancias=j)
            # Calcular Silhouette para cada iteración
            if unique > 1:
                silhouette = silhouette_score(data, labels)
                silhouette_scores.append(silhouette)
                archivo.write(f"Silhouette Score: {silhouette}\n")

            Z.append(
                [clusters_cercanos[0], clusters_cercanos[1], distancia_minima, len(lista_clusters[id_cluster_nuevo])])

            i += 1

    return lista_clusters, np.array(Z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering jerarquico")
    parser.add_argument('fich_entrada', type=str, help="Ruta al fichero de entrada")
    parser.add_argument('salida', type=str, help="Nombre del archivo de salida")
    parser.add_argument('--PCA', type=int, help="opcional, indica la dimension del PCA")
    parser.add_argument('--distancia', type=int, choices=[1, 2, 3], help="opcional, indica la distancia entre clusters a utilizar\n"
                                                                        "1 para single link\n"
                                                                      "2 para complete link\n"
                                                                      "3 para mean link")
    parser.add_argument('--mink', type=int , help='Distancia minkowski', required=False)

    args = parser.parse_args()

    fich_salida = args.salida

    datos = leer_datos(args.fich_entrada)

    if args.PCA is not None:
        datos = reducir_dimensionalidad_pca(datos, args.PCA)
        print("Dimensionalidad reducida con PCA:", datos.shape)

    if args.distancia is not None:
        t_distancia = args.distancia

    if args.mink is not None:
        if args.mink%2 == 0:
            t_mink = args.mink
        else:
            print("distancia no valida, se va a utilizar la euclidea")
            input("pulsa para continuar... ")


    print(f"Numero de instancias: {len(datos)}")

    # Obtener clusters jerárquicos
    clusters, Z = cluster_jerarquico(datos)

    print(clusters)

    # Mostrar dendrograma
    dendrogram(Z)
    plt.gcf().set_size_inches(38.4, 21.6)
    plt.savefig(args.salida+".png", dpi=500, bbox_inches='tight')
    plt.show()
