import argparse
import json
import pickle
import numpy as np
import pdb
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import bowAndTf_idf as bow
import clustering
import limpiarDatos as limpiar

archivoSalida = "salida.csv"


def asignar_labels(num_instancias, lista_clusters):
    """
    :param num_instancias:
    :param lista_clusters:
    :return:
    """
    labels = np.arange(num_instancias)
    lista = list(lista_clusters.keys())
    while True and len(lista) > 0:
        elemento = lista.pop()
        if elemento < num_instancias:
            break
        marcar_labels(lista_clusters[elemento], labels, elemento, lista, lista_clusters)
    unique_labels = np.unique(labels)
    return labels, unique_labels


def marcar_labels(cluster, labels, label, lista, lista_clusters):
    """
    :param cluster:
    :param labels:
    :param label:
    :param lista:
    :param lista_clusters:
    :return:
    """
    for elemento in cluster:
        idElemento = lista.index(elemento)
        lista.pop(idElemento)
        if not isinstance(lista_clusters[elemento][0], int):  # es decir, no es un cluster
            labels[elemento] = label
        else:
            marcar_labels(lista_clusters[elemento], labels, label, lista, lista_clusters)


def distancia(instancias, distancia):
    """
    :param instancias:
    :param distancia:
    :return:
    """
    # leer los clusters
    # clusters = {}
    num_inst = len(instancias)
    with open(archivoSalida, 'r') as file:
        lines = file.readlines()
    for line in lines[1:]:
        columns = line.strip().split(',')
        second_column = columns[1].strip()
        dist_union = float(second_column.split(":")[1])
        if dist_union > distancia:
            break
        third_column = columns[2] + ',' + columns[3]
        parts = third_column.split(":")
        key = int(parts[1])
        values = (parts[2])
        array = json.loads(values)
        instancias[key] = array

    labels, num_clusters = asignar_labels(num_inst, instancias)
    fichero = input("Nombre del fichero donde se va a guardar la asignación de instancias...\n")
    with open(fichero + "_info.txt", 'w') as archivo:
        archivo.write(f"Numero de clusters : {len(num_clusters)}\n")
        archivo.write(f"clusters: \n {num_clusters}")
    with open(fichero, 'w') as archivo:
        archivo.write("Instancia; Cluster; Vector\n")
        for i in range(0, len(labels)):
            linea = instancias[i][0]
            linea_str = np.array2string(linea, precision=8, separator=',', suppress_small=True)
            linea_str = linea_str.replace('\n', '')
            archivo.write(f"{i};{labels[i]};{linea_str}\n")
    print("El fichero se ha generado correctamente.")


def numero_de_clusters(numero):
    """
    :param numero: clusters que quieres conseguir
    :return: los centroides dado el número de clusters
    """
    # Lógica para determinar el número de clusters utilizando los datos procesados
    with open(archivoSalida, 'r') as file:
        lines = file.readlines()
        fusion_data_dict = {}
    for line in lines[1:]:
        columns = line.strip().split(',')
        if len(columns) >= 3:
            third_column = columns[2].strip()
            info = columns[3].strip()
            third_column += ', ' + info
            if third_column.startswith("clusters fusionados:"):
                parts = third_column.split(":")[1].split("[")
                parts += third_column.split(":")[2:]
                if len(parts) == 2:
                    key = int(parts[0])
                    values = [int(x.strip(' []')) for x in parts[1].split(",")]
                    fusion_data_dict[key] = values

    lista = fusion_data_dict.copy()
    centroides = {}

    for key, values in instancias.items():
        centroides[key] = values[0]

    for key, values in fusion_data_dict.items():
        instancias_fusionadas = []
        for v in values:
            centroide_fusionado = calcular_centroide_recursivo_instancias(v, instancias, lista, instancias_fusionadas)
            centroide = np.mean(centroide_fusionado, axis=0)
            centroides[key] = centroide

    i = len(fusion_data_dict) - 1
    imprimir = []
    while True:
        keys = list(fusion_data_dict.keys())
        current_key = keys[i]
        if numero == 1:
            imprimir.append(current_key)
        else:
            min_value = min(fusion_data_dict[current_key])
            max_value = max(fusion_data_dict[current_key])
            imprimir.append(min_value)
            imprimir.append(max_value)
            while max_value in fusion_data_dict and i > len(fusion_data_dict) - numero + 1:
                current_key = max_value
                if current_key in imprimir and current_key in fusion_data_dict:
                    imprimir.remove(current_key)
                min_value = min(fusion_data_dict[current_key])
                max_value = max(fusion_data_dict[current_key])
                imprimir.append(min_value)
                imprimir.append(max_value)
                i -= 1
        break

    diccionario = {}
    for key in imprimir:
        diccionario[key] = centroides[key]

    with open("instancias_clusters2.txt", 'w') as instancias_file:
        instancias_file.write(lines[0])
        for key, values in fusion_data_dict.items():
            if key in diccionario.keys():
                instancias_fusionadas = []
                vectores = []
                for v in values:
                    vectores = calcular_centroide_recursivo_instancias(v, instancias, lista, vectores)
                    centroide_fusionado = calcular_centroide_recursivo_indices(v, instancias, lista,
                                                                               instancias_fusionadas)
                # Eliminar saltos de línea de linea
                centroide_fusionado_str = str(centroide_fusionado).replace('\n', '')
                vectores_str = str(vectores).replace('\n', '')
                centroide_dict_str = str(diccionario[key][0]).replace('\n', '')
                instancias_file.write(
                    f"Cluster: {key}; Centroide: {centroide_dict_str}; Instancias: {centroide_fusionado_str}; Vectores: {vectores_str}\n")

        for key, values in diccionario.items():
            if key not in fusion_data_dict.keys():
                instancias_fusionadas = []
                vectores = []
                vectores = calcular_centroide_recursivo_instancias(key, instancias, lista, vectores)
                centroide_fusionado = calcular_centroide_recursivo_indices(key, instancias, lista,
                                                                           instancias_fusionadas)
                # Eliminar saltos de línea de linea
                centroide_fusionado_str = str(centroide_fusionado).replace('\n', '')
                vectores_str = str(vectores).replace('\n', '')
                centroide_dict_str = str(diccionario[key][0]).replace('\n', '')
                instancias_file.write(
                    f"Cluster: {key}; Centroide: {centroide_dict_str}; Instancias: {centroide_fusionado_str}; Vectores: {vectores_str}\n")
    print("guardado en instancias_cluster2")
    return diccionario


def calcular_centroide_recursivo_indices(cluster, instancias, lista, instancias_fusionadas):
    """
    Función recursiva para obtener los índices de los clusters
    """
    if cluster in lista.keys():  # Si es un índice de instancia
        a = lista[cluster]
        for v in a:
            calcular_centroide_recursivo_indices(v, instancias, lista, instancias_fusionadas)
    else:
        instancias_fusionadas.append(cluster)
    return instancias_fusionadas


def calcular_centroide_recursivo_instancias(cluster, instancias, lista, instancias_fusionadas):
    """
    Función recursiva para obtener las instancias de los clusters
    """
    if cluster in lista.keys():  # Si es un índice de instancia
        a = lista[cluster]
        for v in a:
            calcular_centroide_recursivo_instancias(v, instancias, lista, instancias_fusionadas)
    else:
        instancias_fusionadas.append(instancias[cluster])
    return instancias_fusionadas


def asignar_instancias_nuevas(numero, nuevas_instancias):
    """
    :param numero: número de clusters que quieres conseguir
    :param nuevas_instancias: nuevas instancias que quieres asignar
    :return:
    """
    # Calcular los centroides de los clusters a partir de los datos procesados
    centroides = numero_de_clusters(numero)  # centroides es un dicionario
    # Inicializar un diccionario para mantener un registro de las instancias asignadas a cada cluster
    asignaciones = {cluster: [] for cluster in centroides}
    # Recorrer las nuevas instancias y asignarlas al cluster más cercano
    for instancia in nuevas_instancias:
        cluster_mas_cercano = asignar_a_cluster(instancia, centroides)
        asignaciones[cluster_mas_cercano].append(instancia)

    # Devolver el diccionario de asignaciones
    return asignaciones


def euclidean_distance(point1, point2):
    """
    :param point1: punto 1 para distancia euclidea
    :param point2: punto 2 para distancia euclidea
    :return: distancia euclidea
    """
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


def asignar_a_cluster(instancia, centroides):
    """
    :param instancia: instancia a asignar a cluster
    :param centroides: centroides previamente calcular
    :return:
    """
    # Calcular la distancia entre la instancia y los centroides de los clusters
    distancias = []
    for key, centroide in centroides.items():
        distancia = euclidean_distance(instancia, centroide)
        distancias.append((key, distancia))

    # Encontrar el índice del cluster con la distancia mínima
    cluster_mas_cercano = min(distancias, key=lambda x: x[1])[0]

    return cluster_mas_cercano


def mostrar_instancias_originales(cluster):
    """
    :param cluster: cluster del cual quieres obtener las instancias
    :return: tupla de vector, distancia
    """
    with open(archivoSalida, 'r') as file:
        lines = file.readlines()
        fusion_data_dict = {}
    for line in lines[1:]:
        columns = line.strip().split(',')
        if len(columns) >= 3:
            third_column = columns[2].strip()
            info = columns[3].strip()
            third_column += ', ' + info
            if third_column.startswith("clusters fusionados:"):
                parts = third_column.split(":")[1].split("[")
                parts += third_column.split(":")[2:]
                if len(parts) == 2:
                    key = int(parts[0])
                    values = [int(x.strip(' []')) for x in parts[1].split(",")]
                    fusion_data_dict[key] = values

    lista = fusion_data_dict.copy()
    # Obtener el diccionario de vectores para el cluster dado
    vector_dict = {}
    for v in cluster.keys():
        vectores = []
        vectores = calcular_centroide_recursivo_indices(v, instancias, lista, vectores)
        vector_dict[v] = vectores

    distancias_cercanas = []

    for key in cluster:
        vectores_cluster_asociado = vector_dict.get(key, []) # Obtener el cluster asociado a la clave en cluster

        instancias_nuevas = cluster.get(key, [])  # Obtener la instancia nueva asociada a la clave en instancia
        for instancia_nueva in instancias_nuevas:
            # Calcular las distancias entre el valor de cluster y los vectores del cluster
            distancias = [(vector, euclidean_distance(instancia_nueva, instancias[vector])) for vector in vectores_cluster_asociado]

            # Ordenar las distancias de menor a mayor
            distancias_ordenadas = sorted(distancias, key=lambda x: x[1])

            # Obtener los dos vectores más cercanos
            dos_vectores_cercanos = [(distancia[0], distancia[1]) for distancia in distancias_ordenadas[:2]]

            # Guardar los dos vectores más cercanos en la lista principal
            distancias_cercanas.extend(dos_vectores_cercanos)
            # Mostrar las distancias más cercanas

    return distancias_cercanas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Descripción de tu programa')
    parser.add_argument('opcion', type=int, choices=[1, 2, 3],
                        help='Elija una opción: 1 para distancia, 2 para número de clusters, 3 para asignar instancias nuevas')
    parser.add_argument('input_file', type=str, help='Ruta al archivo de entrada')
    parser.add_argument('--distancia', type=float, help='Maxima distancia', required=False)
    parser.add_argument('--num', type=int, help='Número de clusters', required=False)
    parser.add_argument('--nuevas_instancias', type=str, help="fichero que contiene las nuevas instancias")

    args = parser.parse_args()

    archivoSalida = args.input_file

    try:
        with open('instancias.pickle', 'rb') as f:
            instancias = pickle.load(f)
    except FileNotFoundError:
        print("El archivo 'instancias.pickle' no se encontró.")
    except pickle.UnpicklingError:
        print("Error al cargar el archivo 'instancias.pickle'. Asegúrate de que el archivo sea válido.")

    if args.opcion == 1:
        if args.distancia is not None:
            distancia(instancias, args.distancia)

    elif args.opcion == 2 and args.num is not None:
        numero_de_clusters(args.num)

    elif args.opcion == 3 and args.nuevas_instancias is not None and args.num is not None:
        texto_original = pd.read_csv(args.nuevas_instancias)
        texto_original = texto_original['Post']
        limpiar.main(args.nuevas_instancias, "new_vector_result2.csv")
        _, documentos = bow.read_csv("new_vector_result2.csv")
        documentos = bow.preprocess_text(documentos)
        model = Doc2Vec.load("d2v.model")
        nuevas_instancias = bow.add_new_instance(model, documentos)
        nuevas_instancias = [nuevas_instancias[0].tolist()]

        instancias_origales = clustering.leer_datos2("doc2vec_results2.csv")
        instancias_origales.append(nuevas_instancias[0])
        num_instancias = len(instancias)
        instacias_con_PCA = clustering.reducir_dimensionalidad_pca(instancias_origales, len(instancias[0][0]))
        ultimos_valores = instacias_con_PCA[num_instancias:]
        asignaciones = asignar_instancias_nuevas(args.num, ultimos_valores)
        # Imprimir los resultados de asignaciones
        # for cluster, instancias_asignadas in asignaciones.items():
        #   print(f"Cluster {cluster}:")
        #  for i, instancia in enumerate(instancias_asignadas, 1):
        #     print(instancia)
        instancias_mas_cercanas = mostrar_instancias_originales(asignaciones)
        with open("Instancias_cercanas_originales.txt", 'w') as archivo:
            archivo.write(f"Instancia original: {texto_original[0]}\n")
            for indice, distancia in instancias_mas_cercanas:
                df_ml_dataset = pd.read_csv("500_Reddit_users_posts_labels.csv")
                instancia_original = df_ml_dataset.iloc[indice]['Post']
                archivo.write(f"Instancia cercana: {instancia_original}\n")
                archivo.write(f"Distancia: {distancia}\n")
