import argparse
import json
import pickle
import csv
import numpy as np

archivoSalida = "salida.csv"

def asignar_labels(num_instancias, lista_clusters):
    labels = np.arange(num_instancias)
    lista = list(lista_clusters.keys())
    while True and len(lista)>0:
        elemento = lista.pop()
        if elemento<num_instancias:
            break
        marcar_labels(lista_clusters[elemento], labels, elemento,lista ,  lista_clusters)
    unique_labels = np.unique(labels)
    return labels, unique_labels


def marcar_labels(cluster, labels, label, lista, lista_clusters):
    for elemento in cluster:
        idElemento = lista.index(elemento)
        lista.pop(idElemento)
        if not isinstance(lista_clusters[elemento][0], int): #es decir, no es un cluster
            labels[elemento] = label
        else:
            marcar_labels(lista_clusters[elemento], labels, label, lista, lista_clusters)


def distancia(instancias, distancia):
    # leer los clusters
    # clusters = {}
    num_inst = len(instancias)
    with open(archivoSalida, 'r') as file:
        lines = file.readlines()
    for line in lines[1:]:
        columns = line.strip().split(',')
        second_column = columns[1].strip()
        dist_union = float(second_column.split(":")[1])
        if dist_union>distancia:
            break
        third_column = columns[2]+','+columns[3]
        parts = third_column.split(":")
        key = int(parts[1])
        values = (parts[2])
        array = json.loads(values)
        instancias[key] = array

    labels, num_clusters = asignar_labels(num_inst, instancias)
    fichero = input("Nombre del fichero donde se va a guardar la asignación de instancias...\n")
    with open(fichero+"_info.txt", 'w') as archivo:
        archivo.write(f"Numero de clusters : {len(num_clusters)}\n")
        archivo.write(f"clusters: \n {num_clusters}")
    with open(fichero, 'w') as archivo:
        archivo.write("Instancia; Cluster; Vector\n")
        for i in range (0, len(labels)):
            linea = instancias[i][0]
            linea_str = np.array2string(linea, precision=8, separator=',', suppress_small=True)
            linea_str = linea_str.replace('\n', '')
            archivo.write(f"{i};{labels[i]};{linea_str}\n")
    print("El fichero se ha generado correctamente.")


def numero_de_clusters(numero):
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
    for key, values in fusion_data_dict.items():
        instancias_fusionadas = []
        for v in values:
            centroide_fusionado = calcular_centroide_recursivo(v, instancias, lista, instancias_fusionadas)
            centroide = np.mean(centroide_fusionado, axis=0)
            centroides[key] = centroide

    return dict(list(centroides.items())[-numero:])


def calcular_centroide_recursivo(cluster, instancias, lista, instancias_fusionadas):
    if cluster in lista.keys():  # Si es un índice de instancia
        a = lista[cluster]
        for v in a:
            calcular_centroide_recursivo(v, instancias, lista, instancias_fusionadas)
    else:
        instancias_fusionadas.append(instancias[cluster][0])
    return instancias_fusionadas


def asignar_instancias_nuevas(numero, nuevas_instancias):
    # Calcular los centroides de los clusters a partir de los datos procesados
    centroides = numero_de_clusters(numero) #centroides es un dicionario

    # Inicializar un diccionario para mantener un registro de las instancias asignadas a cada cluster
    asignaciones = {i: [] for i in range(len(centroides))}

    # Recorrer las nuevas instancias y asignarlas al cluster más cercano
    for instancia in nuevas_instancias:
        cluster_mas_cercano = asignar_a_cluster(instancia, centroides)
        asignaciones[cluster_mas_cercano].append(instancia)

    # Devolver el diccionario de asignaciones
    return asignaciones

def asignar_a_cluster(instancia, centroides):
    # Calcular la distancia entre la instancia y los centroides de los clusters
    distancias = [(instancia, centroide[0]) for key, centroide in centroides.items()]

    # Encontrar el índice del cluster con la distancia mínima
    cluster_mas_cercano = distancias.index(min(distancias))

    return cluster_mas_cercano


def cargar_nuevas_instancias(ruta_archivo):
    nuevas_instancias = []
    with open(ruta_archivo, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Saltar la primera fila si contiene encabezados
        for row in reader:
            # Aquí, procesa cada fila del archivo CSV y agrega las instancias a nuevas_instancias
            # En este ejemplo, se convierten los valores a punto flotante y se omiten las dos últimas columnas (User y Label)
            nueva_instancia = [float(value) for value in row[:-2]]
            nuevas_instancias.append(nueva_instancia)
    return nuevas_instancias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Descripción de tu programa')
    parser.add_argument('opcion', type=int, choices=[1, 2, 3],
                        help='Elija una opción: 1 para distancia, 2 para número de clusters, 3 para asignar instancias nuevas')
    parser.add_argument('--input_file', type=str, help='Ruta al archivo de entrada', required=False)
    parser.add_argument('--distancia', type=float, help='Maxima distancia', required=False)
    parser.add_argument('--num', type=int, help='Número de clusters', required=False)

    args = parser.parse_args()

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

    elif args.opcion == 2:
        centroides = numero_de_clusters(args.num)
        print(centroides)
    elif args.opcion == 3:
        nuevas_instancias = cargar_nuevas_instancias(args.input_file)
        asignaciones = asignar_instancias_nuevas(args.num, nuevas_instancias)

        # Imprimir los resultados de asignaciones
        for cluster, instancias_asignadas in asignaciones.items():
            print(f"Cluster {cluster}:")
            for i, instancia in enumerate(instancias_asignadas, 1):
                print(f"Instancia {i}")
