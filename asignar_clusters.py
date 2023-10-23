import argparse
import json
import numpy as np
import pdb
def distancia(data):
    # Lógica para calcular la distancia utilizando los datos procesados
    pass



def numero_de_clusters(data):
    # Lógica para determinar el número de clusters utilizando los datos procesados
    fusion_data_dict = {}

    for line in lines[num_instances + 1:]:
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

    return dict(list(centroides.items())[:data])
    for cluster, centroide in zip(fusion_data_dict.keys(), centroides.values()):
        print(f"Centroide para Cluster {cluster}:", centroide)

def calcular_centroide_recursivo(cluster, instancias, lista, instancias_fusionadas):
    if cluster in lista.keys():  # Si es un índice de instancia
        a = lista[cluster]
        for v in a:
            calcular_centroide_recursivo(v, instancias, lista, instancias_fusionadas)
    else:
        instancias_fusionadas.append(instancias[cluster])
    return instancias_fusionadas

def asignar_instancias_nuevas(data):
    # Lógica para asignar nuevas instancias utilizando los datos procesados
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Descripción de tu programa')
    parser.add_argument('opcion', type=int, choices=[1, 2, 3], help='Elija una opción: 1 para distancia, 2 para número de clusters, 3 para asignar instancias nuevas')
    parser.add_argument('input_file', type=str, help='Ruta al archivo de entrada')
    parser.add_argument('num', type=int, help='Número de clusters')

    args = parser.parse_args()

    with open(args.input_file, 'r') as file:
        lines = file.readlines()
        if lines:
            first_line = lines[0].strip()
            if first_line.startswith("numero de instancias:"):
                num_instances = int(first_line.split(":")[1])
                instancias = {}
                for line in lines[1:num_instances + 1]:
                    parts = line.strip().split(',')
                    key = int(parts[0])
                    v = str(parts[1:])
                    v = v.strip('[ ]').replace("'", '').strip('[ ]')
                    v = v.split(',')
                    value = list(map(float, v))
                    instancias[key] = value
            else:
                print("El formato del archivo no es válido.")
        else:
            print("El archivo está vacío.")

    if args.opcion == 1:
        distancia(data)
    elif args.opcion == 2:
        centroides = numero_de_clusters(args.num)
        print("Últimos centroides:", centroides)
    elif args.opcion == 3:
        asignar_instancias_nuevas(data)


"""
fusion_data_dict = {}

for line in lines[num_instances + 1:]:
    columns = line.strip().split(',')
    if len(columns) >= 3:
        third_column = columns[2].strip()
        if third_column.startswith("clusters fusionados:"):
            parts = third_column.split(":")[1].split("[")
            if len(parts) == 2:
                key = int(parts[0])
                values = [int(x) for x in parts[1].rstrip(']').split(',')]
                fusion_data_dict[key] = values
"""