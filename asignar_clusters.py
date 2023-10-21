import argparse

def distancia(data):
    # Lógica para calcular la distancia utilizando los datos procesados
    pass

def numero_de_clusters(data):
    # Lógica para determinar el número de clusters utilizando los datos procesados
    pass

def asignar_instancias_nuevas(data):
    # Lógica para asignar nuevas instancias utilizando los datos procesados
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Descripción de tu programa')
    parser.add_argument('opcion', type=int, choices=[1, 2, 3], help='Elija una opción: 1 para distancia, 2 para número de clusters, 3 para asignar instancias nuevas')
    parser.add_argument('input_file', type=str, help='Ruta al archivo de entrada')

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
                    value = list(map(int, parts[1:]))
                    instancias[key] = value
            else:
                print("El formato del archivo no es válido.")
        else:
            print("El archivo está vacío.")

    if args.opcion == 1:
        distancia(data)
    elif args.opcion == 2:
        numero_de_clusters(data)
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