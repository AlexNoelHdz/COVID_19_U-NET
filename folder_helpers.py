import os
import pickle

def list_full_paths(directory):
    paths_dict = {}
    # Recorrer los directorios de nivel superior: Test, Train, Val
    for root_dir in next(os.walk(directory))[1]:
        categories_dict = {}
        # Construir la ruta completa hacia este directorio
        path_to_root_dir = os.path.join(directory, root_dir)
        # Recorrer los subdirectorios: COVID-19, Non-COVID, Normal
        for category in next(os.walk(path_to_root_dir))[1]:
            types_dict = {}
            # Construir la ruta completa hacia la categoría
            path_to_category = os.path.join(path_to_root_dir, category)
            # Recorrer los tipos: images, lung masks
            for type_dir in next(os.walk(path_to_category))[1]:
                # Construir la ruta completa hacia este tipo
                path_to_type = os.path.join(path_to_category, type_dir)
                # Listar todos los archivos en este tipo
                full_paths = [os.path.join(path_to_type, file) for file in os.listdir(path_to_type)]
                # Asegurarse de que solo se incluyan archivos y no subdirectorios
                full_paths = [path for path in full_paths if os.path.isfile(path)]
                types_dict[type_dir] = full_paths
            categories_dict[category] = types_dict
        paths_dict[root_dir] = categories_dict

    return paths_dict

base_directory = r'C:\Users\anhernan\Python\DeepLearning\S07_Exam_2\m3ex02-data'
full_paths_dict = list_full_paths(base_directory)

# Guardar pickle
with open('full_paths_dict.pickle', 'wb') as handle:
    pickle.dump(full_paths_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Carga:
# with open('full_paths_dict.pickle', 'rb') as handle:
#     loaded_dict = pickle.load(handle)
# print(loaded_dict)