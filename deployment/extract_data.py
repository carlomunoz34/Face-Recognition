import numpy as np
from glob import glob
import os
from shutil import copyfile


if __name__ == '__main__':
    files_path = "/home/carlo/Documentos/Datasets/lfw/"
    files = glob(files_path + "*")
    data_path = "/home/carlo/Documentos/Proyectos/Face-Recognition/data/"
    persons_path = "/home/carlo/Documentos/Proyectos/Face-Recognition/data/persons.csv"

    csv_string = 'id,name,file\n'

    for i in range(100):
        file = np.random.choice(files)
        file_name = files_path + file.split(os.sep)[-1]
        old_image = glob(file_name + '/*')[0]

        new_file = data_path + file_name

        names = file.split(os.sep)[-1].split("_")
        name = f'{names[0]} {names[1]}' if len(names) == 2 else names[0]

        new_image_relative = "images/" + glob(file + '/*')[0].split(os.sep)[-1]
        new_image = data_path + new_image_relative

        csv_string += f'{i},{name},{new_image_relative}\n'

        copyfile(old_image, new_image)

    f = open(persons_path, 'w')
    f.write(csv_string)
    f.close()
