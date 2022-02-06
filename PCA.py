from datetime import datetime
from random import randint
import os
import os.path
from os import path
import cv2
from matplotlib import pyplot as plt
from sklearn.decomposition import IncrementalPCA
import numpy as np
import pickle
import tkinter as tk
from tkinter import messagebox
import errno


def get_image(index, input_array, dim_1=28, dim_2=28):
    return input_array[index].reshape(dim_1, dim_2)


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def get_image_shape(img_list):
    img = cv2.imread(img_list[0], 0)
    return img.shape[0], img.shape[1]


class App(tk.Frame):
    def __init__(self, master=None, **kw):
        self.parameters = {}
        tk.Frame.__init__(self, master=master, **kw)

        tk.Label(self, text="Input folder (images)", anchor="w", width=30).grid(row=0, column=0)
        self.question1 = tk.Entry(self)
        self.question1.grid(row=0, column=1)

        tk.Label(self, text="Image file endings (.*)", anchor="w", width=30).grid(row=1, column=0)
        self.question2 = tk.Entry(self)
        self.question2.grid(row=1, column=1)

        tk.Label(self, text="Output folder for model", anchor="w", width=30).grid(row=2, column=0)
        self.question3 = tk.Entry(self)
        self.question3.grid(row=2, column=1)

        tk.Label(self, text="Number of components", anchor="w", width=30).grid(row=3, column=0)
        self.question4 = tk.Entry(self)
        self.question4.grid(row=3, column=1)

        tk.Label(self, text=" ").grid(row=4, column=0)
        tk.Label(self, text="Press button to start", anchor="w", width=30).grid(row=5, column=0)

        tk.Button(self, text="Perform the PCA", bg='#54FA9B', command=self.read_input_and_run_pca).grid(row=5, column=1)

    def read_input_and_run_pca(self):
        self.parameters['input_folder'] = self.question1.get()
        self.parameters['file_ending'] = self.question2.get()
        self.parameters['output_folder'] = self.question3.get()
        self.parameters['PCA_count'] = self.question4.get()

        run_checks_and_pca(self.parameters)


def check_all_image_dimensions(dimensions, img_list):
    shape_list = [cv2.imread(img, 0).shape for img in img_list]
    return all([el[0] == dimensions[0] and el[1] == dimensions[1] for el in shape_list])


def input_checks(checked_input, checked_image_list, dimensions):
    if not path.exists(checked_input["input_folder"]):
        messagebox.showerror("Error ", f'Input file "{checked_input["input_folder"]}" not found ')
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "filename")

    try:
        int(checked_input['PCA_count'])
    except ValueError:
        messagebox.showerror("Error ", f'Please provide valid input for PCA count')
        raise ValueError("PCA count must be integer")

    try:
        os.makedirs(checked_input["output_folder"], exist_ok=True)
    except:
        messagebox.showerror("Error ", 'Could not create file. Please check permissions and/or path.')
        raise ValueError("Path invalid or user lacking permission.")

    if (len(checked_image_list)) * 3 <= int(checked_input['PCA_count']):
        messagebox.showerror("Error", 'Number of PCA components too large for chosen images.\n'
                                      f'Max number of components is 3 X number of images({len(checked_image_list)}).')
        raise ValueError("PCA count must not be greater than count of data point entries.")

    if not check_all_image_dimensions(dimensions, checked_image_list):
        messagebox.showerror("Error", 'Images are of different sizes')
        raise ValueError("PCA count must not be greater than count of data point entries.")


def perform_pca_for_images(pca_count, img_list, dimensions):
    incremental_pca = IncrementalPCA(n_components=pca_count)
    checked_images = []
    last_chunk = int(len(img_list) / pca_count) - 1
    for i in range(0, last_chunk):
        x_temp = [item.reshape(dimensions[0] * dimensions[1])
                  for sublist in
                  [np.split(cv2.cvtColor(cv2.imread(fig_name), cv2.COLOR_BGR2RGB), 3, axis=2)
                   for fig_name in img_list[i * pca_count: (i + 1) * pca_count]]
                  for item in sublist]
        checked_images.extend(range(i * pca_count, (i + 1) * pca_count))
        print(f'Started PCA for chunk {i} of {last_chunk} at: {datetime.now().strftime("%H:%M:%S")}')
        incremental_pca.partial_fit(x_temp)
        print(f'Finished PCA for chunk {i} of {last_chunk} at: {datetime.now().strftime("%H:%M:%S")}')
        print()
    checked_images.extend(range(last_chunk * pca_count, len(img_list)))
    x_remaining = [item.reshape(dimensions[0] * dimensions[1])
                   for sublist in
                   [np.split(cv2.cvtColor(cv2.imread(fig_name), cv2.COLOR_BGR2RGB), 3, axis=2)
                    for fig_name in img_list[last_chunk * pca_count: len(img_list)]]
                   for item in sublist]
    print(f'Started PCA for final and largest chunk at: {datetime.now().strftime("%H:%M:%S")}')
    incremental_pca.partial_fit(x_remaining)
    print(f'Finished PCA learning at: {datetime.now().strftime("%H:%M:%S")}')
    print()
    return incremental_pca


def apply_pca(i_pca, img_list, dimensions, pca_count):
    print(f'Starting PCA transform at: {datetime.now().strftime("%H:%M:%S")}')
    transformed_images = []
    last_chunk = int(len(img_list) / pca_count) - 1
    for i in range(0, last_chunk):
        x_temp = [item.reshape(dimensions[0] * dimensions[1])
                  for sublist in
                  [np.split(cv2.cvtColor(cv2.imread(fig_name), cv2.COLOR_BGR2RGB), 3, axis=2)
                   for fig_name in
                   img_list[i * pca_count: (i + 1) * pca_count]]
                  for item in sublist]
        transformed_images.extend(i_pca.transform(x_temp))
    x_remaining = [item.reshape(dimensions[0] * dimensions[1])
                   for sublist in
                   [np.split(cv2.cvtColor(cv2.imread(fig_name), cv2.COLOR_BGR2RGB), 3, axis=2)
                    for fig_name in img_list[last_chunk * pca_count: len(img_list)]]
                   for item in sublist]
    transformed_images.extend(i_pca.transform(x_remaining))
    print(f'PCA transform finished at: {datetime.now().strftime("%H:%M:%S")}')
    print()
    return transformed_images


def save_image_components(transformed_images, full_path, image_names):
    pca_data_file = open(full_path, "w")
    for image_index in range(int(len(transformed_images) / 3)):
        image_data = transformed_images[image_index * 3:image_index * 3 + 3]
        for colour_PCA_data in image_data:
            pca_data_file.write(image_names[image_index].split("\\")[-1] + " $:$ [" +
                                ", ".join(map(str, colour_PCA_data)) + "]\n")
    pca_data_file.close()


def get_reconstructed_data(pca_data_filename, model_filename):
    principal_components = []
    with open(pca_data_filename) as f:
        for line in f:
            pca_colour_data = [float(pca_element.strip(" ,]\n")) for pca_element in line.split("[")[1].split(" ")]
            principal_components.append(pca_colour_data)
    pca_transformation = pickle.load(open(model_filename, 'rb'))
    return principal_components, pca_transformation


def reconstruct_image(test_image_index, pca_data, pca_model, dimensions):
    returned_image = pca_model.inverse_transform(pca_data[test_image_index * 3:test_image_index * 3 + 3])
    returned_image[returned_image < 0] = 0
    returned_image[returned_image > 255] = 255
    return np.dstack((returned_image[0].reshape(dimensions[0], dimensions[1]),
                      returned_image[1].reshape(dimensions[0], dimensions[1]),
                      returned_image[2].reshape(dimensions[0], dimensions[1]))).astype(np.uint8)


def compare_pca_to_original(test_image_index, img_list, model_filename, pca_data_filename, dimensions):
    plt.show()
    original_test_image = cv2.cvtColor(cv2.imread(img_list[test_image_index]), cv2.COLOR_BGR2RGB)
    plt.figure(1)
    plt.imshow(original_test_image)

    pca_data, pca_model = get_reconstructed_data(pca_data_filename, model_filename)
    reconstructed_test_image = reconstruct_image(test_image_index, pca_data, pca_model, dimensions)

    plt.figure(2)
    plt.imshow(reconstructed_test_image)


def run_checks_and_pca(input_from_gui):
    # read the relevant image data
    img_list = [input_from_gui["input_folder"] + '\\' + f
                for f in os.listdir(input_from_gui["input_folder"]) if f.endswith(input_from_gui["file_ending"])]
    dim1, dim2 = get_image_shape(img_list)

    # check the data
    input_checks(input_from_gui, img_list, [dim1, dim2])

    # develop PCA
    i_pca = perform_pca_for_images(int(input_from_gui['PCA_count']), img_list, [dim1, dim2])

    # apply PCA
    transformed_images = apply_pca(i_pca, img_list, [dim1, dim2], int(input_from_gui['PCA_count']))

    # save model
    backslash = "\\"
    model_filename = input_from_gui["output_folder"] + backslash + "PCA_model.sav"
    pickle.dump(i_pca, open(model_filename, 'wb'))
    print(f'PCA model saved to {input_from_gui["output_folder"] + backslash + "PCA_model.sav"}')

    # save images
    pca_data_filename = input_from_gui["output_folder"] + "\\PCA_data.txt"
    save_image_components(transformed_images, pca_data_filename, img_list)
    print(f'PCA image data saved to {input_from_gui["output_folder"] + backslash + "PCA_data.txt"}')

    # review reconstructed images
    test_image_index = randint(0, len(img_list) - 1)
    compare_pca_to_original(test_image_index, img_list, model_filename, pca_data_filename, [dim1, dim2])
    plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    App(root).grid()
    root.mainloop()
