# Multiple image PCA tutorial
This is a tutorial on how to apply the PCA transform to several images simultaneosly. The IncrementalPCA is used which enables us to work with an otherwise too large body of image data in separated chunks. So the images are not all processed at once but instead, they are separated into chunks, of which, all but the last have the size equal to the specified number of components, while the last one is appropriately larger to accommodate for the leftover (max: 2x number of components - 1). At the start, a GUI opens up where the user provides:
 - the input folder which contains the images
 - the image suffix (e.g.: jpg, png, etc. )
 - the output folder where the model as where as the image data will be stored
 - the number of components to which the data dimensionality will be reduced
## Overview 
The main question when dealing with image transform by PCA is how to translate the image data into somehting that the PCA transform will understand, i.e. points in some hyperspace (fancy word for high dimensional vercotrs basically). In this tutorial each channel of an image is taken as its own variable so each image is represented as three variables (RGB). The length of the vectors upon which the PAC must operate is `dim1*dim2` where `dim1` and `dim2` are the horizontal and vertical size of the images respectively. This is slightly different than what tends to be taken when applying PCA to a single image where the image's spreadsheet is generally also the matrix supplied into the PCA. 

Basic checks are performed over the input like whether the input directory exists, whether the images are of the same sizes (it makes no sense in bringing them to the same PCA transform matrix otherwise) and whether the PCA count is not too large for all the images in the provided directory which would cause an undefined system of equations in the background. After providing the required data, the user can start the PCA for the selected images. 

For a large collection of images the entire process may take a long time so there is some output printed to the console that helps the user predict when the process will be finished - the output informs the user of how many chunks have been passed out of the total and at what times they were finished.
After the PCA is finished, the model is saved using pickle.dump() to a file named "PCA_model.sav" in the output folder. The output folder is specified through the GUI before the PCA is ran. It is worth noting, that other manners of saving PCA (for example storing the coordinates of each principal component) would be more space efficient, but this is a tutorial and pickle is the generally used pythonic way of saving models so we're going to stick with that. 

The PCA data is stored in a textfile called "PCA_data.txt". In this text file, each image is represented by three lines, each of which saves the PCAs for the corresponding colour channel. 

To read in the data the `get_reconstructed_data` method is used as shown here:
```python
def get_reconstructed_data(pca_data_filename, model_filename):
    principal_components = []
    with open(pca_data_filename) as f:
        for line in f:
            pca_colour_data = [float(pca_element.strip(" ,]\n")) for pca_element in line.split("[")[1].split(" ")]
            principal_components.append(pca_colour_data)
    pca_transformation = pickle.load(open(model_filename, 'rb'))
    return principal_components, pca_transformation
```

The images are reconstructed using the `reconstruct_image` method below:
```python
def reconstruct_image(test_image_index, pca_data, pca_model, dimensions):
    returned_image = pca_model.inverse_transform(pca_data[test_image_index * 3:test_image_index * 3 + 3])
    returned_image[returned_image < 0] = 0
    returned_image[returned_image > 255] = 255
    return np.dstack((returned_image[0].reshape(dimensions[0], dimensions[1]),
                      returned_image[1].reshape(dimensions[0], dimensions[1]),
                      returned_image[2].reshape(dimensions[0], dimensions[1]))).astype(np.uint8)
```


Note here, the statements that lock the reconstructed values to above zero and below 255 in the `reconstruct_image` method above. This is done to avoid getting values that cannot be interpreted by matplotlib and would cause those weird dots as can be seen in the example below.
| ![Uncleaned PCA output sample](https://www.askpython.com/wp-content/uploads/2020/10/compressed-image-with-first-50-Principal-Components.jpeg.webp) | 
|:--:| 
| *Uncleaned PCA output of an image -  some colour channels go below 0 or over 255 creating invalid pixels* |


The images obtained from the methods `get_reconstructed_data` and `reconstruct_image` are sorted in such a way that matplotlib can show them without needing to reorder any colour channels. 
The repo contains a zip file a batch sample batch of images that was obtained by taking some images from the COCO competition of 2014 and cropping them to 640 x 480. 




## A note on limitations
Memory may be an issue on large images involving high PCA count. The reason for this is that even though the Incremental PCA is used to break the data into chunks and the call to the fitting method only operates over a single chunk, the chunk RAM usage may become prohibitively high. This is due to the fact that a bitmap-like representation is used when dealing with pixelated numerical image representation and from the fact that the chunk size must not be smaller than the PCA count. If one wanted for example, to have 1000 principal components of 1000 * 1000 images (having 3 colour channels) this would require a minimum chunk size of 24 GB (using the standard float64).
Seeing how the call of the fitting method is always done on a full chunk there is probably no easy way to overcome this hurdle other than writing out custom PCA.
