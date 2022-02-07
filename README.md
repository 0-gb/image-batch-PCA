# Multiple image PCA tutorial
This is a tutorial on how to do incorporate several images into the PCA transform.  The IncrementalPCA. The images are not all processed at once but instead, they are separated into chunks (batches of images considered at once), of which, all but the last have the size equal to the number of components, while the last one is approprietly larger to accoomodate for the leftover (max: 2x number of components - 1). At the start, a GUI opens up where the user provides:
 - the input folder which contains the images
 - the image suffix (e.g.: jpg, png, etc. )
 - the output folder where the model as where as the image data will be stored
 - the number of components to which the data dimensionality will be reduced
## Overview 
Basic checks are performed over the input like whether the input directory exists, whether the images are of the same sizes (it makes no sense in bringing them to the same PCA transform matrix otherwise) and whether the PCA count is not too large for the image count in the provided directory. After providing the required data, the user can start the PCA for the selected images. 
For a large collection of images the entire process may take a long time so there is also some printed output to the console that helps the user predict when the process will be finished - the provided output at the console is the time and past chunk count overview.
## A note on limitations
Memory may be an issue on large images involving high PCA count. The reason for this is that even though the Incremental PCA is used to break the data into chunks and the call to the fitting method only operates over a single chunk, the chunk RAM usage may become prohibitevly high. This is due to the fact that a bitmap-like representation is used when dealing with pixelated numerical image representation and from the fact that the chunk size must not be smaller than the PAC count. If one wanted for example, to have 1000 principal components of 1000 * 1000 images (having 3 colour channels) this would require a minimum chunk size of 24 GB (using the standard float64).
Seeing how the call of the fitting method is always done on a full chunk there is probably no easy way to overcome this hurdle other than writing out custom PCA. 
