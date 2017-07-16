import numpy as np
import warnings
from sklearn.cluster import KMeans
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import img_as_ubyte
from scipy.sparse import coo_matrix

def handle_out_of_borders(img, slice_x, slice_y):
    """handle_out_of_borders
    
    This function get a pair of slices and check if both are within
    the bounds of an image dimensions. If a slice is partially outside the
    image boundaries it will be moved until it's border touch that of the
    image.
    
    If some dimension of the slice is greater than the corresponding dimensions
    on the image, a ValueError Exception will be thrown.
    
    Parameters:
    - img: Numpy array containing the image. Can be 2 or 3 dimensions
    - slice_x: Python slice object containing the x coordinate (Width) slice
    - slice_y: Python slice object containing the y coordinate (height) slice
    
    Returns: Adjusted (slice_x, slice_y) tuple.    
    """
    start_x, end_x = (slice_x.start, slice_x.stop)
    start_y, end_y = (slice_y.start, slice_y.stop)

    img_height = img.shape[0]
    img_width = img.shape[1]
    
    if start_x < 0:
        end_x = end_x + abs(start_x)
        start_x = 0
    if start_y < 0:
        end_y = end_y + abs(start_y)
        start_y = 0
    if end_x >= img_width:
        start_x = start_x - (end_x - img_width + 1)
        end_x = img_width -1
    if end_y >= img_height:
        start_y = start_y - (end_y - img_height)
        end_y = img_height
    
    if start_x < 0 or end_x > img_width:
        raise ValueError("Width of slice is greater than width of image")
    
    if start_y < 0 or end_y > img_height:
        raise ValueError("Height of slice is greater than height of image")   
        
    return slice(start_x, end_x), slice(start_y, end_y)
    

def get_slice(img,centroid,size):
    """get_slice
    
    This function define a slice bounded by the image centered over a coordinate
    (Centroid). If the exact centroid center makes the slice go outside the image,
    it will be adjusted to fit. If the slice cannot be fit inside the image, a
    ValueError exception will be thrown.
    
    Parameters:
    - img: Numpy array containing the image. Can be 2 or 3 dimensions.
    - Centroid: (y, x) tuple containing the centroid coordinates. The usual
    coordinate order is inverted to conform with matplotlib conventions.
    - size: (height, width) tuple containing the slice size
    
    """
    slice_height, slice_width = size

    start_x = int(centroid[1] - slice_width / 2)
    start_y = int(centroid[0] - slice_height / 2)
    
    end_x = start_x + slice_width
    end_y = start_y + slice_height  
    
    slice_x = slice(start_x, end_x)
    slice_y = slice(start_y, end_y)
    
    slice_x, slice_y = handle_out_of_borders(img, slice_x, slice_y)

    return img[slice_y, slice_x]
    
def matrix2tuples(matrix):
    """matrix2tuples
    
    Convert a dense binary matrix representation into a sparte coordinate format
    list of tuples for input on clustering algorithms.
    
    Parameters: 
    
    - matrix: A 2-D numpy array
    
    Returns: (non-zero-entries, 2) shaped numpy array.
    
    """
    coo = coo_matrix(matrix)
    tuples = np.zeros((coo.row.shape[0],2),np.uint32)
    tuples[:,0] = coo.row
    tuples[:,1] = coo.col
    
    return tuples

def get_slice_shape(img, size, n_clusters):
    """get_slice_shape
    
    Get the slice shape, conditioned on image format (Grayscale vs. RGB)
    
    Parameters:
    
    - img: Numpy array containing the image
    - size: Tuples (height, width) containing the slice size
    - n_clusters: The number os slices
    
    Returns: (n_clusters, height, width, color_channels) tuple        
    """
    if len(img.shape) == 3:
        return (n_clusters,) + size + (img.shape[2],)
    elif len(img.shape) == 2:
        return (n_clusters,) + size 
    else:
        raise ValueError("img argument doesn't seem to be a valid image")

def get_gray_image(img):
    """get_gray_image
    
    Ensure consistency over the gray image format to be that of a flot
    image. This is due to the inconsistent behaviour of the rgb2gray
    sklearn function.
    
    This function, when applied    
    """
    gray_image = rgb2gray(img)
    if gray_image.dtype != np.uint8:
        #We have to catch the warnings to avoid the precision ones
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    
            gray_image = img_as_ubyte(gray_image)
    return gray_image
        
def generate_slices(img, size, n_clusters=3):
    """generate_slices
    
    Generate slices based on the edge clustering pipeline describe on the 
    exploration notebook.
    
    Parameters:
    
    - img: Image we want the slices to be based on
    - size: (height, width) tuple containing the desired slices dimensions
    - n_clusters: Number of clusters we want the slices to be based upon
    
    Returns: (slices, centroids)
        - slices: (n_clusters, height, width, color_channels) numpy array containing
                  the slices.
        - centroids: (n_centroids, 2) numpy array containing the centroids obtained.
    """
    gray_img = get_gray_image(img)
    edges = canny(gray_img,sigma=3)
    tuples = matrix2tuples(edges)
    
    model = KMeans(n_clusters=n_clusters)
    model.fit(tuples)
    centroids = model.cluster_centers_        

    ret = np.zeros(get_slice_shape(gray_img, size, n_clusters), np.uint8)    
    
    for idx, centroid in enumerate(centroids):        
        ret[idx] = get_slice(gray_img, centroid,size=(512,512))        

    return ret, centroids
            