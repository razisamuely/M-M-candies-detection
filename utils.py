import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def detect_color(pixels_list_hsv: list, color_limits_map: dict):
    """
    This function detects the color in a list of pixels using the given color limits in the HSV color space.
    The function counts the number of pixels that fall within the lower and upper bounds of each color defined in
    `color_limits_map` using the `cv2.inRange` function. The color with the maximum count of pixels within its bounds
    is then returned as the result.
    Parameters:
    pixels_list_hsv (list): A list of pixel values in the HSV color space.
    color_limits_map (dict): A dictionary that maps color names to their lower and upper bounds in the HSV color space.

    Returns:
    str: The name of the color with the maximum number of pixels within its bounds.
    """
    colors_counts = []

    # Per color, test the meximum occurrences which each pixle is falling
    for color in color_limits_map.keys():
        lower_bound = color_limits_map[color][0]
        upper_bound = color_limits_map[color][1]
        is_in = 0
        for j in pixels_list_hsv:
            is_in += cv2.inRange(src=j,
                                 lowerb=lower_bound,
                                 upperb=upper_bound)[0][0]
        colors_counts.append(is_in)

    # Pick color with maximum occurrences
    color = list(color_limits_map.keys())[np.argmax(colors_counts)]
    return color


def crop_circle_pixels(image: np.array, center: tuple, radius: int):
    """
    This function crop_circle_pixels takes an image and a center and radius as inputs,
    and returns the pixels within the specified circle in the image.
    The function first determines the shape of the input image and creates a mask array filled with zeros.
    The function then uses the cv2.circle function to draw a filled circle on the mask,
    with the specified center and radius. The relevant pixels within the circle are then cropped from the image
    by using the non-zero values in the mask as indices. The cropped pixels are returned as the result.

    Parameters:
    image (np.array): The input image.
    center (tuple): The center of the circle, given as a tuple of (x, y) coordinates.
    radius (int): The radius of the circle.

    Returns:
    np.array: The cropped pixels within the specified circle.
    """
    # Get image shape
    w0 = image.shape[0]
    h0 = image.shape[1]

    # Prepare Mask array
    mask = np.zeros(shape=(w0, h0),
                    dtype=np.uint8)

    # Draw filled circle mask
    cv2.circle(img=mask,
               center=center,
               radius=int(radius),
               color=(255, 255, 255),
               thickness=-1,
               lineType=8,
               shift=0)

    # Crop relevant pixels
    i = image[np.where(mask == 255)[0], np.where(mask == 255)[1], :]
    return i


def bgr_to_hsv(bgr_pixel_list: list):
    """
    This function converts a list of BGR pixels to HSV color space.

    Parameters:
    bgr_pixel_list (list): The list of BGR pixels to be converted.

    Returns:
    list: A list of HSV pixels.
    """
    return [cv2.cvtColor(j.reshape(1, 1, 3), cv2.COLOR_BGR2HSV) for j in bgr_pixel_list]


def draw_circles(img: np.array, center: tuple, radius: int, color: tuple, thickness: int):
    """
    This function draws a circle on an image.

    Parameters:
    img (np.array): The input image.
    center (tuple): The center of the circle, given as a tuple of (x, y) coordinates.
    radius (int): The radius of the circle.
    color (tuple): The color of the circle, given as a tuple of (B, G, R) values.
    thickness (int): The thickness of the circle edge.

    Returns:
    np.array: The input image with the drawn circle.
    """
    return cv2.circle(img=img, center=center, radius=radius, color=color, thickness=thickness)


def scatter_plot_radius_vs_color(df: pd.DataFrame):
    """
    This function creates a scatter plot of radius vs color.

    Parameters:
    df (pd.DataFrame): The input DataFrame with columns representing the colors and rows representing the radius values.

    Returns:
    None: The function displays the scatter plot using Matplotlib's plt.show().
    """
    vals, box_val, names, xs = [], [], [], []
    for i, col in enumerate(df.columns):
        box_val.append(df[col].dropna().values)
        vals.append(df[col].values)
        names.append(col)
        # adds jitter to the data points - can be adjusted
        xs.append(np.random.normal(i + 1, 0.04, df[col].values.shape[0]))

    plt.boxplot(box_val, labels=names)
    palette = ['red', 'orange', 'blue', 'green', 'yellow', 'brown']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.4, color=c)

    plt.ylim(30, 45)
    plt.title("Radius vs Color")
    plt.xlabel("Candy color")
    plt.ylabel("Radius (in general units)")
    plt.show()


def detect_circles(image: np.array, minDist: int = 20, param1: int = 50,
                   param2: int = 15, minRadius: int = 32, maxRadius: int = 42):
    """
    This function takes an image in BGR format and returns the circles detected in the image using Hough Circle Transform.
    Parameters:
    image (np.array): The image in BGR format.
    minDist (int): The minimum distance between the centers of the detected circles.
    param1 (int): The first parameter for the Hough Circle Transform.
    param2 (int): The second parameter for the Hough Circle Transform.
    minRadius (int): The minimum radius of the circles to be detected.
    maxRadius (int): The maximum radius of the circles to be detected.
    Returns:
    detected_circles (np.array): The detected circles in the image in the format [x, y, r],
    where (x, y) is the center of the circle and r is the radius of the circle.
    """

    # Convert the image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the grayscale image using a 3 x 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough Circle Transform on the blurred grayscale image.
    detected_circles = cv2.HoughCircles(image=gray_blurred, method=cv2.HOUGH_GRADIENT, dp=1, minDist=minDist,
                                        param1=param1,
                                        param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    return detected_circles


def circle_raiduses_dict_to_df(color_circel_radiuses: dict) -> pd.DataFrame:
    """
    Convert a dictionary of color-circles radiuses to a Pandas DataFrame.

    Parameters:
    color_circel_radiuses (dict): A dictionary with color names as keys and lists of radiuses as values.

    Returns:
    pd.DataFrame: A Pandas DataFrame with columns named after the color names and their respective radiuses in the rows.
    """
    return pd.DataFrame.from_dict(color_circel_radiuses, orient='index').T


def plot_bgr_image(image_bgr: np.array):
    """
    Plot a BGR image using Matplotlib.

    Parameters:
    image_bgr (np.array): BGR image to be plotted

    Returns:
    None
    """
    # Turn off axis display.
    plt.axis("off")

    # Show the BGR image using Matplotlib.
    # The order of color channels needs to be reversed for correct display using Matplotlib.
    plt.imshow(image_bgr[:, :, ::-1])

    # Display the plot.
    plt.show()


def showimage(myimage: np.array, color: str, std: float):
    if (myimage.ndim > 2):
        myimage = myimage[:, :, ::-1]
    fig, ax = plt.subplots(figsize=[10, 10])
    ax.imshow(myimage, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])
    plt.title(f"Filtering {color} by limiting {std:.1f} stds from mean of {color} distribution")
    plt.show()


def stuck_image_pixels(images_dir_path: str):
    joined_images = None
    for i in [i for i in os.listdir(images_dir_path) if 'png' in i]:
        image = cv2.imread(images_dir_path + i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if joined_images is None:
            joined_images = image.reshape(image.shape[0] * image.shape[1], 3)
        else:
            im = image.reshape(image.shape[0] * image.shape[1], 3)
            joined_images = np.concatenate([joined_images, im], axis=0)

        return joined_images
