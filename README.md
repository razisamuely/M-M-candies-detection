

# M&M candies detection 


<p align="center">
    <img src="https://media.tenor.com/wLK6MnXvrtQAAAAC/m-and-ms-mm.gif"  width="330" height="200">
</p>



The goal of this project is to detect and quantify the number of `M&M` candies, 
as well as their radius, per color in a given image. 

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/mnm_image.png"  width="200" height="150">
</p>



To accomplish this, the project employs traditional 
and basic feature extraction techniques, as well as simple rules.
The project hence some general assumption:
1. The distance from all candies to the camera lens is equal.
2. The distance from all candies is similar (even though there some that a bit closer, on top of the pile)
3. There are only 6 colors (blue, green, brown, blue, red, orange)
4. The project uses discrete general distance units and therefore the radius value of 46 
does not represent any specific unit of measurement in reality.

It is important to emphasize that there is significant potential 
for further research and a deeper examination of each approach, 
which goes beyond the scope of this project.

However, after presenting the final results and discussing 
the strengths and weaknesses of the chosen pipeline, 
the project will also examine alternative basic approaches to accomplish the same task.

## Technical Details
1. The main code can be found within the `main.ipynb` Jupyter Notebook.
2. The functionality is separated and can be found within `utils.py`.
3. The data, including main image, pixels samples per color and gifs, can be found within the `data` dir.

## Traditional approaches
To utilize traditional techniques, we must take advantage of certain properties of the objects in the image, 
such as: distinct color differences, well-defined shapes (such as circles), and clear edges. 
When using predefined features, we have a prior understanding of what functions need to be estimated, 
as opposed to advanced learning techniques where the model has to learn these features itself.

### HoughCircles & Color distribution
In this approach, we will take advantage of the fact that `M&M` candies have a circular shape, 
allowing us to use the `Circle Hough Transform`, a fundamental feature extraction technique ([link](https://en.wikipedia.org/wiki/Circle_Hough_Transform)).
Moreover, we will also make use of the distinct color distribution of `M&M` candies.

The steps involved are:

1. Detect circles using `cv2.HoughCircles` in regard to radius boundaries and sensitivity parameters.
2. Crop the pixels within each circle once the circles have been detected.
3. Identify the color of each pixel by matching it with predefined color intervals.
4. Determine the color of each circle by analyzing the frequency of the cropped pixels and their corresponding colors.

# Final results 

### Candies detection and color classification
<p align="center">
    <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/HoughCircles_gif.gif"  width="300" height="230">
</p>

### Radius vs Color plot 
<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/HoughCircles_radius_vs_color_scatter_box_plot.png"  width="400" height="300">
</p>


# Process 

### Detect Candy area  
The `Circle Hough Transform` works as follows:

1. Edge detection is performed using an edge detector such as the `Canny edge detector`. 
2. Circles are drawn on the detected edges with predefined radii.
3. Centers for the circles are selected where the highlights will appear.

<p align="center">
    <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/CircleHough_Transform.gif"  width="300" height="230">
</p>

### Classify candy color
The process for determining the color of a circle is based on the proportion of 
pixels with colors within predefined upper and lower color bounds. 
The method for choosing these bounds is implemented here [`find_color_limits.ipynb`](https://github.com/razisamuely/MnM-candies-detection/blob/main/finde_color_limits.ipynb) 
In essence, for each color, pixels were manually sampled and cropped 
(these samples can be found in the `data` directory). 
Then, the mean and standard deviation were calculated and a series of 
confidence intervals were used until satisfactory results were achieved. 
The following animation illustrates the process, starting with strict limits where orange is barely visible and 
gradually moving towards more permissive limits where orange is brightly displayed, until yellow starts to become prominent:

<p align="center">
    <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/2023-02-08%2018.23.12.gif"  width="450" height="170">
</p>



## Performance discussion 

#### Plot
1. It's easy to observe that the radius distribution is limited between `[33,41]`. 
This is due to the use of radius thresholds to minimize false negatives.
2. The mean radius of all colors is approximately 40 +-1, which is what we expected, 
as the primary factor that distinguishes the candies is their color.


#### Detection

The advantage of using this technique is that it does not require a separate training phase, 
making it faster compared to other methods.

However, there is a clear drawback to this method:
1. `Circle Hough Transform` may not accurately detect candies that are positioned in a way that breaks the circular shape,
as indicated by the blue arrow and box in the image.

<p align="center">
    <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/HoughCircles_miss.png"  width="200" height="170">
</p>


2. Furthermore, the color classification approach that is based on color frequency may also produce inaccurate results. 
This is illustrated in the image where a yellow candy is mistakenly detected as brown.


<p align="center">
    <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/false_color_detection.png"  width="200" height="170">
</p>

It's important to keep in mind that this technique is not very reliable and can easily 
be impacted by even slight changes such as different lighting conditions, candy positions, 
or even similar but not perfectly rounded M&M candy types 
(as seen in 
the example at [these](https://cdn.shopify.com/s/files/1/0613/7842/9113/products/51KArsUJoqL_1024x1024.jpg?v=1644154577)).


##### Further research
Additionally,for further research there are steps that can be taken to improve the performance 
of this technique. For instance: 
1. Optimizing the parameters of `cv2.HoughCircles`
2. Defining more sophisticated color classification rules that take into account cutting and merging of areas and pixel color proportion.
3. Using box detection after cropping for cases which we want to average the radius for not complete circle candies position






### Color Edge detection & contours
Of-curse there is another ways for retrieving the same target without the need of training process.
For example:
1) Using color limit for extracting pixels
2) Implement edge detection in order to mark the border between candies 
3) Apply contours per color.


Even the project isn't implementing it end2end I thought it would be nice to show present images which describes the flow.

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/color_edged_cotours.png"  width="360" height="200">
</p>



# Advanced deep object detection technique
Due to the limitations of having only one data point and the time constraints associated with tagging and training, 
the use of advanced deep object detection techniques is not highly recommended for this project. 
Despite this, I was curious about the potential results that could be achieved by using *transfer learning*
and *overfitting* a model specifically for the M&M use case.

The following steps were taken:
1. Labeling into 6 categories (one for each color).

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/LabelImage.png"  width="250" height="150">
</p>

2. Augmenting the single labeled image (including flipping, cropping, adding noise etc.)

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MnM-candies-detection/main/data/Augmentation.png"  width="300" height="300">
</p>

3. Selecting a lightweight model, as `ssd_mobilenet_v2_fpnlite_320x320`, that could be efficiently trained on a local machine.
4. Utilizing transfer learning by using a pre-trained model, `coco17`, which includes a diverse range of colors, as the starting point. (Note that other datasets could potentially be more preferable).



In conclusion, the results were not satisfactory, likely due to the limitations mentioned above. 
Nonetheless, the code for this exploration can be found at the following [link](https://github.com/razisamuely/hens)  
and can serve as a basis for further research.


# Summary 
In short, traditional computer vision object detection methods require domain knowledge and are usually efficient in 
terms of computational resources, but they have difficulty detecting complex objects and adapting to changes in appearance
For the specific task of detecting the color and radius of M&Ms, the `Circle Hough Transform` combined with `color 
classification based limits` approach is a more advantageous solution in terms of simplicity, 
speed, and resource consumption when compared to more advanced deep learning techniques
and even other traditional techniques.

## Note
Please note that this is a first and short iteration, and its main goal is to prove the concept. 
Given this, there is a lot of room for improvement and further research. 
Some things that could be added  include: 
- Parameters search for optimizing performance.
- Showing Comparison with the other mentioned and additional techniques.
- Showing results on similar images.
- Presenting results statistics as confidence intervals and percentiles.
- Optimizing the parameters of `cv2.HoughCircles`
- Defining more sophisticated color classification rules that take into account cutting and merging of areas and pixel color proportion.
- Using box detection after cropping for cases which we want to average the radius for not complete circle candies position



## Code Structure
Please note that the code for this project is not written in classes as is typically done. 
Instead, the code is organized in a procedural manner for simplicity and ease of understanding.




<p align="center">
  <img src="https://i.giphy.com/media/l03xNVz3HEcL76r5uq/giphy.webp"  width="300" height="150">
</p>

