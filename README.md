<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->

[//]: # (</style><br />)
<div align="center">
  <a href="https://github.com/gihankaushyal/InternalTool">
    <img src="images/logo.png" alt="Logo" width="128" height="128">
  </a>

<h3 align="center">PixelAnomalyDetector</h3>

  <p align="center">
    A user-friendly graphical user interface (GUI) tool written in Python for analyzing and sorting *.cxi files in detector panels. This tool utilizes machine learning algorithms to quickly and accurately identify and isolate pixel anomalies in detector panel images (*.cxi files), making it easier to optimize performance and improve results. With an easy-to-use interface and powerful sorting capabilities, PixelAnomalyDetector streamlines the analysis process and saves valuable time for researchers. The tool also enables the user to train their own models and fine-tune the detection process, providing great flexibility. Whether you're working on a large dataset or a single image, PixelAnomalyDetector is the perfect tool for identifying and resolving pixel anomalies in detector panels.
    <br />
    <a href="https://github.com/gihankaushyal/PixelAnomalyDetector"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://gihankaushyal.github.io/PixelAnomalyDetector/">View Demo</a>
    ·
    <a href="https://github.com/gihankaushyal/PixelAnomalyDetector/issues">Report Bug</a>
    ·
    <a href="https://github.com/gihankaushyal/PixelAnomalyDetector/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![product-screenshot]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python]][Python-url]
* [![Sciket-learn][Scikit-learn]][Scikit-url]
* [![PyQt5][PyQt5]][PyQt5-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started


To get a local copy up and running follow these simple example steps.

### Prerequisites
[![PyPI - Python Version][Python-badge]][Python-url]

In order of all the dependencies and packages to run smoothly, make sure you have Python version v3.7 or higher by running.
* In Linux or Mac: run the following command on a Terminal 
    ```sh
    python --version
    ```
* in Window: run the following command on Windows command prompt or Windows Powershell
  ```sh
  python --version
  ```
  
You can either download the latest version of [Python][Python-url] or get an interactive 
development environment (IDE) such as [PyCharm][PyCharm-url]  or a Python distribution 
like [Anaconda][Anaconda-url].

Make sure to have the following python packages are installed;
    ``` h5py, numpy, pandas, matplotlib, seaborn, PyQt5, pyqtgraph, scikit-learn
    ```


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/gihankaushyal/PixelAnomalyDetector.git
   ```
2. Install dependency packages: 

   * on your Terminal or Command Prompt
      ```sh
      pip install package name
      ```
     
   <div style="text-align: center;">or</div>
   
   * Download and installed the dependency packages through the IDE
   
   
3. In your Terminal or Command Prompt go into the cloned ```PixelAnomalyDetector``` folder and type:
    ```sh
   cd /place/you/download/PixelAnoalyDetector
   ```
   ```sh
   python main.py
   ```
4. Click the following links to download a copies of practice data for [model training][model-training-url], 
data for the [model to be used][model-to-be-used-url]
and the [geometry][geom-url].
    * <b> Although not required, downloading the entire dataset for model training is strongly encouraged as it will greatly improve the model's predicting accuracy. If you elect to only download a subset of the training data we recommend selecting the <i>*-r0484_*.cxi</i> files for optimal results.</b>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


### 1. Displaying the HDF5 file
To open up the HDF5 files (*.h5, *.cxi) you need to point the PixelAnomalyDetector to where is you HDF5 file is 
located (1) and where is the appropriate geometry file is located (2). Then press ``` View File``` button (3).

![displayingTheCXIFile]

Some useful hits and guides are shown in the status bar (1) and on the bottom right conner (2) a solid green light 
indicating the GUI in idle or a blinking yellow light indicating the GUI is busy will be shown.

![statusBarAndLights]

Once you pressed the ```View File``` button a separate window will open with all the detector panel assembled
(similar with [cxiview][cxiview-url]) and one of the ASICs already been 
selected (1). Title of the widow show the event/image number showing out the total number of events 
available in the HDF5 file (2). Bottom left corner has a checkbox to turn on and off the found peaks (3). On the 
right-hand side of the window, you can change the brightness by changing the histogram (4).

![fileViewWindow]

### 2. Displaying the pixel intensity profile and toggle between images
once the HDF5 file is viewed ```Plot Pixel Intnstiy``` button (1) get enabled and when pressed, vertically average intensity 
profile (2) for the selected panel will be displayed. You can toggle between the images by pressing ```Next``` and 
``` Previous ``` buttons (3) or if you want to jump into a particular image you can enter the event number in the 
```Frame Number``` (4) field. When you are changing the event number on the main GUI the display for viewing the HDF5
file gets updated. 
 >- If you wish to plot the pixel intensity for a different detector panel, simply mouse click on the desired panel on 
 >the window for HDF5 file display and the pixel intensity plot on the main GUI will be automatically updated.

![plottingPixelIntensity]

### 3. Curve Fitting and Sort data for model training. 
Bottom left corner of the main GUI, clicked on the ```Plot a Fit``` checkbox (1) to a polynomial fit the pixel intensity 
plot (2). By default a 4<sup>th</sup> order polynomial will be applied, but you can change the order of the polynomial 
simply typing in the order in the ```Order of Fit``` field (4). Once you are satisfied with the fit of the polynomial, 
you can press on the ```Sort for ML``` button (4). A separate window will be opened up with the distributions for 
inflection points (5). On the Top part of the window, there will be suggested values for inflection points based on each 
distribution. (6) and thresholds for each inflection points (7). You can also type in the values for each section. Once 
you are satisfied with the sorting parameters click on the ```Sort``` button (8). You can now safely close this window.

>* In the process of training a machine learning model, it is important to ensure that the data used for training is 
> accurate and representative of the real-world scenario that the model will be used in. One way to achieve this is by 
> sorting the data into "good" and "bad" frames. "Good" frames are those that are accurate and representative of 
> the detector pixel intensities, while "bad" frames are with pixel gain switching issues.

>* One technique for sorting the frames is to use inflection points of an nth order polynomial. An inflection point is a 
> point on a graph where the concavity of the function changes. In other words, it is a point where the function changes
> from being concave up to concave down or vice versa. By analyzing the inflection points of a polynomial that fits the 
> data, it is possible to identify regions of the data that are likely to be "good" or "bad." This technique can be used
> to sort the frames and ensure that only the most accurate and representative data is used for training the model.

>* Additionally, this technique can be used to identify the outliers in the data, which can be removed before training 
> the model to improve its performance.

![sortingForML]

><div style="text-align: center; "> <b><u> **It is important to ensure that there are enough files available for training 
>the model. A general rule of thumb is to sort through at least ten (10) CXI files before proceeding with training the model. 
>This will ensure that the model has a sufficient amount of data to learn from and will increase the chances of achieving
>accurate results.** </u></b> </div>
<n> </n>

### 4. Training a model
Now that you have sorted data to train a machine learning model to be trained, click on the ``` Train a Model``` button.

![trainingAModel1]

A separate widow will appear with directions to train a model.
To train the model, follow these steps:

1. Click the "Browse" button to find and select the file that will be used for training. 
Currently, the files are saved in the InternalTool folder.
2. From the "Model Selection" dropdown menu, choose the machine learning model you prefer.
3. In the section for Training and Testing, enter the desired percentages for each. 

<b><i>Note: that the sum of these percentages should equal 100%. The default settings are 70% for Training and 30% for 
Testing. </i></b>
4. Finally, click the "Train" button to initiate the training process.
>* Training a model may take some time depending on the size of the training data set.

![trainingAModel2]

Once the model training is completed, the ```Test``` button (1) will get enabled for you do a model testing.
Have a look at the ```Confusion Matrix``` (2) and the ```Classification Report``` (3). 
 >* Confusion Matrix and Classification Report are essential evaluation metrics used to assess the quality and accuracy of
 > a machine learning model. Confusion matrix provides a summary of the model's predictions, comparing them to the 
 > actual class labels. It presents the number of correct and incorrect predictions for each class, allowing you to 
 > visualize the model's performance and identify areas of improvement. On the other hand, the classification report 
 > provides a more comprehensive evaluation of the model's performance by calculating various metrics such as precision, 
 > recall, F1-score, and support. These metrics give you an idea of the model's ability to accurately predict the 
 > class labels, how many true positives and false positives it has, and its overall performance. By using both 
 > the confusion matrix and the classification report, you can get a more comprehensive understanding of 
 > the model's performance and fine-tune it to improve its accuracy.

The off diagonal numbers (the false positives 
and false negative) should be close to Zero. In the Classification Report, have a look at ```Precision, Recall, F1-score
and Accuracy``` if the model is performing well, those numbers should be as close as possible to One. You can now safely
close this window.

![trainingAModel3]

To proceed, you have two options after reviewing the model quality matrices:

 1. Select ```Save``` (1) to keep the current model.
 2. Choose ```Rese``` (2) to start over and train a new model with a different algorithm.

![trainingAModel4]
### 5. Sorting the Data set using the trained model
Finally, you are here, Sorting the entire data set using the trained model. To being, click on the ```Sort Data``` 
button.

![sortData1]

A separate window will open with instructions. First, click on the ```Browes``` button (1) to point where in your folder 
with *.cxi files to be sorted. Then the space bellow ```Available *.cxi files in the folder``` will automatically
populate with all the available *.cxi file in folder you've selected. Lastly, clicked on the ```Sort``` button (3). 

![sortData2]

The 
tool will go through all the *.cxi files and sort "good events" from the "bad events" and save them in separate text 
files ([Crystfel][Crystfel-url] friendly) format. 

![sortData3]

See the [open issues](https://github.com/gihankaushyal/InternalTool/issues) for a full list of proposed features 
(and known issues).


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact


Gihan Ketawala - gihan.ketawala@asu.edu

Project Link: [https://github.com/gihankaushyal/PixelAnomalyDetector][repo-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* I acknowledge the guidance and support provided by [Sabine Botha][Sabine-github] in the development of 
the tool, PixelAnomalyDetector. Her contributions were crucial in ensuring the success of the project. 
Thank you, Sabine, for your dedication and expertise.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]:https://img.shields.io/github/contributors/gihankaushyal/InternalTool?style=for-the-badge
[contributors-url]: https://github.com/gihankaushyal/InternalTool/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/gihankaushyal/InternalTool.svg?style=for-the-badge
[forks-url]: https://github.com/gihankaushyal/InternalTool/network/members
[stars-shield]: https://img.shields.io/github/stars/gihankaushyal/InternalTool.svg?style=for-the-badge
[stars-url]: https://github.com/gihankaushyal/InternalTool/stargazers
[issues-shield]: https://img.shields.io/github/issues/gihankaushyal/InternalTool.svg?style=for-the-badge
[issues-url]: https://github.com/gihankaushyal/InternalTool/issues
[license-shield]: https://img.shields.io/github/license/gihankaushyal/InternalTool.svg?style=for-the-badge
[license-url]: https://github.com/gihankaushyal/InternalTool/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/gihan-ketawala/
[product-screenshot]: images/mainWindow.png
[displayingTheCXIFile]: images/diplayingTheCXIFile.png
[statusBarAndLights]: images/statusBarAndLigghts.png
[fileViewWindow]: images/fileViewWindow.png
[plottingPixelIntensity]: images/plotingPixelIntensity.png
[sortingForML]: images/sortingForML.png
[trainingAModel1]: images/trainingAModel-1.png
[trainingAModel2]: images/trainingAModel-2.png
[trainingAModel3]: images/trainingAModel-3.png
[trainingAModel4]: images/trainingAModel-4.png
[sortData1]: images/sortData-1.png
[sortData2]: images/sortData-2.png
[sortData3]: images/sortData-3.png
[Python-badge]: https://img.shields.io/pypi/pyversions/seaborn?style=for-the-badge
[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=yellow
[Python-url]: https://www.python.org/downloads/
[PyCharm-url]:https://www.jetbrains.com/pycharm/download/
[Anaconda-url]: https://www.anaconda.com
[Scikit-learn]: https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=61DAFB
[Scikit-url]: https://scikit-learn.org/
[PyQt5]: https://img.shields.io/badge/PyQt5-3776AB?style=for-the-badge&logo=pyqt5&logoColor=yellow
[PyQt5-url]: https://pypi.org/project/PyQt5
[model-training-url]:https://www.dropbox.com/scl/fo/391ged4bi7yqgwnshosm4/h?dl=0&rlkey=ygfmaw24kppcnh59ww9k17uh5
[model-to-be-used-url]:https://www.dropbox.com/scl/fo/g7bxy5mbp6vxcqkdqooke/h?dl=0&rlkey=3shte43a3k3tw3isss1a25ho5
[geom-url]:https://www.dropbox.com/scl/fo/x2b52tqpq8lpouxqbnb9g/h?dl=0&rlkey=r2ty8sijt0qtx9d0k2kt0sdgi
[cxiview-url]:https://www.desy.de/~barty/cheetah/Cheetah/cxiview.html
[Crystfel-url]:https://www.desy.de/~twhite/crystfel/index.html
[repo-url]:https://github.com/gihankaushyal/PixelAnomalyDetector
[Sabine-github]:https://github.com/sbotha89