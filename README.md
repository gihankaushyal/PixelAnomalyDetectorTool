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
    A user-friendly graphical user interface (GUI) tool written in Python for analyzing and sorting *.cxi files in detector panels. This tool utilizes machine learning algorithms to quickly and accurately identify and isolate pixel anomalies in detector panel images (*.cxi files), making it easier to optimize performance and improve results. With an easy-to-use interface and powerful sorting capabilities, PixelAnomalyDetector streamlines the analysis process and saves valuable time for researchers. The tool also enables the user to train their models and fine-tune the detection process, providing great flexibility. Whether you're working on a large dataset or a single image, PixelAnomalyDetector is the perfect tool for identifying and resolving pixel anomalies in detector panels.
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

In order for all the dependencies and packages to run smoothly, make sure you have Python version v3.8 or higher by running.
* In Linux or Mac: run the following command on a Terminal 
    ```sh
    python --version
    ```
* in Windows: run the following command on Windows command prompt or Windows Powershell
  ```sh
  python --version
  ```
  
You can either download the latest version of [Python][Python-url] or get an interactive 
development environment (IDE) such as [PyCharm][PyCharm-url]  or a Python distribution 
like [Anaconda][Anaconda-url].

Make sure to have the following python packages are installed;
    ``` h5py, numpy, pandas, matplotlib, seaborn, PyQt5, pyqtgraph, scikit-learn, tqdm, plotly, psutil
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
```Download and install the dependency packages through the IDE```
   
   
3. In your Terminal or Command Prompt go into the cloned ```PixelAnomalyDetector``` folder and type:
    ```sh
   cd /place/you/download/PixelAnoalyDetector
   ```
   ```sh
   python main.py
   ```
4. Click the following links to download a copy of practice data for [model training][model-training-url], 
data for the [model to be used][model-to-be-used-url]
and the [geometry][geom-url].
    * <b> Although not required, downloading the entire dataset for model training is strongly encouraged as it will greatly improve the model's predicting accuracy. If you elect to only download a subset of the training data we recommend selecting the <i>*-r0484_*.cxi</i> files for optimal results.</b>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


### 1. Displaying the HDF5 file
To display HDF5 files (*.h5, *.cxi), direct the PixelAnomalyDetectorTool to the storage location of your HDF5 file/s `(1)` and the corresponding geometry file `(2)`. Initiate the display by clicking the View File Button `(3)`. If you need to start over, the Reset Button `(4)` clears the current display. For those with a pre-trained model, load it using the Load Model Button `(5)` to apply your custom analyses.

![displayingTheHDF5File] 

To visualize HDF5 files (*.h5, .cxi) from a compiled list (.list), first, navigate to the location of your *.list file `(1)` and select the required detector description file `(2)`. This action populates the dropdown menu `(3)` with all the available files listed in the *.list file. To view the files, simply click the View Files Button `(4)`.

![displayingAListOfHDF5Files]


  >Helpful hints and instructions are displayed within the status bar `(1)` at the bottom of the GUI. Additionally, the GUI's current state is indicated by a light in the bottom right corner: a solid green light `(2)` signifies that the GUI is idle, while a blinking yellow light denotes that the GUI is processing or busy.


![statusBarAndLights]

When you click the `View File button`, a new window will open, displaying all the detector panels assembled, with a layout similar to [cxiview][cxiview-url]. One of the ASICs will be pre-selected (1). The window's title bar displays the current event/image number and the total count of events/images in the HDF5 file (2). To the right, you have a histogram to adjust image brightness (3). The bottom left corner features checkboxes to toggle the visibility of peaks detected (4) and to apply the current histogram settings to all images in the file. Lastly, in the bottom right corner, radio buttons allow the classification of the image as 'Good', 'Bad', or 'Ignore' for machine learning purposes.



![fileViewWindow]

### 2. Displaying the pixel intensity profile and toggle between images
Upon clicking the `View File` button `(1)`, the GUI presents a vertical average intensity profile `(2)` for the selected panel, accompanied by a default 4th order polynomial fit. Navigate through images with the `Next` and `Previous` buttons `(3)` or go directly to a specific image by inputting its event number in the `Frame Number` field `(4)`. As you alter the event number in the main GUI, the HDF5 file viewer will refresh accordingly.

 >* To analyze a different detector panel, click on the desired panel within the HDF5 file viewer `(5)`. The main GUI will then automatically update to display the new panel's pixel intensity plot.

![plottingPixelIntensity]

### 3. Curve Fitting and Sort data for model training.

In the main GUI's bottom left corner, toggle the polynomial fit for the pixel intensity plot by clicking the Plot a `Fit checkbox` `(1)`. A 4th-order polynomial is applied by default `(2)`. To adjust the polynomial's order, simply enter a new value in the Order of Fit field `(3)`. After fine-tuning the fit, click the `Label Data` button `(4)` to proceed.

A new window will open, displaying the distributions for inflection points `(5)`. At the window's top, you'll find suggested values for the inflection points `(6)` and adjustable thresholds for each `(7)`. You're free to enter custom values for each parameter as needed. Finalize the sorting parameters by pressing the `Sort button` `(8)` once you're content with the setup.

>* For the efficacy of a machine learning model, particularly in pixel anomaly detection, it is imperative that the training data is both accurate and reflective of the actual conditions in which the model will operate. This necessitates a meticulous sorting of the data into 'good' and 'bad' frames. 'Good' frames are defined as those with precise and reliable measurements of detector pixel intensities, free from anomalies. Conversely, 'bad' frames are identified by issues such as pixel gain switching, which can distort the model's learning process. By segregating the data thus, the integrity and representativeness of the training set are preserved, laying the foundation for a robust and reliable machine learning model.

>* Sorting frames effectively for machine learning model training can be accomplished through the analysis of inflection points on an nth order polynomial fit to the data. Inflection points are specific locations on a curve where the curvature switches direction, marking a transition from concave upward to concave downward, or the reverse. This characteristic is instrumental in distinguishing between 'good' and 'bad' regions within the data set. By fitting a polynomial to the pixel intensity data and examining its inflection points, we can discern the most reliable and representative frames for inclusion in the training set. Utilizing this technique ensures the machine learning model is trained on high-quality data, enhancing its predictive accuracy in real-world applications.

>* Furthermore, the inflection point analysis within an nth order polynomial fit offers a robust mechanism for outlier detection. Outliers are atypical data points that deviate significantly from the overall pattern and can adversely affect the model's learning trajectory. By pinpointing these anomalies prior to training, they can be selectively excluded, thereby enhancing the model's ability to generalize from the training data. This pre-processing step is crucial for improving the performance and accuracy of the machine learning model, ensuring that it is fine-tuned to reflect the true nature of the dataset it represents.

![dataLabeling]

><div style="text-align: center; "> <b><u> **To ensure the robustness of the machine learning model, it's crucial to provide it with an ample training dataset. A recommended minimum is to curate and sort through at least ten (10) CXI files before commencing the training phase. This volume of data helps to establish a comprehensive learning foundation for the model, thereby enhancing the likelihood of obtaining precise and reliable outcomes. Adequate data quantity is a key factor in the model's ability to accurately generalize and perform well in practical scenarios.** </u></b> </div>
<n> </n>

### 4. Training a model
With the data now sorted and ready, initiate the machine learning model's training by clicking the `Train a Model` button.

![trainingAModel1]
A new window will guide you through the model training process. Follow these instructions:

1. Use the "Browse" button to locate and select your training file, which is typically stored in the InternalTool folder.
2. Choose your preferred machine learning model from the "Model Selection" dropdown menu.
3. Allocate the percentages for Training and Testing in their respective sections. \
<b><i>Note: that the sum of these percentages should equal 100%. The default settings are 70% for training and 30% for testing. </i></b>
4. Finally, click the "Train" button to initiate the training process.
>* Training a model may take some time depending on the size of the training data set.

![trainingAModel2]

After the completion of model training, the `Test button` `(1)` will become active, enabling you to conduct model testing. Pay close attention to the `Confusion Matrix` `(2)` and the `Classification Report` `(3)`:

>In the Confusion Matrix, aim for the off-diagonal numbers (representing false positives and false negatives) to be as close to zero as possible.\
>In the Classification Report, evaluate the Precision, Recall, F1-score, and Accuracy metrics. Ideally, these values should approach one, indicating high model performance.


![trainingAModel3]

To proceed, you have two options after reviewing the model quality matrices:

 1. Select `Save` `(1)` to keep the current model.
 2. Choose `Reset` `(2)` to start over and train a new model with a different algorithm.

![trainingAModel4]
### 5. Sorting the Data set using the trained model
You've reached the final step: Sorting the entire dataset using the trained model. To start, click on the `Sort Data` button `(1)`. Alternatively, if you have a pre-trained model, you can load it using the `Load Data` button `(2)`.

![sortData1]

Once you initiate the process, a separate window will appear, providing further instructions. Follow these steps to proceed:

1. Click on the Browse button (1) to navigate to the folder containing your HDF5 files.
2. The area under Available HDF5 files in the folder will automatically populate with all the HDF5 files present in the selected folder.
3. To begin sorting, click on the Sort button (3).

![sortData2]

The tool efficiently processes all the *.cxi files, segregating 'good events' from 'bad events'. It then saves these sorted events into separate text files, formatted to be compatible with [Crystfel][Crystfel-url].

![sortData3]

See the [open issues](https://github.com/gihankaushyal/InternalTool/issues) for a full list of proposed features 
(and known issues).


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

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
[displayingTheHDF5File]: images/displayingTheHDF5File.png
[displayingAListOfHDF5Files]:images/displayingAListOfHDF5Files.png
[statusBarAndLights]: images/statusBarAndLights.png
[fileViewWindow]: images/fileViewWindow.png
[plottingPixelIntensity]: images/plottingPixelIntensity.png
[dataLabeling]: images/dataLabeling.png
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