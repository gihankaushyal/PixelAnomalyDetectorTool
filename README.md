<a name="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/gihankaushyal/InternalTool">
    <img src="images/logo.png" alt="Logo" width="128" height="128">
  </a>

<h3 align="center">PixelAnomalyDetector</h3>

  <p align="center">
    A user-friendly graphical user interface (GUI) tool written in Python for analyzing and sorting *.cxi files in detector panels. This tool utilizes machine learning algorithms to quickly and accurately identify and isolate pixel anomalies in detector panel images (*.cxi files), making it easier to optimize performance and improve results. With an easy-to-use interface and powerful sorting capabilities, PixelAnomalyDetector streamlines the analysis process and saves valuable time for researchers. The tool also enables the user to train their own models and fine-tune the detection process, providing great flexibility. Whether you're working on a large dataset or a single image, PixelAnomalyDetector is the perfect tool for identifying and resolving pixel anomalies in detector panels.
    <br />
    <a href="https://github.com/gihankaushyal/InternalTool"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/gihankaushyal/InternalTool">View Demo</a>
    ·
    <a href="https://github.com/gihankaushyal/InternalTool/issues">Report Bug</a>
    ·
    <a href="https://github.com/gihankaushyal/InternalTool/issues">Request Feature</a>
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
    <li><a href="#roadmap">Roadmap</a></li>
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

This software was written in Python v3.10. Therefore, make sure you have Python version v3.10 or higher by running.
* python v3.10
  ```sh
  pip install --upgrade
  ```
Make sure you have the following python packages installed;
    ``` h5py, numpy, pandas, matplotlib, seaborn, PyQt5, pyqtgraph, scikit-learn
    ```


### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/gihankaushual/InternalTool.git
   ```
2. Install dependency packages
   ```sh
   pip install <package name>
   ```
3. In your terminal go into the cloned InternalTool folder and type
   ```sh
   python main.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### 1. Displaying the *CXI file
To open up the cxi file you need to point the PixelAnomalyDetector to where is you *.cxi file is located (1) and where 
is the appropriate geometry file is located (2). Then press ``` View File``` button (3)

![displayingTheCXIFile]

Once you pressed the ```View File``` button a separate window will open with all the detector panel assembled
(similar with [cxiview](https://www.desy.de/~barty/cheetah/Cheetah/cxiview.html)) and one of the ASCIIs already been 
selected (1). Title of the widow show the event/image number showing out the total number of events 
available in the *cxi file (2). Bottom left corner has a checkbox to turn on and off the found peaks (3). On the right 
hand side of the window, you can change the brightness by changing the histogram (4).

![fileViweWindow]

### 2. Displaying the pixel intensity profile and toggle between images
once the cxi is viewed ```Plot Pixel Intnstiy``` button (1) get enabled and once pressed vertically average intensity 
profile (2) for the selected panel will be displayed. You can toggle between the images by pressing ```Next``` and 
``` Previous ``` buttons (3) or if you want to jump into a particular image you can enter the event number in the 
```Frame Number``` (4) field. When you are changing the event number on the main GUI the display for viewing the *.cxi
file get updated. 
 - If you wish to plot the pixel intensity for a different detector panel, simply mouse click on the desired panel on 
the window for *.cxi file display and the pixel intensity plot on the main GUI will be automatically updated.

![plottingPixelIntensity]

### 3. Curve Fitting and Sort data for model training. 
Bottom left corner of the main GUI, clicked on the ```Plot a Fit``` checkbox (1) to a polynomial fit the pixel intensity 
plot (2). By default a 4<sup>th</sup> order polynomial will be applied, but you can change the order of the polynomial 
simply typing in the order in the ```Order of Fit``` field (4). Once you are satisfied with the fit of the polynomial, 
you can press on the ```Sort for ML``` button (4). A separate window will be opened up with the distributions for 
inflection points (5). On the Top part of the window, there will be suggested values for inflection points based on each 
distribution. (6) and thresholds for each inflection points (7). You can also type in the values for each section. Once 
you are satisfied with the sorting parameters click on the ```Sort``` button (8).

![sortingForML]
_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/gihankaushyal/InternalTool/issues) for a full list of proposed features (and known issues).

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

Project Link: [https://github.com/gihankaushyal/InternalTool](https://github.com/gihankaushyal/InternalTool)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* []()
* []()
* []()

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
[fileViweWindow]: images/fileViewWindow.png
[plottingPixelIntensity]: images/plotingPixelIntensity.png
[sortingForML]: images/sortingForML.png
[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=yellow
[Python-url]: https://python.org/
[Scikit-learn]: https://img.shields.io/badge/scikitlearn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=61DAFB
[Scikit-url]: https://scikit-learn.org/
[PyQt5]: https://img.shields.io/badge/PyQt5-3776AB?style=for-the-badge&logo=pyqt5&logoColor=yellow
[PyQt5-url]: https://pypi.org/project/PyQt5
