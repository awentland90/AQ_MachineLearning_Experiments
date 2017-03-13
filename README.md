# Air Quality Machine Learning Experiments

A collection of machine learning experiments to investigate the feasbility of fast and accurate air quality modeling using machine learning and neural network techniques.

### Background

Air quality modeling today is most commonly and effectively done using computational Eularian (fixed grid), Lagrangian (moving frame of reference), and Gaussian (puff-plume) models. Popular Eularian computational models include [CAMx] and [CMAQ] while common Lagrangian and Gaussian models include [CALPUFF] and [SCICHEM], respectively. These models, along with many others not mentioned here, are state of the art with regard to both their computational complexity and accuracy. 

In recent years, the field of machine learning has grown substantially as a new statistical method to conduct predictive modeling. Common uses for machine learning today include self-driving cars, user recommendations on websites, and fraud detection. While these applications deploy mathematically and statistically advanced techniques in modeling, their inputs are often somewhat linear in nature. One of the greatest challenges in developing machine learning models for air quality will be to overcome the chaotic and unstable nature of the atmosphere as a combination of dynamics, radiation, and thermodynamics, coupled with often-unpredictable emission releases and nonlinear relations between atmospheric species exist making basic linear machine learning approaches substandard to computational models. It may very well turn out to be impossible to replicate the accuracy of Eularian models that have parameratized the governing equations of atmospheric science and chemistry with a stand alone machine learning model (here's to throwing all caution to the wind).


* [Air Pollution Modeling] - Overview  on Computational Air Quality Modeling
* [Machine Learning for Software Engineers] - Guide to Applied Machine Learning

[Air Pollution Modeling]: <http://home.iitk.ac.in/~anubha/Modeling.pdf>
[CAMx]: <http://www.camx.com/about/default.aspx>
[CMAQ]: <https://www.cmascenter.org/cmaq/>
[CALPUFF]: <http://www.src.com/>
[SCICHEM]: <http://download.ramboll-environ.com/environcorp/SCICHEM%20Air%20Quality%20Model.pdf>
[Machine Learning for Software Engineers]: <https://github.com/ZuzooVn/machine-learning-for-software-engineers>

### Current Experiments

| Directory | Description |
| ------ | ------ |
| BasicAlgorithms | Testing simple machine learning algorithms |

