# Air Quality Machine Learning Experiments

A collection of machine learning experiments to investigate the feasbility of accurate air quality modeling using machine learning and neural network techniques.

### Background

Air quality modeling today is most commonly and effectively done using computational Eularian (fixed grid), Lagrangian (moving frame of reference), and Gaussian (puff-plume) models. Popular Eularian computational models include CAMx and CMAQ, a common Lagrangian model is CALPUFF, and a common Gaussian model is SCICHEM. These models, along with others not mentioned here, are state of the art with regard to both computational complexity and accuracy. 

In recent years, the field of machine learning has grown substantially as an alternative method to conduct predictive modeling. Common uses for machine learning today include self-driving cars, user recommendations on websites, and fraud detection. While these applications deploy mathematically and statistically advanced techniques in modeling, their inputs are often somewhat linear in nature. One of the greatest challenges in developing machine learning models for air quality will be to overcome the chaotic and unstable nature of the atmosphere as a combination of fluid dynamics, radiation, and thermodynamics, couple with often-unpredictable emission releases and nonlinear relations between atmospheric species exist. Because of this, it may very well turn out to be impossible to replicate the accuracy of Eularian models that have parameratized the governing equations of atmospheric science and chemistry with a stand alone machine learning model.


* [Air Pollution Modeling] - Overview  on Computational Air Quality Modeling

[Air Pollution Modeling]: <http://home.iitk.ac.in/~anubha/Modeling.pdf>
