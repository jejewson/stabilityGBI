# stabilityGBI

stabilityGBI contains the *R* scripts and *stan* software to reproduce the experiments from the paper "On the Stability of General Bayesian Inference" J. Jewson, J.Q.Smith and C. Holmes (2023). The article and supplementary material can be found at **.

The repository contains the following .Rmd files:
+ `stability_Gaussian_Student.Rmd` which contains the code to reproduce Figures 1, 2, 3, B.1, B.2 and B.3 of the paper 
+ `stability_regression.Rmd` which contains the code to reproduce the experiements of Sections 6.1.1 and B.1.4 of the paper 
+ `stability_mixture_modelling.Rmd` which contains the code to reproduce the experiements of Section 6.2 of the paper
+ `stability_binaryClassification.Rmd` which contains the code to reproduce the experiements of Section 6.3 of the paper
```
The *data* folder contain the DLD, TGF-$\beta$, Shapley Galaxy and Pima Indians datasets used in the paper. The *stan* folder contains the necessary .stan files to sample from the KLD-Bayes and $\beta$D-Bayes posteriors.

### Contact Information and Acknowledgments

stabilityGBI was developed by Jack Jewson, Universitat Pompeu Fabra, Barcelona and The Barcelona School of Economics (jack.jewson@upf.edu). 

The project was was partially funded by the Ayudas Fundación BBVA a Equipos de Investigación Cientifica 2017, the Government of Spain's Plan Nacional PGC2018-101643-B-I00, and Juan
de la Cierva Formación fellowship  FJC2020-046348-I.
