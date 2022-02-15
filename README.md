# README #

The provided code and figures are given for reproducibility of the dejittering results we obtained for ICIP 2022.

### How do I get set up? ###

* Get python
* Make sure to have numpy, matplotlib and scikit-image libraries installed (we had numpy=1.22.2 matplotlib=3.5.1 and scikit-image=0.19.1)
* To reproduce the figures obtained for the ICIP publication, launch ICIP_Figures.py. Results are saved in ICIP_Figures directory.
* To use one of the two dejittering algorithms (DP or Weight relaxation) go to the corresponding file. A "demo" function allows you to launch the algorithm on an example. From its code you can deduce how to use it for your own jittered images.
* A specificity of the Weight Relaxation file is that there are 2 algorithms. They correspond to 2 different implementations, the one using a "pre-computer" is faster and was the one used for the results reported in the submission.

### Who do I talk to? ###

For any issues with the code or questions regarding the article you can contact Pierre Weiss at pierre.armand.weiss@gmail.com and Landry Duguet at landry.duguet@free.fr
