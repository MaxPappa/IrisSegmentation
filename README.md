# CV_IrisSegmentation
The aim of this project is to re-create (in the numpiest way) an iris segmentation.
Iris detection is done using Daugman's integro-differential operator [[1]](#1) on the multi-channel as done by Haindl-Krupička in [[2]](#2). Daugman's operator is then used on the red channel to detect the pupil.
After doing this, normalization is done with the rubber sheet model and occlusions are detected using methods described in [[2]](#2).

***

# References

1. <a id="1"/> J. G. Daugman, "High confidence visual recognition of persons by a test of statistical independence," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 15, no. 11, pp. 1148-1161, Nov. 1993, doi: 10.1109/34.244676
2. <a id="2"/> Michal Haindl, Mikuláš Krupička, Unsupervised detection of non-iris occlusions, Pattern Recognition Letters, Volume 57, 2015, Pages 60-65, ISSN 0167-8655, https://doi.org/10.1016/j.patrec.2015.02.012