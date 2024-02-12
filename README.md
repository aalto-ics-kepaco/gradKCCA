# Large-Scale Sparse Kernel Canonical Correlation Analysis

The matlab folder contains the MATLAB codes applied in [1]. The python folder contains the python version of gradKCCA together with examples to simulate data.

## Real Datasets

* [MNIST Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
* [MediaMill](https://rdrr.io/github/fcharte/mldr.datasets/man/mediamill.html)

## Authors and Contact Information

* **Viivi Uurtio** * - <viivi.uurtio@aalto.fi>
* **Sahely Bhadra** * - <sahely@iitpkd.ac.in>
* **Juho Rousu** - <juho.rousu@aalto.fi>

\* Answer considerations regarding the codes 

## Reference

Viivi Uurtio, Sahely Bhadra, and Juho Rousu. Large-scale sparse kernel canonical correlation analysis. In Kamalika
Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning,
volume 97 of Proceedings of Machine Learning Research, pages 6383–6391, Long Beach, California, USA, 09–15 Jun
2019. PMLR.



## Update on February 2024
Fixed a few small bugs:
- **Matlab**: added centering after kernel evaluations
- **Python**: 
  - Added centering after kernel evaluations
  - Fixed an indexing error in L1 norm
  - Fixed RBF kernel calculation
  - Introduced a more flexible way of defining kernel parameters using dictionaries for the parameters