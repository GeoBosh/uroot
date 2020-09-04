[![CRANStatusBadge](http://www.r-pkg.org/badges/version/uroot)](https://cran.r-project.org/package=uroot)


Unit Root Tests for Seasonal Time Series


# Installing uroot

Install the [latest stable version](https://cran.r-project.org/package=uroot) of
`uroot` from CRAN:

    install.packages("uroot")


You can install the [development version](https://github.com/GeoBosh/uroot) of
`uroot` from Github:

    library(devtools)
    install_github("GeoBosh/uroot")


# Overview

**Note:** All CUDA related stuff was removed in version 2.1.0 of uroot. 
        The last version with CUDA support was 2.0.11.


Seasonal unit roots and seasonal stability tests.
P-values based on response surface regressions are available for both tests.
P-values based on bootstrap are available for seasonal unit root tests.


** Windows systems:

GPU parallelization is not
currently available on Windows systems.


** Unix systems:

For full operational capabilities,
the 'uroot' package requires the following installed on the system:

  1) CUDA capable GPU with compute capability >= 3.0.
  
  2) CUDA software, which includes the 'nvcc' (release >= 7.1)
     NVIDIA Cuda Compiler driver (available at 
     https://www.nvidia.com).

  3) A general purpose C compiler is needed by nvcc.

By default the package is installed without the GPU capabilities.  To
request them, set environment variable CUDA_IGNORE to any nonempty value
for the R package installer.
