\name{uroot-package}
\docType{package}
\alias{uroot-package}

\title{Unit Root Tests for Seasonal Time Series}

\description{
Canova and Hansen (CH) test for seasonal stability and
Hylleberg, Engle, Granger and Yoo (HEGY) test for seasonal unit roots.
}

\details{
Version >= 2.0.0 is a revival of the former package \pkg{uroot}. 
Some of the functions provided in the original versions have been coded 
from the scratch.

The current version provides 
the Canova and Hansen (CH) test for seasonal stability and
the Hylleberg, Engle, Granger and Yoo (HEGY) test for seasonal unit roots.

%To my knowledge,  
%\code{\link[forecast]{ndiffs}} in the \pkg{forecast} package 
%is the only function on CRAN that tests for seasonal unit roots. 
%This function provides the CH test (based on the former version 
%of \pkg{uroot}) and the Osborn-Chui-Smith-Birchenhall test.

\emph{New features:}

The original functions have been enhanced with the following new features:
1) the tests are now applicable to series of any seasonal periodicity 
(not only quarterly and monthly data),
2) p-values based on response surface regressions are available,
3) bootstrapped p-values are available for the HEGY test statistics.

\emph{Old features not currently supported:}

Some of the utilities available in the initial versions of the package 
are not available now: graphics for seasonal series, graphical user interface, 
bootstrap versions of the statistics, ADF test and KPSS test.
Some of these utilities are available in other packages and some 
will probably be incorporated in future versions of this package.

\emph{System requirements:}

Windows systems: GPU parallelization of the bootstrap
is not currently operational on windows systems.

Unix systems: GPU parallelization of the bootstrap requires a
CUDA capable GPU with compute capability >= 3.0.
}

%\note{
%P-values based on the response surface method employs the tables obtained 
%and provided in Díaz-Emparanza (2014) and Díaz-Emparanza and Moral (2013). 
%To reduce the size of the package, only the tables that employ a fixed lag 
%order or the BIC for the HEGY test are included in the package; 
%for the CH test, the tables for the dummy version are included.
%The complete set of tables can be downloaded from the original sources or 
%can be directly loaded as an \file{rda} file available 
%\href{http://www.jalobe.com/wp-content/uploads/uroot/uroot-data.rda}{here}.
%}

\author{
Javier López-de-Lacalle \email{javlacalle@yahoo.es}.

%\url{http://jalobe.com}.
}

\references{
Burridge, P. and Taylor, R.  (2004)
"Bootstrapping the HEGY seasonal unit root tests."
\emph{Journal of Econometrics} \bold{123}(1), pp. 67-87.
DOI: \url{http://dx.doi.org/10.1016/j.jeconom.2003.10.029}.

Canova, F. and Hansen, Bruce E. (1995)
"Are seasonal patterns constant over time? A test for seasonal stability".
\emph{Journal of Business & Economic Statistics}, \bold{13}(3), pp. 237-252.
DOI: \url{http://dx.doi.org/10.1080/07350015.1995.10524598}.

Díaz-Emparanza, I. (2014)
"Numerical distribution functions for seasonal unit root tests"-
\emph{Computational Statistics and Data Analysis}, \bold{76}, pp. 237-247.
DOI: \url{http://dx.doi.org/10.1016/j.csda.2013.03.006}.

Díaz-Emparanza, I. and Moral, M. P. (2013).
\emph{Seasonal stability tests in gretl. An application to international tourism data}.
Working paper: Biltoki D.T. 2013.03. 

Hylleberg, S., Engle, R., Granger, C. and Yoo, B. (1990)
"Seasonal integration and cointegration".
\emph{Journal of Econometrics} \bold{44}(1), pp. 215-238.
DOI: \url{http://dx.doi.org/10.1016/0304-4076(90)90080-D}.
}
\keyword{package}
\keyword{ts}
