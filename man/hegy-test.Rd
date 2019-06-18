\name{hegy.test}
\alias{hegy.test}
\alias{hegy.regressors}
\alias{hegy.print}
\alias{hegy.residuals}
\alias{hegy.summary}

\title{Hylleberg, Engle, Granger and Yoo Test for Seasonal Unit Roots}

\description{
Hylleberg, Engle, Granger and Yoo (HEGY) test statistics for the null hypothesis 
seasonal unit roots.
}

\usage{
hegy.test(x, deterministic = c(1,0,0),
  lag.method = c("fixed", "AIC", "BIC", "AICc"), maxlag = 0,
  pvalue = c("RS", "bootstrap", "raw"), rs.nobsreg = 15, 
  boot.args = list(seed = 123, lag.method = lag.method[1], maxlag = maxlag, 
    byseason = FALSE, nb = 1000, BTdim = c(100, 10), debug.tid = -1))

hegy.regressors(x)
}

\arguments{
\item{x}{a univariate seasonal time series.}
%
\item{deterministic}{a vector of length three containing zeros or ones to indicate, respectively,
whether a constant, a trend or seasonal dummies are included in the regression equation 
of the test.}
%
\item{lag.method}{a character specifying the lag order selection method.}
%
\item{maxlag}{the maximum lag order to be considered by \code{lag.method}.}
%
%\item{xreg}{an optional vector or matrix of external regressors with the same length or number of 
%rows as the length of \code{diff(x, frequency(x))}. Ignored if \code{pvalue = "bootstrap"}.}
%
\item{pvalue}{a character specifying the method employed to compute p-values:
\code{"RS"}, the default, interpolation based on response surface regressions;
\code{"bootstrap"}, bootstrap;
\code{"raw"} interpolation in the tables provided in the reference papers.}
%
\item{rs.nobsreg}{an integer indicating the number of points employed in the response surface 
regression (only for \code{pvalue = "RS"}).}
%
\item{boot.args}{a list containing the parameters employed in the bootstrap. See details below.}
}

\details{
The regression equation employed to obtain the tests statistics may include 
the following deterministic terms: a constant, a linear trend, seasonal dummies.
These terms are selected by setting to 1 the corresponding element
in the vector \code{deterministic}: 
\code{deterministic = c(0,0,0)} no deterministic components,
\code{deterministic = c(1,0,0)} includes a constant,
\code{deterministic = c(1,0,1)} a constant and seasonal dummies,
\code{deterministic = c(1,1,0)} a constant and a trend and
\code{deterministic = c(1,1,1)} includes the three components.

The regression equation may include lags of the dependent variable.
When \code{lag.method = "fixed"}, the lag order is fixed to \code{maxlag}; 
otherwise, \code{maxlag} is the maximum number of lags considered in a 
lag selection procedure that minimises the \code{lag.method} criterion, 
which can be \code{\link[stats]{AIC}} or \code{\link[stats]{BIC}} or 
corrected AIC, \code{AICc}, obtained as \eqn{AIC + \frac{2k(k+1)}{n-k-1}}{AIC + (2k(k+1))/(n-k-1)}, 
where \eqn{k} is the number of parameters and \eqn{n} is the number of 
available observations in the model.

Response surface based \eqn{p}-values, \code{pvalue="RS"}, is not available 
with option \code{lag.method = "AICc"}.

%Currently, argument \code{xreg} is not considered if \code{pvalue = "bootstrap"};
%a warning is returned if \code{xreg} is not \code{NULL}.

P-values based on response surface regressions follow the method described in 
MacKinnon (1994), Harvey+vanDijk (2006) and Díaz-Emparanza (2014).
Gretl code provided by Díaz-Emparanza (2014)
has been ported to function \code{\link{hegy.rs.pvalue}}.
Note: the case \code{deterministic = c(0,0,0)} is not considered;
\code{NA}s are returned if p-values based on response surfaces are selected for that case.

Bootstrapped p-values follow the approach described in 
Burridge and Robert Taylor (2004).
The following arguments can be defined in \code{boot.args}:

\itemize{
\item \code{seed}: a numeric, the seed of the random generator employed for resampling the residuals.
%
\item \code{lag.method}: a character, the lag order selection method. This is the same 
as \code{lag.method} employed in the original regression, except that here, 
the AIC, BIC and AICc are obtained upon the residual sums of squares rather than the likelihood value.
%
\item \code{maxlag}: a numeric, maximum number of lags. Same behaviour as \code{maxlag} 
explained above for the original regression.
%
\item \code{byseason}: logical, should the residuals be resampled by season? If \code{TRUE}, 
the residuals belonging to the same season are resampled 
(at each time \eqn{t}, the residuals belonging to the corresponding season 
at time \eqn{t} are resampled);
otherwise, the entire series of residuals is resampled regardless of the season they 
belong to.
%the residuals are split by the season they belong to and resampled accordingly
%
\item \code{nb}: a numeric, the number of bootstrap replicates.
%
\item \code{BTdim}: a vector of length two containing the number of blocks and the number 
of threads per block to be run on parallel on the GPU. The product of these two elements must 
be equal or greater than \code{nb}.
%
\item \code{debug.tid}: an integer employed for debugging. Currently ignored.
%a thread id for which intermediate results are printed
%tracing information/results
}

By default \code{boot.args$lag.method} and \code{boot.args$maxlag} are set equal 
to the same options employed for the original data in arguments \code{lag.method} 
and \code{maxlag}; if the default options are desired, these values need not be explicitly 
defined in the list \code{boot.args}.

The standard definition of the AIC, BIC and AICc criteria is used for the original series.
For the bootstrapped series, these criteria are defined upon the residual sum squares of 
the model:
%
\deqn{\hbox{AIC} = n \log\left(\hbox{RSS}/n\right) + 2 k}{%
AIC = n log RSS/n + 2 k}
%
\deqn{\hbox{BIC} = n \log\left(\hbox{RSS}/n\right) + k \log(n)}{%
BIC = n log RSS/n + k log(n)}
%
\deqn{\hbox{AICc} = \hbox{AIC} + \frac{2k(k+1)}{n-k-1}}{%
AICc = AIC + (2 k (k+1)) / (n - k - 1)}
%
where \eqn{RSS} is the residual sum of squares, \eqn{k} is the number of parameters 
in the model and \eqn{n} is the number of available observations.

Given a maximum lag, \code{maxlag}, the first \code{maxlag} observations are removed 
before fitting the models to be compared by the selected criterion. In this way, 
all the models contain the same number of observations. 
Once the lag order is selected, the statistics are obtained using 
the entire sample (except those observations that are missed due to the eventually chosen lags).

The HEGY regressors were originally proposed in Hylleberg \emph{etal.} (1990) 
for quarterly data. They are generalized to monthly series in Beaulieu and Miron (1993)
and to weekly data in Cáceres (1996).
Franses and Hobijn (1997) show tabulated values for bimonthly and biannual data 
as well as quarterly and monthly data. 
\code{hegy.rergessors} follows the expressions given in 
Smith \emph{etal.} (2009), which define the regressors for a general periodicity 
of the time series.
}

\value{
\code{hegy.test} returns a list of class \code{"HEGYtest"} with the following components:
%
\item{statistics}{the value of the test statistics.}
%
\item{pvalues}{the p-values for each test statistics.}
%
\item{method}{a character string describing the type of test.}
%
\item{data.name}{a character string giving the name of the data.}
%
\item{fitted.model}{the fitted regression model.}
%
\item{lag.method}{a character, the lag order selection criterion.}
%
\item{lag.order}{a numeric, the number of lags included in the regression.}
%
\item{strdet}{a character, auxiliary element for \code{print} describing the deterministic 
elements that were selected.}
%
%\item{isNullxreg}{logical, auxiliary element for \code{print} method. Were external 
%regressors defined in the argument \code{xreg}?}
%
\item{type.pvalue}{a character, the value of the input argument code{pvalue}.}
%
\item{bootstrap}{a list, parameter options employed in the bootstrap (if \code{pvalue = "bootstrap"}).}
%
\item{boot.chosen.lags}{a vector, the lag orders chosen for each bootstrap replicate 
(if \code{pvalue = "bootstrap"} and \code{boot.args$lag.method != "fixed"}).}
%
\item{pvlabels}{a vector of characters containing a label related to each p-values. 
Auxiliary element for \code{print} method.}

The method \code{print} displays the test statistics and p-values;
\code{summary} shows the same output and includes the fitted regression model;
\code{residuals} returns the residuals from the regression model fitted to the 
original data.

\code{hegy.regressors} returns a matrix containing the HEGY regressors 
which are used to test null of unit root at different frequencies.
}

\seealso{
\code{\link{hegy.rs.pvalue}}, \code{\link{uroot.raw.pvalue}}.
}

\references{
Beaulieu, J. J. Miron, J. A. (1993)
"Seasonal unit roots in aggregate U.S. data."
\emph{Journal of Econometrics} \bold{55}(1-2), pp. 305-328.
DOI: \url{http://dx.doi.org/10.1016/0304-4076(93)90018-Z}.

Burridge, P. and Taylor, R.  (2004)
"Bootstrapping the HEGY seasonal unit root tests."
\emph{Journal of Econometrics} \bold{123}(1), pp. 67-87.
DOI: \url{http://dx.doi.org/10.1016/j.jeconom.2003.10.029}.

Cáceres, J. J. (1996)
"Contraste de raíces unitarias en datos semanales."
\emph{Estadística Española} \bold{38}(41), pp. 139-159.

Díaz-Emparanza, I. (2014)
"Numerical distribution functions for seasonal unit root tests"-
\emph{Computational Statistics and Data Analysis}, \bold{76}, pp. 237-247.
DOI: \url{http://dx.doi.org/10.1016/j.csda.2013.03.006}.
Gretl code: \url{http://www.ehu.eus/ignacio.diaz-emparanza/packages/GHegy.gfn}.

Franses, F. H. (1991)
"Seasonality, non-stationarity and the forecasting of monthly time series."
\emph{International Journal of Forecasting} \bold{7}(2), pp. 199-208.
DOI: \url{http://dx.doi.org/10.1016/0169-2070(91)90054-Y}.

Franses, P. H. and Hobijn, H. (1997)
"Critical values for unit root tests in seasonal time series."
\emph{Journal of Applied Statistics} \bold{24}(1), pp. 25-47.

Harvey D. I. and van Dijk D. (2006). "Sample size, lag order and critical values of seasonal
unit root tests." \emph{Computational Statistics & Data Analysis}, \bold{50}(10), 2734-2751. 
DOI: \url{http://dx.doi.org/10.1016/j.csda.2005.04.011}.

Hylleberg, S., Engle, R., Granger, C. and Yoo, B. (1990)
"Seasonal integration and cointegration."
\emph{Journal of Econometrics} \bold{44}(1), pp. 215-238.
DOI: \url{http://dx.doi.org/10.1016/0304-4076(90)90080-D}.

MacKinnon J. G. (1994). "Approximate asymptotic distribution functions for unit-root and
cointegration tests." \emph{Journal of Business and Economic Statistics}, \bold{12}(2), 167-176. 
DOI: \url{http://dx.doi.org/10.1080/07350015.1994.10510005}.

Smith, R. J., Taylor, A. M. R. and del Barrio Castro, T. (2009)
"Regression-based seasonal unit roots."\emph{Econometric Theory} \bold{25}(2), pp. 527-560.
DOI: \url{http://dx.doi.org/10.1017/S0266466608090166}.
}

\examples{
x <- bgt.data[["LCONSEXPCO"]]
hegy.test(x, deterministic = c(1,1,1), lag.method = "fixed", maxlag = 1)

\dontrun{
# this requires CUDA capable GPU
hegy.test(x, deterministic = c(1,1,1), lag.method = "fixed", maxlag = 1, 
  pvalue = "bootstrap")}
}

\keyword{ts}
