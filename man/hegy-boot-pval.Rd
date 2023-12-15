\name{hegy.boot.pval}
\alias{hegy.boot.pval}

\title{Bootstrapped P-Values for the HEGY Test Statistics}

\description{
Compute p-values for the HEGY test statistics by means of bootstrap.
}

\usage{
hegy.boot.pval(x, model0, stats0, 
  deterministic = c(1,0,0), lag.method = c("fixed", "AIC", "BIC"), maxlag = 0, 
  byseason = FALSE, nb = 500, u = NULL, debug.tid = -1)
}

\arguments{
\item{x}{a univariate seasonal time series.}
%
\item{model0}{the \code{fitted.model} returned by \code{\link{hegy.test}} for the original data.}
%
\item{stats0}{the \code{statistics} returned by \code{\link{hegy.test}} for the original data.}
%
\item{deterministic}{a vector of length three containing zeros or ones to indicate, respectively,
whether a constant, a trend or seasonal dummies are included in the regression equation 
of the test.}
%
\item{lag.method}{a character specifying the lag order selection method.}
%
\item{maxlag}{the maximum lag order to be considered by \code{lag.method}.}
%
\item{byseason}{logical, should the residuals be resampled by season? If \code{TRUE}, 
the residuals are split by the season they belong to and resampled accordingly;
otherwise, the entire series of residuals is resampled regardless of the season they belong to.}
%
\item{nb}{the number of bootstrap replicates.}
%
\item{u}{optional matrix of integers giving the indices of the resampled residuals. 
Intended for debugging.}
%
\item{debug.tid}{numeric, if positive, the bootstrap replicate of the data generated 
at iteratin \code{debug.tid} is returned (intended for debugging).}
}

\details{
See \code{\link{hegy.test}} for further details about the arguments that have the 
same name in both functions (\code{deterministic}, \code{lag.method}, \code{maxlag}).

Bootstrapped p-values follow the approach described in 
Burridge and Robert Taylor (2004), except that here, the residuals
are resampled regardless of the season they belong to.
}

\value{
A numeric vector containing the p-values of the the test statistics.
The vector is named following the same convention as \code{statistics} and 
\code{pvalues} returned by \code{\link{hegy.test}}.

If the number of bootstrap replicates is \code{nb = 1}, the resampled 
series is returned (relevant for inspection of how the resampled series look like and 
for debugging).
}

\seealso{
\code{\link{hegy.test}}.
}

\references{
Burridge, P. and Taylor, R.  (2004)
"Bootstrapping the HEGY seasonal unit root tests."
\emph{Journal of Econometrics} \bold{123}(1), pp. 67-87.
DOI: \doi{10.1016/j.jeconom.2003.10.029}.

Hylleberg, S., Engle, R., Granger, C. and Yoo, B. (1990)
"Seasonal integration and cointegration."
\emph{Journal of Econometrics} \bold{44}(1), pp. 215-238.
DOI: \doi{10.1016/0304-4076(90)90080-D}.
}

\examples{
\dontrun{
x <- bgt.data[["LCONSEXPCO"]]
# this requires CUDA capable GPU
hegy.test(x, deterministic = c(1,1,1), lag.method = "fixed", maxlag = 1, 
  pvalue = "bootstrap")
# alternatively, full R non-parallel version
res <- hegy.test(x, deterministic = c(1,1,1), lag.method = "fixed", maxlag = 1)
hegy.boot.pval(x, res$fit, res$stat, deterministic = c(1,1,1), 
  lag.method = "fixed", maxlag = 1, nb = 1000)}
}

\keyword{ts}
