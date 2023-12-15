\name{hegy.rs.pvalue}
\alias{hegy.rs.pvalue}

\title{P-values based on response surface regressions for the HEGY test statistics }

\description{
Compute p-values for the Hylleberg, Engle, Granger and Yoo (HEGY) test statistic 
by interpolation in precompiled response surfaces.
}

\usage{
hegy.rs.pvalue(x, type = c("zero", "pi", "pair", "seasall", "all"), 
  deterministic = c(1, 0, 0), lag.method = c("fixed", "AIC", "BIC"),
  lag.order, S, n, nobsreg)
}

\arguments{
\item{x}{a numeric, the value of the HEGY statistic.}
%
\item{type}{a character, the type of test statistic, the regressor(s) to which 
the statistic is related.}
%
\item{deterministic}{a vector of length three containing zeros and ones to indicate, respectively,
whether a constant, a trend or seasonal dummies should be included in the regression equation 
of the test.}
%
\item{lag.method}{a character specifying the lag order selection criterion.}
%
\item{lag.order}{a numeric, the lag order employed in the auxiliary regression.}
%
\item{S}{numeric, the periodicity of the data.}
%
\item{n}{numeric, the number of observations.}
%
\item{nobsreg}{an integer indicating the number of points employed in the response surface 
regression.}
}

\details{
Ported from Gretl code provided by Díaz-Emparanza (2014).
%
The original source includes tables with coefficients for the 
calculation of p-values when the HQC lag order selection criterion is used.
These tables are not included here.
%
The case with no deterministic terms (\code{deterministic = c(0,0,0)}) is not considered.

This function is mainly intended to be used internally by \code{\link{hegy.test}}.
}

\value{
A numeric, the p-value.
}

\seealso{
\code{\link{hegy.test}}.
}

\references{
Díaz-Emparanza, I. (2014)
"Numerical Distribution Functions for Seasonal Unit Root Tests"-
\emph{Computational Statistics and Data Analysis} \bold{76}, pp. 237-247.
DOI: \doi{10.1016/j.csda.2013.03.006}.
Gretl code: https://www.ehu.eus/ignacio.diaz-emparanza/packages/Canova_Hansen.gfn/
(seems unavailable, so not linked)
}

\keyword{ts}
