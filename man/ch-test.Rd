\name{ch.test}
\alias{ch.test}
\alias{ch.print}
\alias{ch.summary}

\title{Canova and Hansen Test for Seasonal Stability}

\description{
Canova and Hansen (CH) test statistic for the null hypothesis 
of a stable seasonal pattern.
}

\usage{
ch.test(x, type = c("dummy", "trigonometric"), lag1 = FALSE, NW.order = NULL, 
  sid = NULL, xreg = NULL, pvalue = c("RS", "raw"), rs.nobsreg = 13)
}

\arguments{
\item{x}{a univariate seasonal time series.}
%
\item{type}{a character string specifying the formulation of the test,
\code{"dummy"} for seasonal dummies or \code{"trigonometric"} for seasonal cycles.}
%
\item{lag1}{logical, if \code{TRUE} a first order lag of the time series \code{x} is included 
in the regression model. The default is \code{FALSE}.}
%
\item{NW.order}{an integer, the lag truncation parameter to be used in 
the Newey and West covariance matrix.}
%
\item{sid}{an optional numeric vector, the target seasonal dummies or cycles to be tested. 
By default all the individual and joint test statistics are returned.}
%
\item{xreg}{an optional vector or matrix of external regressors with the same length or number of 
rows as the length of the input time series \code{x}.}
%
\item{pvalue}{a character specifying the method employed to compute p-values: 
\code{"RS"}, the default, interpolation based on response surface regressions;
\code{"raw"}, interpolation in the tabulated values provided in the reference paper 
for the Von Mises distribution.}
%
\item{rs.nobsreg}{an integer indicating the number of points employed in the response surface 
regression (only for \code{pvalue = "RS"}).}
}

\details{
The seasons or seasonal cycles to be tested can be chosen through 
an indicator variable defined in the argument \code{sid}. 
By default, all the \eqn{t}-statistics
related to each individual dummy or cycle and the joint \eqn{F}-statistic 
%for the test for seasonal stability at all seasonal dummies or cycles 
are returned.
%
If \code{type = "dummy"}, the index of the target seasons can be specified in \code{sid}. 
For example, in a quarterly series:
\code{sid=c(2)} returns the test statistic to the stability of the second quarter;
\code{sid=c(1,3)} returns the joint test statistic for the first and third quarters;
\code{sid=c(1,2,3,4)} returns the joint test statistic for the null of seasonal 
stability at all seasons.
%
If \code{type = "trigonometric"}, the indicator vector \code{sid} must be of length
\code{floor(frequency(x)/2)} and will consist of ones and zeros. Each element in 
\code{sid} is related to each seasonal cycle according to the same order in which 
the seasonal frequencies, \eqn{w_j}, are defined: \eqn{w_j=2\pi j/S}, \eqn{j=1,...,Sh}, 
where \eqn{S} is the periodicity and \eqn{Sh} is \code{floor(frequency(x)/2)}.
For example, in a monthly series:
\code{sid=c(0,0,0,0,0,1)} returns the test statistic to the stability of the cycle with 
frequency \eqn{w_6=\pi};
\code{sid=c(1,0,0,0,0,1)} returns the joint test statistic for cycles related 
to frequencies \eqn{w_1=\pi/6} and \eqn{w_6=\pi};
\code{sid=c(1,1,1,1,1,1)} returns the joint test statistic for the stability of 
all seasonal cycles.
The following keywords are also admitted:
\code{sid="all"}, computes all the test statistic related to each individual season 
or cycle as well as the joint test statistic for all seasons or cycles;
\code{sid="joint"} computes the joint test statistic for all seasons or cycles.
}

\note{
When \code{type = "dummy"}, the p-value for the joint test statistic 
based on response surface regressions is not available. If \code{pvalue = "RS"}, 
the p-value reported for the joint test statistic in the trigonometric version is based 
on the tables given in the reference paper, Canova and Hansen (1995).

When \code{sid} is a numeric (selected combination of dummies or cycles), 
the reported p-values are based on interpolation in tables;
if \code{pvalue = "RS"}, it is changed to \code{"raw"} and a warning is given.
}

\value{
A list of class \code{"CHtest"} with components:

\item{statistics}{the value of the test statistics.}
%
\item{pvalues}{the p-values for each test statistics.}
%
\item{method}{a character string describing the type of test.}
%
\item{data.name}{a character string giving the name of the data.}
%
\item{type}{the value of the input argument \code{type}.}
%
\item{fitted.model}{the fitted regression model.}
%
\item{NW.order}{the value of the input argument code{NW.order}.}
%
\item{isNullxreg}{logical, auxiliary element for \code{print} method. Were external 
regressors defined in the argument \code{xreg}?}
%
\item{type.pvalue}{character, the value of the input argument code{pvalue}.}
%
\item{pvlabels}{a vector of characters containing a label related to each p-values. 
Auxiliary element for \code{print} method.}

The method \code{print} displays the test statistics and p-values;
\code{summary} shows the same output and includes the fitted regression model.
}

\seealso{
\code{\link{ch.rs.pvalue}} \code{\link{seasonal.cycles}}, 
\code{\link{seasonal.dummies}}, \code{\link{uroot.raw.pvalue}}.
}

\references{
Canova, F. and Hansen, Bruce E. (1995)
"Are seasonal patterns constant over time? A test for seasonal stability".
\emph{Journal of Business & Economic Statistics}, \bold{13}(3), pp. 237-252.
DOI: \url{http://dx.doi.org/10.1080/07350015.1995.10524598}.

Díaz-Emparanza, I. and Moral, M. P. (2013).
\emph{Seasonal stability tests in gretl. An application to international tourism data}.
Working paper: Biltoki D.T. 2013.03. 
URL: \url{https://addi.ehu.es/handle/10810/10577}.
Gretl code: \url{http://www.ehu.eus/ignacio.diaz-emparanza/packages/Canova_Hansen.gfn}.
}

\examples{
library(uroot)
# example for the series "hours" with the same options 
# employed in Canova and Hansen (1995)
data("ch-data")
hours <- diff(log(ch.data$hours))
res1 <- ch.test(x = hours, type = "dummy", lag1 = TRUE, NW.order = 4)
res1
# the auxiliary regression is stored in the element "fitted.model"
summary(res1$fit)

\dontrun{
# this requires tables not included in the current version of the package 
# see note in main documentation file, uroot-package
res2 <- ch.test(x = hours, type = "trigonometric", lag1 = TRUE, NW.order = 4)
res2
summary(res2$fit)}
}

\keyword{ts}
