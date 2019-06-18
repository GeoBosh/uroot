\name{uroot.raw.pvalue}
\alias{uroot.raw.pvalue}

\title{Original Tables of Critical Values}

\description{
Compute p-values by interpolation in the tables of critical values 
provided in the original references given below.
}

\usage{
uroot.raw.pvalue(x, type = c("CH", "HEGY"), v, n, ctd, S, Ftpi)
}

\arguments{
\item{x}{a numeric. The value of the CH statistic.}
%
\item{type}{a character specifying the type of test statistic.}
%
\item{v}{numeric, the degrees of freedom of the Von Mises distribution. Only for \code{type="CH"}.}
%
\item{n}{numeric, the number of observations.}
%
\item{ctd}{a character indicating the deterministic elements that were included 
in the HEGY regression. This argument is defined as
\code{paste(deterministic, collapse = "")}, where \code{deterministic} is the 
argument of that name that was passed to \code{\link{hegy.test}}.
Only for \code{type="ADF"} or \code{type="HEGY"}.}
%
\item{S}{numeric, the periodicity of the data.}
%
\item{Ftpi}{a character indicating whether the type of statistic:
\code{"zero"}, \eqn{t}-test for the zero frequency; 
\code{"pi"},  \eqn{t}-test for the frequency \eqn{\pi};
\code{"pair"}, \eqn{F}-test for the pairs of complex conjugates frequencies.
Only for \code{type="ADF"} or \code{type="HEGY"}.}
}

\details{
This function is used internally by \code{\link{ch.test}} and 
\code{\link{hegy.test}}.
}

\value{
A numeric giving the calculated p-value.
}

\seealso{
\code{\link{ch.test}}, \code{\link{hegy.test}}.
}

\references{
Beaulieu, J. J. Miron, J. A. (1993)
"Seasonal Unit Roots in Aggregate U.S. Data".
\emph{Journal of Econometrics}, \bold{55}(1-2), pp. 305-328.
DOI: \url{http://dx.doi.org/10.1016/0304-4076(93)90018-Z}.

Canova, F. and Hansen, Bruce E. (1995)
"Are Seasonal Patterns Constant over Time? A Test for Seasonal Stability".
\emph{Journal of Business & Economic Statistics}, \bold{13}(3), pp. 237-252.
DOI: \url{http://dx.doi.org/10.1080/07350015.1995.10524598}.

Hylleberg, S., Engle, R., Granger, C. and Yoo, B. (1990)
"Seasonal integration and cointegration".
\emph{Journal of Econometrics} \bold{44}(1), pp. 215-238.
DOI: \url{http://dx.doi.org/10.1016/0304-4076(90)90080-D}.

}

\keyword{ts}
