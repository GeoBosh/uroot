\name{ch.rs.pvalue}
\alias{ch.rs.pvalue}

\title{P-values for the CH test statistic}

\description{
Compute p-values for the Canova and Hansen (CH) test statistic 
based on the response surface regressions approach.
}

\usage{
ch.rs.pvalue(x, type, lag1, S, n, nobsreg, VMdf)
}

\arguments{
\item{x}{a numeric. The value of the CH statistic.}
%
\item{type}{a string specifying the formulation of the test, 
\code{"dummy"} for seasonal dummies or \code{"trigonometric"} for seasonal cycles.}
%
\item{lag1}{logical indicating whether a first order lag of was included in the regression model.}
%
\item{S}{numeric, the periodicity of the data.}
%
\item{n}{numeric, the number of observations.}
%
\item{nobsreg}{an integer indicating the number of points employed in the response surface 
regression.}
%
\item{VMdf}{numeric, the degrees of freedom of the Von Mises distribution.}
}

\details{
Ported from Gretl code provided by Díaz-Emparanza and Moral (2013).

For \code{type="dummy"}, 
the p-value for the joint test statistic is not available.

This function is mainly intended to be used internally by \code{\link{ch.test}}.
}

\value{
A numeric giving the calculated p-value.
}

\seealso{
\code{\link{ch.test}}.
}

\references{
Díaz-Emparanza, I. and Moral, M. P. (2013) 
\emph{Seasonal Stability Tests in gretl. An Application to International Tourism Data}.
Working paper: Biltoki D.T. 2013.03. 
URL: \url{https://addi.ehu.es/handle/10810/10577}.
Gretl code: https://www.ehu.eus/ignacio.diaz-emparanza/packages/Canova_Hansen.gfn
(seems unavailable, so not linked)
}

\keyword{ts}
