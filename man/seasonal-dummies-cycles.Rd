\name{seasonal.dummies}
\alias{seasonal.dummies}
\alias{seasonal.cycles}

\title{Seasonal Dummies and Seasonal Cycles}

\description{
Generate variables of seasonal dummies and seasonal cycles.
}

\usage{
seasonal.dummies(x)
seasonal.cycles(x)
}

\arguments{
\item{x}{a univariate seasonal time series.}
}

\value{
A multivariate time series containing the dummies or cycles by columns. 
}

\examples{
# In terms of model fitting
# both sets of variables are equivalent
x <- diff(log(AirPassengers))
sd <- seasonal.dummies(x)
fit1 <- lm(x ~ sd[,-1])
summary(fit1)
sc <- seasonal.cycles(x)
fit2 <- lm(x ~ sc)
summary(fit1)
all.equal(fitted(fit1), fitted(fit2))
}

\keyword{ts}
