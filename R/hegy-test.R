
hegy.test <- function(x, deterministic = c(1,0,0), 
  lag.method = c("fixed", "AIC", "BIC", "AICc"), maxlag = 0,
  pvalue = c("RS", "bootstrap", "raw"), 
  #args related to response surface method
  rs.nobsreg = 15, 
  #args related to bootstrap method
  boot.args = list(seed = 123, 
    #lag.method = c("fixed", "AIC", "BIC", "AICc"), maxlag = 0,
    lag.method = lag.method[1], maxlag = maxlag, 
    byseason = FALSE, nb = 1000, BTdim = c(100, 10), debug.tid = -1))
{
  F.test.stat <- function(id)
  {
    if (length(id) == S)
    {
      if (is.null(xreg))
      {
        #model with no HEGY regressors and deterministic=c(0,0,0)
        rss1 <- sum(dx^2)
        dfs1 <- length(dx)
        return(((rss1 - rss2) / (dfs1 - dfs2)) / (rss2 / dfs2))
      } else
        fit1 <- lm(dx ~ 0 + xreg)
    } else
      fit1 <- lm(dx ~ 0 + cbind(xreg, ypi[,-id]))
    rss1 <- deviance(fit1)
    dfs1 <- df.residual(fit1)
    ((rss1 - rss2) / (dfs1 - dfs2)) / (rss2 / dfs2)
  }

  xreg <- NULL
  stopifnot(is.ts(x))
  n <- length(x)
  S <- frequency(x)
  if (S < 2)
    stop("The dataset either does not conform to a time series format or its frequency is less than two.")
  isSeven <- (S %% 2) == 0
  isSevenp2 <- 2 + isSeven
  dx <- diff(x, S)

  #xreg <- if (is.matrix(xreg)) xreg[-seq_len(S),] else xreg[-seq_len(S)]
  if (!is.null(xreg) && NROW(xreg) != length(dx))
      stop("the number of rows in argument ", sQuote("xreg"), 
        " does not match the length of ", sQuote("diff(x, frequency(x))"))

  data.name <- deparse(substitute(x))
  lag.method <- match.arg(lag.method)
  isNullxreg <- is.null(xreg)
  pvalue <- match.arg(pvalue)

  if (lag.method == "AICc" && pvalue == "RS")
  {
    ##NOTE 
    #response surfaces are not provided by the reference paper DE14
    #they are available for the Hannan-Quinn criterion, see add this option here
    lag.method <- "AIC"
    warning("argument ", sQuote("lag.method"), " was changed to ", sQuote("AIC"))
  }

  if (pvalue == "bootstrap")
  {
    #if (!isNullxreg)
    #  warning("argument ", sQuote("xreg"), " was set to NULL")

    #if (!is.null(boot.args$lag.method))
    #boot.args$lag.method <- boot.args$lag.method[1]

    # default arguments
    #bargs <- list(seed = 123, lag.method = "fixed",
    #  maxlag = 0, byseason = FALSE, nb = 1000, BTdim = c(100, 10), debug.tid = -1)
    bargs <- list(seed = 123, lag.method = lag.method[1],
      maxlag = maxlag, byseason = FALSE, nb = 1000, BTdim = c(100, 10), debug.tid = -1)

    nms1 <- names(boot.args)
    notinldef <- !(nms1 %in% names(bargs))
    wno <- which(notinldef)

    if (any(notinldef))
      warning("the following elements in ", sQuote("boot.args"), " were omitted: ", 
      paste(sQuote(nms1[wno]), collapse = ", "), ".", call. = FALSE)

    if (length(boot.args) > 0)
    {
      if (any(notinldef)) {
        bargs[nms1[which(!notinldef)]] <- boot.args[-wno] 
      } else
        bargs[nms1] <- boot.args
    }
  }

  # storage matrix

  stats <- matrix(nrow = floor(S/2) + 3, ncol = 2)
  id <- seq.int(isSevenp2, S, 2)
  rownames(stats) <- c("t_1", if (isSeven) "t_2" else NULL,
  paste("F", apply(cbind(id, id + 1), 1, paste, collapse=":"), sep = "_"), 
  paste0("F_2:", S), paste0("F_1:", S))
  colnames(stats) <- c("statistic", "p-value")

  # deterministic terms

  if (all(deterministic == c(0,0,1)))
    deterministic <- c(1,0,1)

  if (deterministic[1] == 1) {
    strdet <- "constant"
    xreg <- cbind(xreg, c = rep(1, n - S))
  } else strdet <- ""

  ##NOTE
  #in principle it is not a good idea to define the intercept in "xreg" and 
  #use lm(x ~ 0 + xreg) because stats::summary.lm uses attr(z$terms, "intercept")
  #to compute the R-squared and the F statistic includes the intercept (not the usual test),
  #but here the R-squared and the F-statistic are not used

  if (deterministic[2] == 1) {
    xreg <- cbind(xreg, trend = seq_along(dx))
    strdet <- paste(strdet, "+ trend")
  }

  if (deterministic[3] == 1)
  {
    SD <- do.call("rbind", replicate(ceiling(length(dx)/S), diag(S), simplify = FALSE))
    SD <- ts(SD, frequency = S, start = start(dx))
    # ignore warning "'end' value not changed"
    SD <- suppressWarnings(window(SD, start = start(dx), end = end(dx)))
    colnames(SD) <- paste0("SD", seq_len(S))
    if (deterministic[1] == 1)
      SD <- SD[,-1]
    xreg <- cbind(xreg, SD)
    strdet <- paste0(strdet, ifelse(deterministic[1] == 1, " + ", ""), "seasonal dummies")
  }

  if (strdet == "")
    strdet <- "none"

  # lags of the dependent variable

  if (maxlag > 0)
  {
    dxlags <- dx
    for (i in seq_len(maxlag))
      dxlags <- cbind(dxlags, lag(dx, -i))
    dxlags <- window(dxlags, end = end(dx))[,-1]
    if (is.null(dim(dxlags)))
      dxlags <- as.matrix(dxlags)
    #NOTE this way would be a bit faster 
    #sapply(seq_len(maxlag), function(x) c(rep(NA,length.out=x), c(1:10)[1:(10-x)]))
    colnames(dxlags) <- paste("Lag", seq_len(maxlag), sep ="")

    if (lag.method == "fixed")
    {
      #NOTE 'data.frame' keeps column names, 'cbind' doesn't;
      #'data.matrix' is required in order to be accepted by 'lm'
      xreg <- data.matrix(data.frame(xreg, dxlags))
    } # else, do not add dxlags here since they will be selected below

  } #else # maxlags == 0 #nlags == 0
    #str.lag.order <- lag.order <- 0

  # HEGY regressors

  ypi <- hegy.regressors(x)

  # lag order selection

  if (maxlag > 0)
  {
    if (lag.method != "fixed")
    {
      # information criteria are obtained using the same number of observations
      # regardless of the number of lags included in a given model
      tmp <- vector("list", maxlag + 1)
      # model with no lags
      id <- seq_len(maxlag)
      tmp[[1]] <- lm(dx[-id] ~ 0 + ypi[-id,] + xreg[-id,])
      dxlags2 <- dxlags[-id,]

      #http://stackoverflow.com/questions/23154074/writing-a-loop-in-r-that-updates-a-model?rq=1
      #the loop does not resolve the "i" and new variables are not distinguished
      #tmp[[i+1]] <- update(tmp[[i]], . ~ . + dxlags[,i,drop=FALSE])
      for (i in seq_len(maxlag))
        tmp[[i+1]] <- update(tmp[[i]], as.formula(paste0(". ~ . + dxlags2[,",i,"]")))

      icvals <- unlist(switch(lag.method, 
        "AIC" = lapply(tmp, AIC), "BIC" = lapply(tmp, BIC),
        "AICc" = lapply(tmp, function(x) { k <- x$rank+1; -2*logLik(x) + 2*k + 
          (2*k*(k+1))/(length(residuals(x))-k-1) })))
      id <- which.min(icvals)
      # update lag order (tmp[[1]] contains 0 lags)
      maxlag <- id - 1
      if (maxlag > 0)
      {
        xreg <- data.matrix(data.frame(xreg, dxlags[,seq_len(maxlag),drop=FALSE]))
      }
    } # else, lag.method=="fixed", dxlags was added to reg above
  }

  # fit the model with the chosen lags (if any) 
  # using the entire sample (i.e., including the first 'maxlag' observations)
  fit2 <- lm(dx ~ 0 + ypi + xreg)
  
  # test statistics

  rss2 <- deviance(fit2)
  dfs2 <- df.residual(fit2)

  j <- isSevenp2
  for (i in seq.int(isSevenp2, S, 2))
  {
    id <- c(i, i+1)
    stats[j,1] <- F.test.stat(id)
    j <- j + 1
  }

  stats[j,1] <- F.test.stat(seq.int(2, ncol(ypi)))

  # F-test statistic for all HEGY regressors
  stats[j+1,1] <- F.test.stat(seq_len(S))

  # t-test statistics
  id <- seq_len(1 + isSeven) #+ ncxreg
  stats[seq_along(id),1] <- coef(fit2)[id] / sqrt(diag(vcov(fit2))[id])

  # p-values

  # used with pvalue equal to "RS" or "raw"
  ctd <- paste(deterministic, collapse = "")

  if (pvalue == "RS")
  {
    stats[1,2] <- hegy.rs.pvalue(stats[1,1], "zero", ctd, lag.method, 
      maxlag, S, dfs2, rs.nobsreg)
    if (isSeven)
      stats[2,2] <- hegy.rs.pvalue(stats[2,1], "pi", ctd, lag.method, 
        maxlag, S, dfs2, rs.nobsreg)

    for (i in seq.int(isSevenp2, (S+4)/2-1))
      stats[i,2] <- hegy.rs.pvalue(stats[i,1], "pair", ctd, lag.method, 
        maxlag, S, dfs2, rs.nobsreg)

    j <- nrow(stats) - 1
    stats[j,2] <- hegy.rs.pvalue(stats[j,1], "seasall", ctd, lag.method, 
      maxlag, S, dfs2, rs.nobsreg)
    stats[j+1,2] <- hegy.rs.pvalue(stats[j+1,1], "all", ctd, lag.method, 
      maxlag, S, dfs2, rs.nobsreg)
  } else 
  if (pvalue == "bootstrap")
  {
    e <- residuals(fit2)
    dgp.nlags <- maxlag
    if (dgp.nlags > 0)
    {
      arcoefs <- coef(fit2)
      arcoefs <- arcoefs[grepl("Lag\\d{1,3}$", names(arcoefs))]
      if (length(arcoefs) != dgp.nlags) # debug
        stop("unexpected number of lags")
    } else
      arcoefs <- 0

    BTdim <- bargs$BTdim
    nb <- bargs$nb
    if (is.null(BTdim)) {
      BTdim <- rep(ceiling(sqrt(nb)), 2)
    } else
    if (prod(BTdim) < nb) {
      stop("the number of threads is lower than the number of bootstrap replicates")
    }

    #NOTE n-S is passed as the length of the data, 
    #the length of diff(x,S) is required by hegy_boot_pval

    if (bargs$byseason)
    {
      e <- ts(e, frequency = S)
      csns <- cumsum(table(cycle(e)))
      #names(csns) <- cycle(dx)[seq.int(dgp.nlags+1, length.out=S)]

      e <- unlist(split(e, cycle(e)))

    } else {
      csns <- c(0, length(e))
    }

    boot.lag.method <- match.arg(bargs$lag.method, c("fixed", "AIC", "BIC"))
    chosen.lags <- if (boot.lag.method == "fixed")
      integer(1) else integer(bargs$nb)

    ## removed with cuda stuff
    ## if (@HAS_CUDA@)
    ## {
    ##   tmp <- .C("hegy_boot_pval", dim = as.integer(c(nb, BTdim)), 
    ##     seed = as.integer(bargs$seed), idc = as.integer(deterministic), 
    ##     ip = as.integer(c(S, n-S, dgp.nlags, bargs$maxlag, bargs$debug.tid)),
    ##     csns = as.integer(csns),
    ##     eps = as.double(e), arcoefs = as.double(arcoefs), 
    ##     stats0 = as.double(stats[,1]), 
    ##     ICtype = as.integer(switch(boot.lag.method, 
    ##       "fixed" = 0, "AIC" = 1, "BIC" = 2, "AICc" = 3)), 
    ##     bpvals = double(nrow(stats)), chosen_lags = chosen.lags, PACKAGE = "uroot")
    ##   stats[,2] <- tmp$bpvals
    ##   chosen.lags <- tmp$chosen_lags
    ## } else {
    ##   if (grepl("windows", .Platform$OS.type)) {
    ##     stop("GPU parallelization could not be tested on a Windows system,\n", 
    ##     "consider using the function ",  sQuote("hegy.boot.pval"))
    ##   } else # unix
    ##     stop("CUDA capable GPU was not available when installing the package on this system,\n", 
    ##     "consider using the function ",  sQuote("hegy.boot.pval"))
    ## }
    stop("'pvalue = bootstrap' was only available with CUDA, but\n",
         "CUDA capablities in package 'uroot' were removed from v. 2.1-0,\n", 
         "consider using the function ",  sQuote("hegy.boot.pval"))

  } else
  if (pvalue == "raw")
  {
    #NOTE documentation deterministic == c(0,0,1) not available 
    #in original tables (use c(1,0,1) instead)
    stats[1,2] <- uroot.raw.pvalue(stats[1,1], "HEGY", NULL, dfs2, ctd, S, "zero")
    if (isSeven)
      stats[2,2] <- uroot.raw.pvalue(stats[2,1], "HEGY", NULL, dfs2, ctd, S, "pi")
    for (i in seq.int(isSevenp2, (S+4)/2-1))
      stats[i,2] <- uroot.raw.pvalue(stats[i,1], "HEGY", NULL, dfs2, ctd, S, "pair")
  }

  # output

  res <- list(statistics = stats[,1], pvalues = stats[,2],
    method = "HEGY test for unit roots", data.name = data.name, 
    fitted.model = fit2, 
    lag.method = lag.method, lag.order = maxlag,
    strdet = strdet, type.pvalue = pvalue,
    bootstrap = if (pvalue == "bootstrap") bargs else NULL,
    boot.chosen.lags = if (pvalue == "bootstrap") chosen.lags else NULL,
    pvlabels = symnum(stats[,"p-value"], corr = FALSE, na = FALSE,
      cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
      symbols   =  c("***","**","*","."," ")))
  class(res) <- "HEGYtest"

  res
}
