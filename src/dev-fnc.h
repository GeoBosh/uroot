#ifndef _DEV_FUNCTIONS_
#define _DEV_FUNCTIONS_

#include "R.h"

#include <curand_kernel.h>
//#include <stdio.h>

__device__ void boot_dgp_init(curandState *RNG_state, const int S, 
  const int dgp_nlags, const double *arcoefs, const int *csns, 
  const double *eps, double *dylags, double *y0);

__device__ void crossprod_regmatrix(curandState *RNG_state, const double *eps, 
  double *y0, const int nx, const int *csns, const int S, 
  const int idc0, const int idc1, const int idc2, const int nlags, 
  const int dgp_nlags, const double *arcoefs, 
  double *dylags, double *res);

__device__ void sweep(const int nv, const int nc, const int idcol, double *v);

__device__ void hegy_statistics(double *v, const int nv, const int nc, const int nr, 
  const int S, const int dgp_nlags, const double *arcoefs, double *res);

#endif
