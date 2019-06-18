#ifndef _DEV_KERNEL_
#define _DEV_KERNEL_

#include "R.h"
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "dev-fnc.h"

extern "C" {
void hegy_boot_pval(const int *dim, const unsigned int *seed, 
  const int *idc, const int *ip, const int *csns, const double *eps, const double *arcoefs,
  const double *stats0, const int *ICtype, double *bpvals, double *chosen_lags);
}

__global__ void kernel(const int N, const unsigned int seed, const double *eps,
  const int S, const int idc0, const int idc1, const int idc2, 
  const int nx, const int *csns, const int dgp_nlags, const double *arcoefs, 
  const double *stats0, const int ICtype, const int maxlag, const int debug_tid,
  int *res1, int *res2, int *res3, int *res4, int *res5, int *res6);

#endif
