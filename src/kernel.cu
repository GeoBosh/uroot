#include "kernel.h"

__global__ void kernel(const int N, const unsigned int seed, const double *eps,
  const int S, const int idc0, const int idc1, const int idc2, 
  const int nx, const int *csns, const int dgp_nlags, const double *arcoefs, 
  const double *stats0, const int ICtype, const int maxlag, const int debug_tid,
  int *res1, int *res2, int *res3, int *res4, int *res5, int *res6)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < N)
  {
    //NOTE for nlags < maxlag nvA will be larger than necessary, but eventually 
    //the whole array is used in the loop over nlags; therefore it seems better to 
    //keep it this way instead of resizing the array

    int Sh = int(S/2), 
      // number of rows/columns in matrix of crossproducts, A
      Anrnv = S + idc0 + idc1 + idc2*(S-1) + maxlag + 1, 
      // length of the packed A matrix
      nvA = int((Anrnv*(Anrnv + 1)/2)),
      nlags = maxlag;

    double *dylags;
    if (dgp_nlags > 0)
    {
      dylags = (double *)malloc(dgp_nlags * sizeof(double));
      for (int j = 0; j < dgp_nlags; j++)
        dylags[j] = 0.0;
    }

    double *res_hegyreg = (double *)malloc((Sh+3) * sizeof(double));

    double *vA = (double *)malloc(nvA * sizeof(double));    
    double *y0 = (double *)malloc((S + dgp_nlags) * sizeof(double));
    for (int i = 0; i < S + dgp_nlags; i++)
      y0[i] = 0.0;

    for (int i = 0; i < nvA; i++)
      vA[i] = 0.0;

    // initialize RNG
    // the state of the RNG assigned to each thread is 2^67 elements apart 
    // from the previous thread, tid-1

    curandState localState, init_localState;
    curand_init(seed, tid, 0, &localState);

    if (ICtype > 0)
      init_localState = localState;

    // start up the seasonal random walk

    boot_dgp_init(&localState, S, dgp_nlags, arcoefs, csns, eps, dylags, y0);

    // lag order selection

    if (ICtype > 0) 
    {
      //NOTE 'maxlag' is passed as argument; this makes discarding the first 'maxlag'
      //observations (if (i > nlagsm1) is used in 'crossprod_regmatrix');
      //thus, the information criterion calculated for different lag orders employs
      //the same number of observations;
      //after choosing the lag order, 'crossprod_regmatrix' is called again with 
      //before computing the HEGY statistics passing 'nlags' instead of 'maxlag',
      //this will employ 'maxlag-nlags' more observations, which may be relevant 
      //in small samples; 
      //if maxlag == nlags 'crossprod_regmatrix' must be called as well because
      //'vA' is overwritten here

      crossprod_regmatrix(&localState, eps, y0, nx, csns,
        S, idc0, idc1, idc2, maxlag, dgp_nlags, arcoefs, dylags, vA);

      for (int i = 1; i <= S; i++)
      {
        sweep(nvA, Anrnv, i, vA);
      }

      if (Anrnv > S + maxlag + 1)
      for (int i = S+maxlag+1; i < Anrnv; i++)
      {
        sweep(nvA, Anrnv, i, vA);
      }

      //NOTE here the sample size is nr = nx-maxlag for all models to be compared;
      //once a lag order is chosen, the sample size may be nx-lagorder > nx-maxlag,
      //which may be relevant in small samples, therefore the HEGY regression must 
      //be run again in order to update the statistics (instead of obtaining 
      //the statistics within the lag order selection procedure)

      //number of observations employed in each model
      //the same for all models
      int nrCommon = nx - maxlag;

      //store the number of parameters in y0[2], double instead of int but 
      //it will be multiplied by doubles
      y0[2] = Anrnv - maxlag;

      nlags = 0;

      if (ICtype == 1) // AIC
      {
        y0[0] = nrCommon * logf(vA[nvA-1] / nrCommon) + 2.0 * y0[2];

        for (int i = S+maxlag, j=1; i > S; i--, j++)
        {
          y0[2] += 1; // increment number of parameters, one more lag
          sweep(nvA, Anrnv, i, vA);
          y0[1] = nrCommon * logf(vA[nvA-1] / nrCommon) + 2.0 * y0[2];
          if (y0[1] < y0[0])
          {
            y0[0] = y0[1];
            nlags = j;
          }
        }
      } else 
      if (ICtype == 2) // BIC
      {
        y0[0] = nrCommon * logf(vA[nvA-1] / nrCommon) + y0[2] * logf(nrCommon);

        for (int i = S+maxlag, j=1; i > S; i--, j++)
        {
          y0[2] += 1;
          sweep(nvA, Anrnv, i, vA);
          y0[1] = nrCommon * logf(vA[nvA-1] / nrCommon) + y0[2] * logf(nrCommon);
          if (y0[1] < y0[0])
          {
            y0[0] = y0[1];
            nlags = j;
          }
        }
      } else 
      if (ICtype == 3) // AICc
      {
        y0[0] = nrCommon * logf(vA[nvA-1] / nrCommon) + 2.0 * y0[2] + 
          (2.0 * y0[2] * (y0[2]+1)) / (nrCommon - y0[2] - 1.0);

        for (int i = S+maxlag, j=1; i > S; i--, j++)
        {
          y0[2] += 1;
          sweep(nvA, Anrnv, i, vA);
          y0[1] = nrCommon * logf(vA[nvA-1] / nrCommon) + 2.0 * y0[2] + 
            (2.0 * y0[2] * (y0[2]+1)) / (nrCommon - y0[2] - 1.0);
          if (y0[1] < y0[0])
          {
            y0[0] = y0[1];
            nlags = j;
          }
        }
      }

      res6[tid] = nlags;

      //'vA' is not reallocated but the last terms are discarded 
      //by defining the length, 'nvA', according to the chosen 'nlags';
      //remains the same if nlags == maxlag
      //if (nlags != maxlag)
      Anrnv -= (maxlag - nlags);
      // length of the packed A matrix
      nvA = int((Anrnv*(Anrnv + 1)/2));
      for (int i = 0; i < nvA; i++)
        vA[i] = 0.0;

      // reset state of RNG and initial seasonal random walk
      localState = init_localState;
      boot_dgp_init(&localState, S, dgp_nlags, arcoefs, csns, eps, dylags, y0);

    } // end lag order selection
    // else, ICtype == 0, no lag selection, 'nlags' was set to 'maxlag' above

    // HEGY regressors
    // cross-products regression matrix to be swept by columns

    crossprod_regmatrix(&localState, eps, y0, nx, csns,  
      S, idc0, idc1, idc2, nlags, dgp_nlags, arcoefs, dylags, vA);

    // test statistics

    hegy_statistics(vA, nvA, Anrnv, nx - nlags, S, 
      dgp_nlags, arcoefs, res_hegyreg);

    // counter

    //NOTE reuse 'nvA' (here it is no longer the length of vector 'vA')
    nvA = 0;
    if (res_hegyreg[Sh+2] <= stats0[nvA++])
      res1[tid] = 1;
    if (S%2 == 0) {
      if (res_hegyreg[Sh+1] <= stats0[nvA])
        res2[tid] = 1;
      nvA += 1;
    }
    for (int i = Sh+(S%2), j = tid*(Sh-1+(S%2)); i > 1; i--, j++, nvA++)
      if (res_hegyreg[i] >= stats0[nvA])
        res3[j] = 1;
    if (res_hegyreg[1] >= stats0[nvA++])
      res4[tid] = 1;
    if (res_hegyreg[0] >= stats0[nvA])
      res5[tid] = 1;

/*if (tid == debug_tid)
{
nvA = 0;
printf("\ncounter (stat0, bstat, counter)\n");
printf("t_0    = %15.8lf, %15.8lf, %u\n", 
  stats0[nvA++], res_hegyreg[Sh+2], res1[tid]);
if (S%2 == 0)
  printf("t_pi   = %15.8lf, %15.8lf, %u\n", 
    stats0[nvA++], res_hegyreg[Sh+1], res2[tid]);
for (int i = Sh+(S%2), j = tid*(Sh-1+(S%2)); i > 1; i--, j++, nvA++)
  printf("F_pair = %15.8lf, %15.8lf, %u\n", 
    stats0[nvA], res_hegyreg[i], res3[j]);
printf("F_2:S  = %15.8lf, %15.8lf, %u\n", 
  stats0[nvA++], res_hegyreg[1], res4[tid]);
printf("F_1:S  = %15.8lf, %15.8lf, %u\n", 
  stats0[nvA], res_hegyreg[0], res5[tid]);
}*/

    if (dgp_nlags > 0)
      free(dylags);
    free(res_hegyreg);
    free(vA);
    free(y0);

  } // end if (tid < N)
}

void hegy_boot_pval(const int *dim, const unsigned int *seed, 
  const int *idc, const int *ip, const int *csns, const double *eps, const double *arcoefs,
  const double *stats0, const int *ICtype, double *bpvals, double *chosen_lags)
{
  int N = dim[0], nBlocks = dim[1], nThreadsPerBlock = dim[2];
  int S = ip[0], nx = ip[1], // length of the differenced series 
    Sh = int(S/2), Smod2 = S%2, nFpair = Sh-1+Smod2;
  int Nsizeofint = N * sizeof(int);
  int *d_csns, *res1, *res2, *res3, *res4, *res5, 
    *d_res1, *d_res2, *d_res3, *d_res4, *d_res5, *d_res6;

  if (csns[0] == 0)
  {
    cudaMalloc((void**) &d_csns, 2 * sizeof(int));
    cudaMemcpy(d_csns, csns, 2 * sizeof(int), cudaMemcpyHostToDevice);
  } else {
    cudaMalloc((void**) &d_csns, S * sizeof(int));
    cudaMemcpy(d_csns, csns, S * sizeof(int), cudaMemcpyHostToDevice);
  }

  res1 = (int *) calloc(N, sizeof(int));
  res3 = (int *) calloc(N * nFpair, sizeof(int));
  res4 = (int *) calloc(N, sizeof(int));
  res5 = (int *) calloc(N, sizeof(int));
  cudaMalloc((void**) &d_res1, Nsizeofint);
  cudaMalloc((void**) &d_res3, Nsizeofint * nFpair);
  cudaMalloc((void**) &d_res4, Nsizeofint);
  cudaMalloc((void**) &d_res5, Nsizeofint);
  cudaMemcpy(d_res1, res1, Nsizeofint, cudaMemcpyHostToDevice);
  if (Smod2 == 0)
  {
    res2 = (int *) calloc(N, sizeof(int));
    cudaMalloc((void**) &d_res2, Nsizeofint);
    cudaMemcpy(d_res2, res2, Nsizeofint, cudaMemcpyHostToDevice);
  }
  cudaMemcpy(d_res3, res3, Nsizeofint * nFpair, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_res3, res1, sizeof(res3), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res4, res4, Nsizeofint, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res5, res5, Nsizeofint, cudaMemcpyHostToDevice);

  if (*ICtype > 0)
  {
    cudaMalloc((void**) &d_res6, Nsizeofint);
    cudaMemcpy(d_res6, chosen_lags, Nsizeofint, cudaMemcpyHostToDevice);
  } 
  
  double *d_eps, *d_arcoefs, *d_stats0;
  cudaMalloc((void**) &d_eps, sizeof(double) * nx);
  cudaMemcpy(d_eps, eps, sizeof(double) * nx, cudaMemcpyHostToDevice);

  cudaMalloc((void**) &d_arcoefs, sizeof(double) * ip[2]);
  cudaMemcpy(d_arcoefs, arcoefs, sizeof(double) * ip[2], cudaMemcpyHostToDevice);

  cudaMalloc((void**) &d_stats0, sizeof(double) * (Sh+3));
  cudaMemcpy(d_stats0, stats0, sizeof(double) * (Sh+3), cudaMemcpyHostToDevice);

  kernel<<<nBlocks, nThreadsPerBlock>>>(N, *seed, d_eps, S, idc[0], idc[1], idc[2],
    nx, d_csns, ip[2], d_arcoefs, d_stats0, *ICtype, ip[3], ip[4],
    d_res1, d_res2, d_res3, d_res4, d_res5, d_res6);

  cudaDeviceSynchronize();

  cudaMemcpy(res1, d_res1, Nsizeofint, cudaMemcpyDeviceToHost);
  if (Smod2 == 0)
    cudaMemcpy(res2, d_res2, Nsizeofint, cudaMemcpyDeviceToHost);
  cudaMemcpy(res3, d_res3, Nsizeofint * nFpair, cudaMemcpyDeviceToHost);
  cudaMemcpy(res4, d_res4, Nsizeofint, cudaMemcpyDeviceToHost);
  cudaMemcpy(res5, d_res5, Nsizeofint, cudaMemcpyDeviceToHost);
  if (*ICtype > 0)
    cudaMemcpy(chosen_lags, d_res6, Nsizeofint, cudaMemcpyDeviceToHost);

  // bootstrapped p-values

  bpvals[0] = 1.0 * thrust::reduce(thrust::host, res1, res1 + N) / double(N);
  if (Smod2 == 0)
    bpvals[1] = 1.0 * thrust::reduce(thrust::host, res2, res2 + N) / double(N);
  for (int j = 0; j < N; j++)
    for (int i = 2-Smod2, k=j*nFpair; i < nFpair+2-Smod2; i++, k++)
      bpvals[i] += 1.0 * res3[k];
  for (int i = 2-Smod2; i < nFpair+2-Smod2; i++)
    bpvals[i] = 1.0 * bpvals[i] / double(N);
  bpvals[Sh+1] = 1.0 * thrust::reduce(thrust::host, res4, res4 + N) / double(N);
  bpvals[Sh+2] = 1.0 * thrust::reduce(thrust::host, res5, res5 + N) / double(N);

  free(res1);
  if (Smod2 == 0)
    free(res2);
  free(res3);
  free(res4);
  free(res5);
  cudaFree(d_arcoefs);
  cudaFree(d_csns);
  cudaFree(d_eps);
  cudaFree(d_res1);
  if (Smod2 == 0)
    cudaFree(d_res2);
  cudaFree(d_res3);
  cudaFree(d_res4);
  cudaFree(d_res5);
  if (*ICtype > 0)
    cudaFree(d_res6);
  cudaFree(d_stats0);
}
