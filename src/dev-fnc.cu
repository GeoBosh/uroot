#include "dev-fnc.h"

__device__ void boot_dgp_init(curandState *RNG_state, const int S, 
  const int dgp_nlags, const double *arcoefs, const int *csns, 
  const double *eps, double *dylags, double *y0)
{
  // generate seasonal random walk (the beginning of the series)

  for (int i = 0, id, sid = 0; i < S + dgp_nlags; i++, sid = (sid==S-1 ? 0 : sid+1))
  {
    if (csns[0] == 0)
    {
      id = int(curand_uniform(RNG_state) * csns[1]);
      y0[i] = eps[id];
    } else {
      if (sid == 0) {
        id = int(curand_uniform(RNG_state) * csns[0]);
      } else {
        id = int(curand_uniform(RNG_state) * (csns[sid] - csns[sid-1])) + csns[sid-1];
      }
      y0[i] = eps[id];
    }
  }

  if (dgp_nlags == 0)
  {
    // accumulate here (no warming period is used)
    for (int i = 0, sid = 0, id; i < S; i++, sid = (sid==S-1 ? 0 : sid+1))
    {
      if (csns[0] == 0) // resample the entire series of residuals
      {
        id = int(curand_uniform(RNG_state) * csns[1]);

      } else // resample the residuals by season
      {
        if (sid == 0) {
          id = int(curand_uniform(RNG_state) * csns[0]);
        } else {
          id = int(curand_uniform(RNG_state) * (csns[sid] - csns[sid-1])) + csns[sid-1];
        }
      }

      y0[i] += eps[id];
    }
  } else // dgp_nlags > 0
  {
    for (int i = 0, j = dgp_nlags-1; i < dgp_nlags; i++, j--)
      dylags[i] = y0[j+S] - y0[j];

    // accumulate here (no warming period is used)
    for (int i = dgp_nlags, sid = (dgp_nlags % S), id; i < S + dgp_nlags; 
      i++, sid = (sid==S-1 ? 0 : sid+1))
    {
      if (csns[0] == 0)
      {
        id = int(curand_uniform(RNG_state) * csns[1]);
      } else {
        if (sid == 0) {
          id = int(curand_uniform(RNG_state) * csns[0]);
        } else {
          id = int(curand_uniform(RNG_state) * (csns[sid] - csns[sid-1])) + csns[sid-1];
        }
      }

      double tmp;
      tmp = y0[i];
      for (int j = 0, l = 1; j < dgp_nlags; j++, l++)
      {
        tmp += arcoefs[j] * dylags[j];
      }
      tmp += eps[id];

      if (i == dgp_nlags)
      for (int j = 0; j < dgp_nlags; j++)
        y0[j] = y0[j+S];

      for (int j = dgp_nlags-1; j > 0; j--) {
        dylags[j] = dylags[j-1];
      }

      dylags[0] = tmp - y0[i];
      y0[i] = tmp;
    }
  }
}

__device__ void crossprod_regmatrix(curandState *RNG_state, const double *eps, 
  double *y0, const int nx, 
  const int *csns, const int S, 
  const int idc0, const int idc1, const int idc2, const int nlags, 
  const int dgp_nlags, const double *arcoefs, 
  double *dylags, double *res)
{
  //NOTE "res" is passed already initialized to zeros, this way no need to pass 
  //as input the number of elements in array "res"

  int k, Sm1 = S-1, isSeven = S % 2, Sheo = (isSeven == 0) ? S/2 : S/2 + 1, 
    nlagsm1 = nlags - 1, ip1, y0id; //nmnlags = n - nlags
  //NOTE S/2 is integer division (if S is odd, the decimal part is discarded)
  double PIoSh = M_PI / (S / 2.0);

  double *dy = (double *)malloc((nlags + 1) * sizeof(double));
  double *ypi_row = (double *)malloc(S * sizeof(double));

  for (int j = 0; j < S; j++)
  {
    ypi_row[j] = 0.0;
  }

  //NOTE "sid" is used only if idc2==1 (or with csns, devel)

  for (int i = 0, sid = (dgp_nlags % S), Sref = Sm1; i < nx; i++, sid = (sid==Sm1 ? 0 : sid+1))
  {
    double w = 0.0;
    int sign = 1.0;

    //NOTE this is done at the end of this loop within another loop
    //for (int j = 0; j < S; j++)
    //  ypi_row[j] = 0.0;

    for (int j = 0; j < S; j++)
    {
      k = 1;
    
      y0id = (dgp_nlags == 0 ? Sref : Sref + dgp_nlags);

      ypi_row[0] += y0[y0id];

      if (isSeven == 0)
      {
        sign = -1.0 * sign;
        ypi_row[k++] += sign * y0[y0id];
      }
      
      w += PIoSh;

      ypi_row[k++] += y0[y0id] * cos(w);
      ypi_row[k++] += y0[y0id] * sin(w);
      
      if (S > 4)
      {
        double wxl = w;
        for (int l = 2; l < Sheo; l++)
        {
          wxl += w;
          ypi_row[k++] += y0[y0id] * cos(wxl);
          ypi_row[k++] += y0[y0id] * sin(wxl);
        }
      }

      Sref = Sref > 0 ? Sref - 1 : Sm1;
      if (Sref == Sm1)
      {
        if (nlags > 0)
          for (int l = 0; l < nlags; l++)
            dy[l] = dy[l+1];

        if (csns[0] == 0)
        {
          int id = int(curand_uniform(RNG_state) * csns[1]);
          dy[nlags] = eps[id];
        } else 
        {
          if (sid == 0)
          {
            int id = int(curand_uniform(RNG_state) * csns[0]);
            dy[nlags] = eps[id];
          } else {
            int id = int(curand_uniform(RNG_state) * (csns[sid] - csns[sid-1])) + csns[sid-1];
            dy[nlags] = eps[id];
          }     
        }

        if (dgp_nlags > 0)
        {
          for (int l = 0, j=dgp_nlags-1; l < dgp_nlags; l++, j--)
          {
            dy[nlags] += arcoefs[l] * dylags[l];
          }
          for (int j = dgp_nlags-1; j > 0; j--)
          {
            dylags[j] = dylags[j-1];
          }
          dylags[0] = dy[nlags];
        }

        double tmp = dgp_nlags == 0 ? y0[0] + dy[nlags] : y0[dgp_nlags] + dy[nlags];

        if (dgp_nlags == 0)
        {
          for (int l = 0; l < Sm1; l++)
            y0[l] = y0[l+1];
          y0[Sm1] = tmp;
        } else { // dgp_nlags > 0
          for (int l = dgp_nlags; l < Sm1+dgp_nlags; l++)
            y0[l] = y0[l+1];
          y0[Sm1+dgp_nlags] = tmp;
        }

      } // end if (Sref == Sm1)

    } // end loop for j = 0 to j < S

    // crossproducts

    if (i > nlagsm1)
    {
      //NOTE "k" must be defined outside the loop so that its value remains available 
      //when setting res[k] after this loop
      //(this way avoids calculating or passing as input the length of "res")

      if (idc1 == 1)
        ip1 = i+1;
      
      k = 0;
      for (int j = 0; j < S; j++)
      {
        for (int l = j; l < S; l++)
        {
          res[k++] += ypi_row[j] * ypi_row[l];
        }

        for (int ilags = 0; ilags < nlags; ilags++)
        {
          res[k++] += ypi_row[j] * dy[ilags];
        }

        if (idc0 == 1)
        {
          res[k++] += ypi_row[j];
        }
        
        if (idc1 == 1)
        {
          res[k++] += ypi_row[j] * ip1;
        }
        
        if (idc2 == 1)
        {
          // from l=1 (not l=0), first seasonal dummy discarded
          // to avoid collinearity with intercept
          for (int l = 1; l < S; l++)
          {
            if (l == sid)
              res[k] += ypi_row[j];
            k += 1;
          }
        }

        res[k++] += ypi_row[j] * dy[nlags];
        // reset to zero for the next run of this loop
        ypi_row[j] = 0.0;
      }

      for (int ilags = 0; ilags < nlags; ilags++)
      {
        for (int l = ilags; l < nlags; l++)
        {
          res[k++] += dy[ilags] * dy[l];
        }

        if (idc0 == 1)
          res[k++] += dy[ilags];
        
        if (idc1 == 1)
          res[k++] += dy[ilags] * ip1;

        if (idc2 == 1)
        for (int s = 1; s < S; s++)
        {
          if (s == sid)
            res[k] += dy[ilags];
          k += 1;
        }

        res[k++] += dy[ilags] * dy[nlags];
      }

      if (idc0 == 1)
      {
        // skip this element, which will be set to "n-nlags" outside the loop
        k += 1;

        if (idc1 == 1)
          res[k++] += ip1;

        // skip, done outside the loop (the sums are n/S)
        if (idc2 == 1)
          k += Sm1;

        res[k++] += dy[nlags];
      }

      if (idc1 == 1)
      {
        res[k++] += ip1 * ip1;

        if (idc2 == 1)
        for (int s = 1; s < S; s++)
        {
          if (s == sid)
            res[k] += ip1;
          k += 1;
        }

        res[k++] += dy[nlags] * ip1;
      }

      if (idc2 == 1)
      {
        //NOTE do this here; the crossproducs of seasonal dummies could be 
        //done outside the loop but not much advantage since this loop from s=1 to S
        //is required anyway to add upp dy[nlags]

        for (int s = 1; s < S; s++)
        {
          if (s == sid)
            res[k] += 1.0;
          k += S-s;
          
          if (s == sid)
            res[k] += dy[nlags];
          k += 1;
        }
      }

      res[k] += dy[nlags] * dy[nlags];

    } else { // i <= nlagsm1
        // reset to zero for the next run of this loop
        for (int j = 0; j < S; j++)
          ypi_row[j] = 0.0;
    }

  } // end loop from i = 0 to i < n

  if (idc0 == 1)
  {
    int a = idc0 + idc1 + idc2*Sm1 + 1;
    int id0 = a + nlags + S;
    int id = ((id0*id0 + id0) - (a*a + a))/2;
    res[id] = double(nx-nlags);
    
    if (idc2 == 1)
    {
      //NOTE the crossproducts of intercept with SD are taken from the crossproducts of SD
      //obtained in the loop above, this way it is simpler to deal with the case where 
      //the number of observations is not multiple of S

      id = id + idc1 + 1;
      for (int s = 0, id2 = id + idc1 + S*(idc1+1); s < Sm1; s++, id++, id2 += S + 1 - s)
      {
        res[id] = res[id2];
      }
    }
  }

  free(dy);
  free(ypi_row);
}

__device__ void sweep(const int nv, const int nc, const int idcol, double *v)
{
  //NOTE this version applies the sweep operator one column at a time, i.e., 
  //if columns 1 and 2 are to be swept this function must be called twice with 
  //"idcol" argument equal to 1 and 2; 

  //NOTE idcol=1 is used within lag selection procedures
  //for the calculation of statistics idcol=1 is not reached, 
  //since in the last sweep we are interested only in the element related 
  //to RSS and the sign of the coefficients related to t0 and tpi and, hence,
  //it is not needed to do all the operations in the sweep

  if (idcol == 1)
  {
    int jref = nv - 1, kref2 = nc;

    for (int j = 1; j < nc; j++)
    {
      kref2 -= 1;
      for (int k = 1; k <= j; k++)
      {
        v[jref] -= (v[kref2] * v[nc-k] / v[0]);
        jref -= 1;
      }
    }

    for (int k = 1; k < nc; k++)      
    {
      v[k] = v[k] / v[0];
    }

    v[0] = -1.0 / v[0];

    return;
  }

  int idpivot = ((idcol-1) * nc + idcol) - ((idcol*idcol+idcol)/2 - idcol) - 1,
    jump = nc + 2 - idcol, nir = jump - 2;
  //k0 is the index of the last element in a row in terms of the packed vector, column m[,n]
  int k0 = idcol * nc - ((idcol*idcol+idcol)/2 - idcol) - 1, kref = k0 + 1, 
    kref2 = k0 + 1, jref = nv - 1;

  for (int j=1; j <= nir; j++)
  {
    kref2 -= 1;
    for (int k = 1; k <= j; k++)
    {
      v[jref] -= (v[kref2] * v[kref-k] / v[idpivot]);
      jref -= 1;        
    }      
  }

  int jref0 = idpivot + nir;

  for (int j = nir + 1; j < nc; j++)
  {
    kref2 -= jump;
    if (j > nir + 2)
      kref2 -= (j - nir - 2);

    jref0 -= j;
    jref = jref0;
    
    for (int k = 1; k <= nir; k++)
    {
      v[jref] -= (v[kref2] * v[k0+1-k] / v[idpivot]);
      jref -= 1;
    }

    // omit the element belonging to the pivoting column
    jref -= 1;

    kref = k0 + 1 - nir;

    for (int k = nir + 1; k <= j; k++)
    {
      kref = kref - jump;
      if (k > nir + 2)
        kref -= (k - nir - 2);
      v[jref] -= (v[kref2] * v[kref] / v[idpivot]);
      jref -= 1;
    }
  } // end for j in nir + 1 to nc - 1

  kref = k0 + 1;
  for (int k = 0; k < nir; k++)
  {
    kref -= 1;
    v[kref] /= v[idpivot];
  }

  v[kref-jump] /= v[idpivot];

  if (idcol > 2)
  {
    v[kref-2*jump] /= v[idpivot];
  }

  if (idcol > 3)
  {
    int a = kref - 2*jump;
    for (int k = 1; k < idcol - 2; k++)
    {
      a -= (jump + k);
      v[a] /= v[idpivot];
    }
  }

  v[idpivot] = -1.0 / v[idpivot];
}

__device__ void hegy_statistics(double *v, const int nv, const int nc, const int nr, 
  const int S, const int dgp_nlags, const double *arcoefs, double *res)
{
  //NOTE nc is the number of columns in the square matrix of crossproducts, 
  //thus it includes the crossproduct with the dependent variable and therefore 
  //the number of regressors is nc-1

  int nvm1 = nv - 1, ncm1 = nc - 1, ncm1x2 = ncm1*2, isSeven = S % 2, Sh = int(S/2);
  double tmp1, tmp2, *v_backup;
  if (S > 4)
    v_backup = (double *)malloc(nv * sizeof(double));

  if (nc > S + 1)
  for (int i = nc - 1; i > S; i--)
  {
    sweep(nv, nc, i, v);
  }

  // residual sum of squares in restricted models

  //F1:S

  res[0] = v[nvm1];

  //F2:S:

  res[1] = v[nvm1] - v[ncm1] * v[ncm1] / v[0];

  //Fpair (F34 if S is even; F23 if S is odd)

  if (S > 4)
  {
    for (int i = 0; i < nv; i++)
      v_backup[i] = v[i];

    for (int ifpair = S, j=2; ifpair > 3-isSeven; ifpair-=2, j++)
    {
      for (int i = ifpair; i > 3-isSeven; i-=2)
      {
        if (i != ifpair)
        {
          sweep(nv, nc, i, v);
          sweep(nv, nc, i-1, v);
        }
      }

      if (isSeven == 0)
      {
        tmp1 = v[nc] - v[1] * v[1] / v[0];
        tmp2 = v[ncm1x2] - v[1] * v[ncm1] / v[0];
        res[j] = (v[nvm1] - v[ncm1] * v[ncm1] / v[0]) - tmp2 * tmp2 / tmp1;
      } else {
        // avoid entire sweep
        res[j] = v[nvm1] - v[ncm1] * v[ncm1] / v[0];
      }

      sweep(nv, nc, ifpair, v_backup);
      sweep(nv, nc, ifpair-1, v_backup);

      for (int i = 0; i < nv; i++)
        v[i] = v_backup[i];

    } // end loop ifpair

  } else { // S==4
    tmp1 = v[nc] - v[1] * v[1] / v[0];
    tmp2 = v[ncm1x2] - v[1] * v[ncm1] / v[0];

    res[2] = res[1] - tmp2 * tmp2 / tmp1;
  }

  if (S == 4)
  {
    sweep(nv, nc, 4 - isSeven, v);
    sweep(nv, nc, 3 - isSeven, v);
  }

  // tpi
  
  if (isSeven == 0)
    res[Sh+1] = v[nvm1] - v[ncm1] * v[ncm1] / v[0];

  // t0

  if (isSeven == 0) {
    res[Sh+2] = v[nvm1] - v[ncm1x2] * v[ncm1x2] / v[nc];
  } else {
    res[Sh+2] = v[nvm1];
  }

  if (isSeven == 0)
    sweep(nv, nc, 2, v);

  // residual sum of squares of unrestricted model (reuse "tmp1")

  tmp1 = v[nvm1] - v[ncm1] * v[ncm1] / v[0];
  tmp2 = tmp1 / (nr - ncm1);

  // test statistics

  //F1:S

  res[0] = ((res[0] - tmp1) / S) / tmp2;

  //F2:S

  res[1] = ((res[1] - tmp1) / (S - 1)) / tmp2;

  //Fpair

  for (int i = 2; i < Sh+1+isSeven; i++)
    res[i] = ((res[i] - tmp1) / 2) / tmp2;

  //tpi

  if (isSeven == 0)
    res[Sh+1] = sqrt((res[Sh+1] - tmp1) / tmp2);

  //t0

  res[Sh+2] = sqrt((res[Sh+2] - tmp1) / tmp2);
  
  //sign of t-statistics  

  //tpi
  
  if (isSeven == 0)
  {
    tmp1 = v[ncm1x2] - v[1] * v[ncm1] / v[0];
    if (tmp1 < 0)
      res[Sh+1] = -1.0 * res[Sh+1];
  }
  
  //t0
  
  tmp1 = v[ncm1] / v[0];
  if (tmp1 < 0)
    res[Sh+2] = -1.0 * res[Sh+2];

  if (S > 4)
    free(v_backup);
}
