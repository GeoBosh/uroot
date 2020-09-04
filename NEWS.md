# uroot 2.1-2 2020-09-04

* changed http to https in the nvidia's cuda link.


# uroot 2.1-1 2020-03-26

* corrected a wrong value in the internal dataset `.CH.orig.cv`. Element
  `uroot:::.CH.orig.cv[12, 6]` was `2.510`, now it is `3.510` as in the original
  source (Canova and Hansen 1995, Table 1), reported by Javier Lopez de Lacalle.

* created website (using `pkgdown`).

* Renamed  NEWS and README and added some markdown markup, so that they play
  better with `pkgdown`. Added some links to README and DESCRIPTION. 


# uroot 2.1-0 2019-08-19

* removed all stuff related to CUDA, since the configuration files
  need more than the tweeks by the current maintainer in versions
  2.0.10 - 2.0.11.


# uroot 2.0-11 2019-08-18

* in configuration files, replaced non-POSIX options '-maxdepth' (find) and '-V'
  (sort) with equivalent POSIX approximations.


# uroot 2.0-10 2019-06-17

* new maintainer (Georgi N. Boshnakov).

* changed the configuration files to pass cleanly CRAN checks.  The changes are
  minimal and mainly concern the non-GPU compilation and symbol registration.  I
  didn't make any structural changes except that now the default installation is
  without GPU support, even if it seems present (set environment variable
  CUDA_IGNORE to force using the GPU). The main reason for this is that the
  compilation may fail since the nvcc compiler has rather strict system
  requirements, which are different for different combinations of versions of
  nvcc, flavours of OS, and OS versions. These may conflict with the tools used
  by the R tool chain.  Currently, as in the previous version of 'uroot',
  installation of GPU code on Windows is not attempted.

* deactivated Rd documentation links to the author's website, www.jalobe.com,
  since it seems unreachable. The vignette is unchanged though.


# uroot 2.0-9 2017-01-27

* Further improvements to make configure portable. 

    Following the suggestion by Kurt Hornik, I went back to the 
    initial idea of defining a rule `%cu : %.o` (as those defined in    
    `usr/lib/R/etc/Makeconf`) and let `Makeconf` do the rest.
    The problem I found before was that the default linker (e.g. `g++`) 
    couldn't link objects compiled with `nvcc`. Now, I have found that 
    a second stage `nvcc -dlink` is needed in order to generate 
    objects to be linked by a linker other than nvcc.
    Required flags are passed to the linker via `PKG_LIBS`.

    When cuda files are compiled, R CMD reports 
    "Found ‘exit’, possibly from ‘exit’ (C)"
    This may require some fix in the future; currently 
    no issue was observed when using the package.


# uroot 2.0-8 2017-01-24

* Configure failed on some systems. As reported by CRAN maintainer:

```
    on Fedora we see

    ** libs
    g++ -std=gnu++98 -shared -L/usr/local/lib64 -o uroot.so
    g++: fatal error: no input files

    and the corresponding `.mk` with `uroot`:

    $(CXX) -shared $(LDFLAGS) -o uroot.so

    is simply invalid: not only are there no source files, `-shared` is not
    portable and `CXXFLAGS` and `CXXPICFLAGS` are missing.
```

  Current solution: define a NAMESPACE.in where `"dynload(uroot)"` is 
  included at the configure step depending on whether CUDA is detected or not.
  `configure.win` remains as before (remove directory 'src', as I cannot check
  the configure step on a Windows system with CUDA).
  As reported by `R CMD check`, "NAMESPACE.in" is a non-standard file, 
  but it seems the easiest way to avoid a dummy `.so` file.


# uroot 2.0-7 2017-01-22

* The configure script has been modified slightly in the hope 
    to be more portable. 
    
* The flag -lR has been removed when the dummy uroot.so file is 
    generated (when CUDA is not available).

* `configure.win` has been simplified. CUDA files are ignored because 
  I couldn't test it on a windows system with a CUDA enabled GPU.

* Added document uroot-intro.pdf with links to further documentation.


# uroot 2.0-6 2017-01-05

* A cleanup script has been added.

* GNU make is no longer a SystemRequirements.
    Include files are used in order to avoid the 'ifeq' GNU make extension,
    this makes Makevars more portable.

* The configure script creates now appropriate environment variables 
    when it detects that it is running on the ARINA cluster of the UPV/EHU.


# uroot 2.0-5 2016-03-18

* Based on an older version of package uroot.

  This version provides the CH test for seasonal stability and the HEGY test for
  seasonal unit roots. The functions that implement these tests have been coded
  from the scratch in order to include the following new features:
    
  1) the tests are applicable to series of any seasonal periodicity,

  2) p-values based on response surface regressions are available for both tests,

  3) bootstrapped p-values are available for the HEGY test.
