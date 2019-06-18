# Process this file with autoconf to produce a configure script.

AC_INIT([uroot], [2.0-9])
AC_CONFIG_AUX_DIR([tools])
##AM_INIT_AUTOMAKE

AC_CONFIG_MACRO_DIR([tools])

#./configure
#./configure R_HOME=/usr/local/R-devel/lib/R/
#./configure R_HOME=/usr/local/R-devel/lib/R/ RINCL=-I${R_HOME}/include

AC_ARG_VAR([R_HOME],
           [R home directory])

AC_ARG_VAR([R_INCL],
           [R include path])

AC_ARG_VAR([CUDA_HOME],
           [CUDA home path])

AC_ARG_VAR([CUDA_LD],
           [CUDA library path])

AC_MSG_CHECKING([R home directory])
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
R_INCL=`"${R_HOME}/bin/R" CMD config --cppflags`
#CC=`"${R_HOME}/bin/R" CMD config CC`
AC_MSG_RESULT([${R_HOME}])

AC_CHECK_FILE(${R_HOME}/include/R.h, 
	# guess if file exists: R was not buit as a library (e.g., on server guinness ARINA)
	[R_INCL=-I${R_HOME}/include],
	[R_INCL=`"${R_HOME}/bin/R" CMD config --cppflags`])
AC_MSG_CHECKING([R include path])
AC_MSG_RESULT([${R_INCL}])

AC_MSG_CHECKING([CUDA_HOME environment variable])
if test -z "${CUDA_HOME}"; then
  CUDA_HOME=`find /usr/local/ -maxdepth 1 -type d -name "cuda*" | sort -V | tail -1`
  AC_MSG_RESULT([not set])
  #highest version found ${CUDA_HOME}
else
  AC_MSG_RESULT([${CUDA_HOME}])
fi

AC_MSG_CHECKING([CUDA home path])
if test -z "${CUDA_HOME}"; then
  AC_MSG_RESULT([not found])
else
  AC_MSG_RESULT([${CUDA_HOME}])
fi

AC_MSG_CHECKING([setting CUDA library path])
if test -z "${CUDA_HOME}"; then
  PKG_LIBS=""
  AC_MSG_RESULT([not set])
else
  CUDA_LD=-L${CUDA_HOME}/lib64
  PKG_LIBS="${CUDA_LD} -lcudart -lcuda -lcublas -lcudadevrt -Wno-deprecated-gpu-targets"
  AC_MSG_RESULT([${CUDA_LD}])
fi

AC_MSG_CHECKING([whether CUDA is available])
if test -z "${CUDA_HOME}"; then
  HAS_CUDA=FALSE
  CUDA_OBJECTS=""
  AC_MSG_RESULT([no])
else
  HAS_CUDA=TRUE
  CUDA_OBJECTS="dev-fnc.o kernel.o gpu.o"
  OBJ_AUX1="dev-fnc.o"
  OBJ_AUX2="dev-fnc.o kernel.o"
  AC_MSG_RESULT([yes])
fi

# Libtool
#LT_INIT

#AC_PROG_CXX
AC_PROG_INSTALL
#AC_SUBST(R_HOME)
AC_SUBST(R_INCL)
AC_SUBST(CUDA_HOME)
AC_SUBST(PKG_LIBS)
AC_SUBST(HAS_CUDA)
AC_SUBST(CUDA_OBJECTS)
AC_SUBST(OBJ_AUX1)
AC_SUBST(OBJ_AUX2)
AC_CONFIG_FILES([src/Makevars R/hegy-test.R])
AC_OUTPUT
