/**************************************************************************
* FILE NAME: run_gmres.h                                                  *
*                                                                         *
* PURPOSE: header for GMRes routine wrapper, used by tabipb routine       *
*                                                                         *
* AUTHORS: Leighton Wilson, University of Michigan, Ann Arbor, MI         *
*          Jiahui Chen, Southern Methodist University, Dallas, TX         *
*                                                                         *
* BASED ON PACKAGE ORIGINALLY WRITTEN IN FORTRAN BY:                      *
*          Weihua Geng, Southern Methodist University, Dallas, TX         *
*          Robery Krasny, University of Michigan, Ann Arbor, MI           *
*                                                                         *
**************************************************************************/

#ifndef H_RUN_GMRES_H
#define H_RUN_GMRES_H

#include "struct_particles.h"

int RunGMRES(int nface, double *source_term, int precond,
             double *xvct, long int *iter, struct Particles *particles);

#endif /* H_RUN_GMRES_H */
