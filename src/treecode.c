/**************************************************************************
* FILE NAME: treecode.c                                                   *
*                                                                         *
* PURPOSE: Contains all treecode-related functions and variables,         *
*          including treecode initialization and finalization functions   *
*          that interface with tabipb.c, and matrix-vector multiplication *
*          and solve functions that interface with run_gmres.c            *
*                                                                         *
* AUTHORS: Leighton Wilson, University of Michigan, Ann Arbor, MI         *
*          Jiahui Chen, Southern Methodist University, Dallas, TX         *
*                                                                         *
* BASED ON PACKAGE ORIGINALLY WRITTEN IN FORTRAN BY:                      *
*          Weihua Geng, Southern Methodist University, Dallas, TX         *
*          Robery Krasny, University of Michigan, Ann Arbor, MI           *
*                                                                         *
**************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

//#include <Accelerate/Accelerate.h>
//#include <lapacke.h>
#include <mkl_lapacke.h>

#ifdef MPI_ENABLED
    #include <mpi.h>
#endif

#include "treecode_tabipb_interface.h"
#include "treecode_gmres_interface.h"
#include "utilities.h"

#include "global_params.h"
#include "array.h"

#include "tree/struct_tree_linked_list_node.h"
#include "tree/tree_linked_list.h"

#include "struct_particles.h"
#include "TABIPBstruct.h"


/* runtime treecode parameters */
static int s_numpars;
static int s_order;
static int s_max_per_leaf;
static double theta;

/* runtime physical parameters */
static double s_kappa;
static double s_kappa2;
static double s_eps;

/* variables for tracking tree information */
static int s_min_level;
static int s_max_level;

/* global variables for taylor expansions */
static int s_torder_lim;
static int s_torder3;

/* variable used by kernel independent moment computation */
double *tt, *ww;

/* these point to arrays located in TreeParticles */
static double *s_particle_position_x = NULL;
static double *s_particle_position_y = NULL;
static double *s_particle_position_z = NULL;

static double *s_particle_normal_x = NULL;
static double *s_particle_normal_y = NULL;
static double *s_particle_normal_z = NULL;

static double *s_particle_area = NULL;
static double *s_source_term = NULL;

/* global variables used when computing potential/force */
static double s_target_position[3];
static double s_target_normal[3];

static double **s_target_charge = NULL;
static double **s_source_charge = NULL;

/* global variables for reordering arrays */
//static int *s_order_arr = NULL;

/* root node of tree */
static struct TreeLinkedListNode *s_tree_root = NULL;


/* internal functions */
static int s_Setup(double xyz_limits[6], struct Particles *particles);
static int s_ComputePBKernel(double *phi);
static int s_ComputeAllMoments(struct TreeLinkedListNode *p, int ifirst);
static int s_ComputeMoments(struct TreeLinkedListNode *p);
static int s_RunTreecode(struct TreeLinkedListNode *p, double *tpoten_old,
                         double tempq[4], double peng[2]);
static int s_ComputeTreePB(struct TreeLinkedListNode *p, double tempq[4], double peng[2]);
static int s_ComputeDirectPB(int ibeg, int iend, double *tpoten_old,
                             double peng[2]);
                             
static int s_RemoveMoments(struct TreeLinkedListNode *p);

/* internal preconditioning functions */
static void leaflength(struct TreeLinkedListNode *p, int idx, int *nrow);
static int lu_decomp(double **A, int N, int *ipiv);
static void lu_solve(double **matrixA, int N, int *ipiv, double *rhs);



/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* TreecodeInitialization and Finalization are used by       * * * */
/* tabipb() to interface with the treecode                   * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**********************************************************/
int TreecodeInitialization(TABIPBparm *parm, struct Particles *particles)
{
    /* set up variables used in treecode */
    /* local variables*/
    int level, i, j, k, mm, nn, idx, ijk[3], ierr;

    /* variables needed for reorder */
    double *temp_area, *temp_source;
    double **temp_normal;
    
    double xyz_limits[6];
    
    int rank = 0, num_procs = 1;
    
#ifdef MPI_ENABLED
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

    if (rank == 0) {
        printf("\nInitializing treecode...\n");
    }

    /* setting variables global to file */
    s_numpars = particles->num;
    s_order = parm->order;
    s_max_per_leaf = parm->maxparnode;
    theta = parm->theta;
    
    s_kappa = parm->kappa;
    s_kappa2 = parm->kappa2;
    s_eps = parm->eps;
    
    s_torder_lim = s_order+1;
    s_torder3 = s_torder_lim * s_torder_lim * s_torder_lim;
    
    s_min_level = 50000;
    s_max_level = 0;

    level = 0;

    
    s_particle_position_x = particles->x;
    s_particle_position_y = particles->y;
    s_particle_position_z = particles->z;
    
    s_particle_normal_x = particles->nx;
    s_particle_normal_y = particles->ny;
    s_particle_normal_z = particles->nz;
    
    s_particle_area = particles->area;
    s_source_term = particles->source_term;

    make_matrix(temp_normal, 3, s_numpars);
    make_vector(temp_area, s_numpars);
    make_vector(temp_source, 2 * s_numpars);
    

/* Call SETUP to allocate arrays for Taylor expansions */
/* and setup global variables. Also, copy variables into global copy arrays. */
    s_Setup(xyz_limits, particles);
    
    int numnodes, numleaves, max_depth;
    
    TreeLinkedList_Construct(&s_tree_root, NULL, particles, 0, particles->num-1,
                s_max_per_leaf, xyz_limits, &numnodes, &numleaves,
                &s_min_level, &s_max_level, &max_depth, 0);
    
    if (rank == 0) {
        printf("Created tree for %d particles with max %d per node.\n\n",
               s_numpars, s_max_per_leaf);
    }

    memcpy(temp_normal[0], s_particle_normal_x, s_numpars*sizeof(double));
    memcpy(temp_normal[1], s_particle_normal_y, s_numpars*sizeof(double));
    memcpy(temp_normal[2], s_particle_normal_z, s_numpars*sizeof(double));
    memcpy(temp_area, s_particle_area, s_numpars*sizeof(double));
    memcpy(temp_source, s_source_term, 2*s_numpars*sizeof(double));
    
    for (i = 0; i < s_numpars; i++) {
        s_particle_normal_x[i]    = temp_normal[0][particles->order[i]];
        s_particle_normal_y[i]    = temp_normal[1][particles->order[i]];
        s_particle_normal_z[i]    = temp_normal[2][particles->order[i]];
        s_particle_area[i]           =   temp_area[particles->order[i]];
        s_source_term[i]             = temp_source[particles->order[i]];
        s_source_term[i + s_numpars] = temp_source[particles->order[i] + s_numpars];
    }

    free_matrix(temp_normal);
    free_vector(temp_area);
    free_vector(temp_source);

    make_matrix(s_target_charge, s_numpars, 4);
    make_matrix(s_source_charge, s_numpars, 4);

    return 0;
}
/**********************************************************/


/********************************************************/
int TreecodeFinalization(struct Particles *particles)
{

    int i, ierr;
    double *temp_area, *temp_source, *temp_xvct;
    double **temp_normal, **temp_position;
    
    int rank = 0, num_procs = 1;
    
#ifdef MPI_ENABLED
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif

/***********reorder particles*************/

    make_matrix(temp_position, 3, s_numpars);
    make_matrix(temp_normal, 3, s_numpars);
    make_vector(temp_area, s_numpars);
    make_vector(temp_source, 2 * s_numpars);
    make_vector(temp_xvct, 2 * s_numpars);

    memcpy(temp_position[0], particles->x, s_numpars*sizeof(double));
    memcpy(temp_position[1], particles->y, s_numpars*sizeof(double));
    memcpy(temp_position[2], particles->z, s_numpars*sizeof(double));
    memcpy(temp_normal[0], particles->nx, s_numpars*sizeof(double));
    memcpy(temp_normal[1], particles->ny, s_numpars*sizeof(double));
    memcpy(temp_normal[2], particles->nz, s_numpars*sizeof(double));
    memcpy(temp_area, particles->area, s_numpars*sizeof(double));
    memcpy(temp_source, particles->source_term, 2*s_numpars*sizeof(double));
    memcpy(temp_xvct, particles->xvct, 2*s_numpars*sizeof(double));
    
    for (i = 0; i < s_numpars; i++) {
        particles->x[particles->order[i]]     = temp_position[0][i];
        particles->y[particles->order[i]]     = temp_position[1][i];
        particles->z[particles->order[i]]     = temp_position[2][i];
        particles->nx[particles->order[i]]    = temp_normal[0][i];
        particles->ny[particles->order[i]]    = temp_normal[1][i];
        particles->nz[particles->order[i]]    = temp_normal[2][i];
        particles->area[particles->order[i]]         = temp_area[i];
        particles->source_term[particles->order[i]]  = temp_source[i];
        particles->source_term[particles->order[i] + s_numpars]
                                                     = temp_source[i + s_numpars];
        particles->xvct[particles->order[i]]         = temp_xvct[i];
        particles->xvct[particles->order[i] + s_numpars]
                                                     = temp_xvct[i + s_numpars];
    }

    free_matrix(temp_position);
    free_matrix(temp_normal);
    free_vector(temp_area);
    free_vector(temp_source);
    free_vector(temp_xvct);

/***********treecode_initialization*******/

    free_matrix(s_target_charge);
    free_matrix(s_source_charge);
    
/***********clean tree structure**********/

    TreeLinkedList_Free(&s_tree_root);

/***********variables in setup************/

    free_vector(particles->order);

/*****************************************/

    if (rank == 0) {
        printf("\nTABIPB tree structure has been deallocated.\n\n");
    }

    return 0;
}
/**********************************************************/



/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* matvec and psolve are the functions called by GMRes             */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**********************************************************/
int matvec(double *alpha, double *tpoten_old, double *beta, double *tpoten)
{
/* the main part of treecode */
/* in gmres *matvec(Alpha, X, Beta, Y) where y := alpha*A*x + beta*y */
  /* local variables */
    int i, j, k, ii, ierr;
    double temp_x, temp_area;
    double temp_charge[4];
    double pre1, pre2;
    double peng[2], peng_old[2];
    double *tpoten_temp;
    
    int particles_per_process;
    int rank = 0, num_procs = 1;
    
#ifdef MPI_ENABLED
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif
    
    make_vector(tpoten_temp, 2 * s_numpars);
    memcpy(tpoten_temp, tpoten, 2 * s_numpars * sizeof(double));
    memset(tpoten, 0, 2 * s_numpars * sizeof(double));
    
    s_ComputePBKernel(tpoten_old);
    
  /* Generate the moments if not allocated yet */
    s_ComputeAllMoments(s_tree_root, 1);

    pre1 = 0.50 * (1.0 + s_eps);
    pre2 = 0.50 * (1.0 + 1.0/s_eps);
    
    particles_per_process = s_numpars / num_procs;

    for (ii = 0; ii <= particles_per_process; ii++) {
        i = ii * num_procs + rank;
        
        if (i < s_numpars) {
            peng[0] = 0.0;
            peng[1] = 0.0;
            peng_old[0] = tpoten_old[i];
            peng_old[1] = tpoten_old[i+s_numpars];
            s_target_position[0] = s_particle_position_x[i];
            s_target_position[1] = s_particle_position_y[i];
            s_target_position[2] = s_particle_position_z[i];
            s_target_normal[0] = s_particle_normal_x[i];
            s_target_normal[1] = s_particle_normal_y[i];
            s_target_normal[2] = s_particle_normal_z[i];
        
            for (k = 0; k < 4; k++) {
                temp_charge[k] = s_target_charge[i][k];
            }

      /* remove the singularity */
            temp_x = s_particle_position_x[i];
            temp_area = s_particle_area[i];
            s_particle_position_x[i] += 100.123456789;
            s_particle_area[i] = 0.0;

      /* start to use Treecode */
            s_RunTreecode(s_tree_root, tpoten_old, temp_charge, peng);

            tpoten[i] = tpoten_temp[i] * *beta
                      + (pre1 * peng_old[0] - peng[0]) * *alpha;
            tpoten[s_numpars+i] = tpoten_temp[s_numpars+i] * *beta
                                + (pre2 * peng_old[1] - peng[1]) * *alpha;

            s_particle_position_x[i] = temp_x;
            s_particle_area[i] = temp_area;
        }
    }
    
#ifdef MPI_ENABLED
    ierr = MPI_Allreduce(MPI_IN_PLACE, tpoten, 2 * s_numpars,
                         MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    free_vector(tpoten_temp);

    s_RemoveMoments(s_tree_root);

    return 0;
}
/**********************************************************/


/**********************************************************/
int psolve(double *z, double *r)
{
/* r as original while z as scaled */

        int i;
        double scale1, scale2;
        scale1 = 0.5 * (1.0 + s_eps);
        scale2 = 0.5 * (1.0 + 1.0/s_eps);

        for (i = 0; i < s_numpars; i++) {
                z[i] = r[i]/scale1;
                z[i + s_numpars] = r[i + s_numpars]/scale2;
        }

        return 0;
}
/**********************************************************/


/**********************************************************/
int psolve_precond(double *z, double *r)
{
/* r as original while z as scaled */

    int i, j, idx = 0, nrow = 0, nrow2, ibeg = 0, iend = 0;
    int *ipiv, inc, nrhs, info;
    //double **matrixA;
    double *columnMajorA, *rhs;
    double L1, L2, L3, L4, area;
    double tp[3], tq[3], sp[3], sq[3];
    double r_s[3], rs, irs, sumrs;
    double G0, kappa_rs, exp_kappa_rs, Gk;
    double cos_theta, cos_theta0, tp1, tp2, dot_tqsq;
    double G10, G20, G1, G2, G3, G4;
    double pre1, pre2;

    pre1 = 0.5*(1.0+s_eps);
    pre2 = 0.5*(1.0+1.0/s_eps);
  
    //make_matrix(matrixA, 2*s_max_per_leaf, 2*s_max_per_leaf);
    make_vector(columnMajorA, 4*s_max_per_leaf*s_max_per_leaf);
    make_vector(ipiv, 2*s_max_per_leaf);
    make_vector(rhs, 2*s_max_per_leaf);

    while (idx < s_numpars) {
        leaflength(s_tree_root, idx, &nrow);
        nrow2 = nrow*2;
        ibeg  = idx;
        iend  = idx + nrow - 1;

        memset(columnMajorA, 0, nrow2*nrow2*sizeof(double));
        memset(ipiv, 0, nrow2*sizeof(int));
        memset(rhs, 0, nrow2*sizeof(double));

        for (i = ibeg; i <= iend; i++) {
            tp[0] = s_particle_position_x[i];
            tp[1] = s_particle_position_y[i];
            tp[2] = s_particle_position_z[i];
            tq[0] = s_particle_normal_x[i];
            tq[1] = s_particle_normal_y[i];
            tq[2] = s_particle_normal_z[i];

            for (j = ibeg; j < i; j++) {
                sp[0] = s_particle_position_x[j];
                sp[1] = s_particle_position_y[j];
                sp[2] = s_particle_position_z[j];
                sq[0] = s_particle_normal_x[j];
                sq[1] = s_particle_normal_y[j];
                sq[2] = s_particle_normal_z[j];

                r_s[0] = sp[0]-tp[0]; r_s[1] = sp[1]-tp[1]; r_s[2] = sp[2]-tp[2];
                sumrs = r_s[0]*r_s[0] + r_s[1]*r_s[1] + r_s[2]*r_s[2];
                rs = sqrt(sumrs);
                irs = 1.0/rs;
                G0 = ONE_OVER_4PI * irs;
                kappa_rs = s_kappa * rs;
                exp_kappa_rs = exp(-kappa_rs);
                Gk = exp_kappa_rs * G0;

                cos_theta  = (sq[0]*r_s[0] + sq[1]*r_s[1] + sq[2]*r_s[2]) * irs;
                cos_theta0 = (tq[0]*r_s[0] + tq[1]*r_s[1] + tq[2]*r_s[2]) * irs;
                tp1 = G0* irs;
                tp2 = (1.0 + kappa_rs) * exp_kappa_rs;

                G10 = cos_theta0 * tp1;
                G20 = tp2 * G10;

                G1 = cos_theta * tp1;
                G2 = tp2 * G1;

                dot_tqsq = sq[0]*tq[0] + sq[1]*tq[1] + sq[2]*tq[2];
                G3 = (dot_tqsq - 3.0*cos_theta0*cos_theta) * irs*tp1;
                G4 = tp2*G3 - s_kappa2*cos_theta0*cos_theta*Gk;

                area = s_particle_area[j];

                L1 = G1 - s_eps*G2;
                L2 = G0 - Gk;
                L3 = G4 - G3;
                L4 = G10 - G20/s_eps;

                //matrixA[i-ibeg][j-ibeg] = -L1*area;
                //matrixA[i-ibeg][j+nrow-ibeg] = -L2*area;
                //matrixA[i+nrow-ibeg][j-ibeg] = -L3*area;
                //matrixA[i+nrow-ibeg][j+nrow-ibeg] = -L4*area;
                columnMajorA[(j-ibeg)*nrow2 + i-ibeg] = -L1*area;
                columnMajorA[(j+nrow-ibeg)*nrow2 + i-ibeg] = -L2*area;
                columnMajorA[(j-ibeg)*nrow2 + i+nrow-ibeg] = -L3*area;
                columnMajorA[(j+nrow-ibeg)*nrow2 + i+nrow-ibeg] = -L4*area;
            }

            //matrixA[i-ibeg][i-ibeg] = pre1;
            //matrixA[i+nrow-ibeg][i+nrow-ibeg] = pre2;
            columnMajorA[(i-ibeg)*nrow2 + i-ibeg] = pre1;
            columnMajorA[(i+nrow-ibeg)*nrow2 + i+nrow-ibeg] = pre2;

            for (j = i+1; j <= iend; j++) {
                sp[0] = s_particle_position_x[j];
                sp[1] = s_particle_position_y[j];
                sp[2] = s_particle_position_z[j];
                sq[0] = s_particle_normal_x[j];
                sq[1] = s_particle_normal_y[j];
                sq[2] = s_particle_normal_z[j];

                r_s[0] = sp[0]-tp[0]; r_s[1] = sp[1]-tp[1]; r_s[2] = sp[2]-tp[2];
                sumrs = r_s[0]*r_s[0] + r_s[1]*r_s[1] + r_s[2]*r_s[2];
                rs = sqrt(sumrs);
                irs = 1.0/rs;
                G0 = ONE_OVER_4PI * irs;
                kappa_rs = s_kappa * rs;
                exp_kappa_rs = exp(-kappa_rs);
                Gk = exp_kappa_rs * G0;

                cos_theta  = (sq[0]*r_s[0] + sq[1]*r_s[1] + sq[2]*r_s[2]) * irs;
                cos_theta0 = (tq[0]*r_s[0] + tq[1]*r_s[1] + tq[2]*r_s[2]) * irs;
                tp1 = G0 * irs;
                tp2 = (1.0 + kappa_rs) * exp_kappa_rs;

                G10 = cos_theta0 * tp1;
                G20 = tp2 * G10;

                G1 = cos_theta * tp1;
                G2 = tp2 * G1;

                dot_tqsq = sq[0]*tq[0] + sq[1]*tq[1] + sq[2]*tq[2];
                G3 = (dot_tqsq - 3.0*cos_theta0*cos_theta) * irs*tp1;
                G4 = tp2*G3 - s_kappa2*cos_theta0*cos_theta*Gk;

                area = s_particle_area[j];

                L1 = G1 - s_eps*G2;
                L2 = G0 - Gk;
                L3 = G4 - G3;
                L4 = G10 - G20/s_eps;

                //matrixA[i-ibeg][j-ibeg] = -L1*area;
                //matrixA[i-ibeg][j+nrow-ibeg] = -L2*area;
                //matrixA[i+nrow-ibeg][j-ibeg] = -L3*area;
                //matrixA[i+nrow-ibeg][j+nrow-ibeg] = -L4*area;
                columnMajorA[(j-ibeg)*nrow2 + i-ibeg] = -L1*area;
                columnMajorA[(j+nrow-ibeg)*nrow2 + i-ibeg] = -L2*area;
                columnMajorA[(j-ibeg)*nrow2 + i+nrow-ibeg] = -L3*area;
                columnMajorA[(j+nrow-ibeg)*nrow2 + i+nrow-ibeg] = -L4*area;
            }
        }

        for (i = 0; i < nrow; i++) {
            rhs[i] = r[i+ibeg];
            rhs[i+nrow] = r[i+ibeg+s_numpars];
        }

        // Jiahui's implementation of LU decomposition
        //inc = lu_decomp(matrixA, nrow2, ipiv);
        //lu_solve(matrixA, nrow2, ipiv, rhs);

        // Apple Accelerate implementation of LAPACK LU decomposition
        //nrhs = 1;
        //dgesv_(&nrow2, &nrhs, columnMajorA, &nrow2, ipiv, rhs, &nrow2, &info);

        // LAPACKE implementation of LAPACK LU decomposition
        LAPACKE_dgesv(LAPACK_COL_MAJOR, nrow2, 1, columnMajorA, nrow2, ipiv, rhs, nrow2);

        for (i = 0; i < nrow; i++) {
            z[i+ibeg] = rhs[i];
            z[i+ibeg+s_numpars] = rhs[i+nrow];
        }

        idx += nrow;
    }

    //free_matrix(matrixA);
    free_vector(columnMajorA);
    free_vector(rhs);
    free_vector(ipiv);
  
    return 0;
}
/**********************************************************/


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* PRIVATE internal treecode functions                       * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**********************************************************/
static int s_Setup(double xyz_limits[6], struct Particles *particles)
{
/* SETUP allocates and initializes arrays needed for the Taylor expansion.
 Also, global variables are set and the Cartesian coordinates of
 the smallest box containing the particles is determined. The particle
 positions and charges are copied so that they can be restored upon exit.*/
 
    int ierr;
    int rank = 0, num_procs = 1;
    
#ifdef MPI_ENABLED
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif
    
    if (rank == 0) {
        printf("Setting up arrays for Taylor expansion...\n");
    }
    
    make_vector(tt, s_torder_lim);
    make_vector(ww, s_torder_lim);
    
    /* initializing array for Chev points */
    for (int i = 0; i < s_torder_lim; i++)
        tt[i] = cos(i * M_PI / s_order);
    
    ww[0] = 0.25 * (s_order*s_order/3.0 + 1.0/6.0);
    ww[s_order] = -ww[0];
    
    for (int i = 1; i < s_order; i++) {
        double xx = i * M_PI / s_order;
        ww[i] = -cos(xx) / (2 * sin(xx) * sin(xx));
    }

/* find bounds of Cartesion box enclosing the particles */

    xyz_limits[0] = MinVal(s_particle_position_x, s_numpars);
    xyz_limits[1] = MaxVal(s_particle_position_x, s_numpars);
    xyz_limits[2] = MinVal(s_particle_position_y, s_numpars);
    xyz_limits[3] = MaxVal(s_particle_position_y, s_numpars);
    xyz_limits[4] = MinVal(s_particle_position_z, s_numpars);
    xyz_limits[5] = MaxVal(s_particle_position_z, s_numpars);
    
    make_vector(particles->order, particles->num);

    for (int i = 0; i < particles->num; i++) {
        particles->order[i] = i;
    }

    return 0;
}
/********************************************************/


/********************************************************/
static int s_ComputePBKernel(double *phi)
{

    for (int i = 0; i < s_numpars; i++) {

        s_target_charge[i][0] = ONE_OVER_4PI;
        s_target_charge[i][1] = ONE_OVER_4PI * s_particle_normal_x[i];
        s_target_charge[i][2] = ONE_OVER_4PI * s_particle_normal_y[i];
        s_target_charge[i][3] = ONE_OVER_4PI * s_particle_normal_z[i];
        
        s_source_charge[i][0] = s_particle_area[i] * phi[s_numpars+i];
        s_source_charge[i][1] = s_particle_normal_x[i] * s_particle_area[i] * phi[i];
        s_source_charge[i][2] = s_particle_normal_y[i] * s_particle_area[i] * phi[i];
        s_source_charge[i][3] = s_particle_normal_z[i] * s_particle_area[i] * phi[i];

    }

    return 0;
}
/********************************************************/


/********************************************************/
static int s_ComputeAllMoments(struct TreeLinkedListNode *p, int ifirst)
{
/* REMOVE_NODE recursively removes each node from the tree and deallocates
 * its memory for MS array if it exits. */

    int i;
    
    if (p->exist_ms == 0 && ifirst == 0) {
        make_matrix(p->ms, 4, s_torder3);
        make_vector(p->tx, s_torder_lim);
        make_vector(p->ty, s_torder_lim);
        make_vector(p->tz, s_torder_lim);
        s_ComputeMoments(p);
        p->exist_ms = 1;
    }

    if (p->num_children > 0) {
        for (i = 0; i < p->num_children; i++) {
            s_ComputeAllMoments(p->child[i], 0);
        }
    }

  return 0;
}
/********************************************************/


/********************************************************/
static int s_ComputeMoments(struct TreeLinkedListNode *p)
{
/* COMP_MS computes the moments for node P needed in the Taylor
 * approximation */

    int i, j, k1, k2, k3, kk;
    double dx, dy, dz;
    double x0, x1, y0, y1, z0, z1;
    double sumA1, sumA2, sumA3;
    double temp11, temp12, temp21, temp22;
    double mom1, mom2, mom3, mom4, mom5, mom6, mom7, mom8;
    double xx, yy, zz;
    double *xibeg, *yibeg, *zibeg, *qq;
    
    double Dd, dj[s_torder_lim];
    double a1i[s_torder_lim], a2j[s_torder_lim], a3k[s_torder_lim];
    double w1i[s_torder_lim];
    double summ[4][s_torder3];
    int a1exactind, a2exactind, a3exactind;
    
    
    for (i = 0; i < 4; i++) {
        for (j = 0; j < s_torder3; j++) {
            p->ms[i][j] = 0.0;
        }
    }
    
    xibeg = &(s_particle_position_x[p->ibeg]);
    yibeg = &(s_particle_position_y[p->ibeg]);
    zibeg = &(s_particle_position_z[p->ibeg]);
    
    x0 = p->x_min;
    x1 = p->x_max;
    y0 = p->y_min;
    y1 = p->y_max;
    z0 = p->z_min;
    z1 = p->z_max;
    
    for (i = 0; i < 4; i++) {
        for (j = 0; j < s_torder3; j++) {
            summ[i][j] = 0.0;
        }
    }
    
    for (i = 0; i < s_torder_lim; i++) {
        p->tx[i] = x0 + (tt[i] + 1.0)/2.0 * (x1 - x0);
        p->ty[i] = y0 + (tt[i] + 1.0)/2.0 * (y1 - y0);
        p->tz[i] = z0 + (tt[i] + 1.0)/2.0 * (z1 - z0);
    }
    
    dj[0] = 0.5;
    dj[s_order] = 0.5;
    for (j = 1; j < s_order; j++)
        dj[j] = 1.0;
    
    for (j = 0; j < s_torder_lim; j++)
        w1i[j] = ((j % 2 == 0)? 1 : -1) * dj[j];

    for (i = 0; i < p->numpar; i++) {
    
        sumA1 = 0.0;
        sumA2 = 0.0;
        sumA3 = 0.0;
        
        a1exactind = -1;
        a2exactind = -1;
        a3exactind = -1;
    
        xx = xibeg[i];
        yy = yibeg[i];
        zz = zibeg[i];
        qq = s_source_charge[p->ibeg+i];
        
        for (j = 0; j < s_torder_lim; j++) {
            a1i[j] = w1i[j] / (xx - p->tx[j]);
            a2j[j] = w1i[j] / (yy - p->ty[j]);
            a3k[j] = w1i[j] / (zz - p->tz[j]);

            sumA1 += a1i[j];
            sumA2 += a2j[j];
            sumA3 += a3k[j];
            
            if (fabs(xx - p->tx[j]) < DBL_MIN) a1exactind = j;
            if (fabs(yy - p->ty[j]) < DBL_MIN) a2exactind = j;
            if (fabs(zz - p->tz[j]) < DBL_MIN) a3exactind = j;
        }
        
        if (a1exactind > -1) {
            sumA1 = 1.0;
            for (j = 0; j < s_torder_lim; j++) a1i[j] = 0.0;
            a1i[a1exactind] = 1.0;
        }
        
        if (a2exactind > -1) {
            sumA2 = 1.0;
            for (j = 0; j < s_torder_lim; j++) a2j[j] = 0.0;
            a2j[a2exactind] = 1.0;
        }
        
        if (a3exactind > -1) {
            sumA3 = 1.0;
            for (j = 0; j < s_torder_lim; j++) a3k[j] = 0.0;
            a3k[a3exactind] = 1.0;
        }
        
        Dd = 1.0 / (sumA1 * sumA2 * sumA3);
        
        kk = -1;
        for (k1 = 0; k1 < s_torder_lim; k1++) {
            for (k2 = 0; k2 < s_torder_lim; k2++) {
                for (k3 = 0; k3 < s_torder_lim; k3++) {
                    kk++;
                
                    mom1 = a1i[k1] * a2j[k2] * a3k[k3] * Dd;
                    
                    summ[0][kk] += mom1 * qq[0];
                    summ[1][kk] += mom1 * qq[1];
                    summ[2][kk] += mom1 * qq[2];
                    summ[3][kk] += mom1 * qq[3];
                }
            }
        }
    }
    
    for (j = 0; j < 4; j++)
        memcpy(p->ms[j], summ[j], s_torder3*sizeof(double));

    return 0;
}
/********************************************************/


/********************************************************/
static int s_RunTreecode(struct TreeLinkedListNode *p, double *tpoten_old, double tempq[4],
                         double peng[2])
{
  /* RunTreecode() is self recurrence function */
  
    double tx, ty, tz, dist, pengchild[2];
    int i;


  /* determine DISTSQ for MAC test */
    tx = p->x_mid - s_target_position[0];
    ty = p->y_mid - s_target_position[1];
    tz = p->z_mid - s_target_position[2];
    dist = sqrt(tx*tx + ty*ty + tz*tz);

  /* initialize potential energy */
    peng[0] = 0.0; 
    peng[1] = 0.0;

/* If MAC is accepted and there is more than 1 particale in the */
/* box use the expansion for the approximation. */

    if (p->radius < dist*theta && p->numpar > 40) {
        s_ComputeTreePB(p, tempq, peng);
    } else {
        if (p->num_children == 0) {
            s_ComputeDirectPB(p->ibeg, p->iend, tpoten_old, peng);
        } else {
      /* If MAC fails check to see if there are children. If not, perform */
      /* direct calculation.  If there are children, call routine */
      /* recursively for each. */
            for (i = 0; i < p->num_children; i++) {
                pengchild[0] = 0.0; 
                pengchild[1] = 0.0;
                s_RunTreecode(p->child[i], tpoten_old, tempq, pengchild);
                peng[0] += pengchild[0];
                peng[1] += pengchild[1];
            }
        }
    }

    return 0;
}
/********************************************************/


/********************************************************/
static int s_ComputeTreePB(struct TreeLinkedListNode *p, double tempq[4], double peng[2])
{
    
    double pt_comp_ = 0.;
    double pt_comp_dx = 0.;
    double pt_comp_dy = 0.;
    double pt_comp_dz = 0.;
    
    double* restrict cluster_x = p->tx;
    double* restrict cluster_y = p->ty;
    double* restrict cluster_z = p->tz;

    double* restrict cluster_q_   = p->ms[0];
    double* restrict cluster_q_dx = p->ms[1];
    double* restrict cluster_q_dy = p->ms[2];
    double* restrict cluster_q_dz = p->ms[3];
    
    double target_x = s_target_position[0];
    double target_y = s_target_position[1];
    double target_z = s_target_position[2];
    
    int ii = 0;
    for (int i = 0; i < s_torder_lim; i++) {
        for (int j = 0; j < s_torder_lim; j++) {
            for (int k = 0; k < s_torder_lim; k++) {

                double dx = target_x - cluster_x[i];
                double dy = target_y - cluster_y[j];
                double dz = target_z - cluster_z[k];

                double r2    = dx*dx + dy*dy + dz*dz;
                double r     = sqrt(r2);
                double rinv  = 1.0 / r;
                double r3inv = rinv  * rinv * rinv;
                double r5inv = r3inv * rinv * rinv;

                double expkr   =  exp(-s_kappa * r);
                double d1term  =  r3inv * expkr * (1. + (s_kappa * r)); 
                double d1term1 = -r3inv + d1term * s_eps;
                double d1term2 = -r3inv + d1term / s_eps;
                double d2term  =  r5inv * (-3. + expkr * (3. + (3. * s_kappa * r)
                                                       + (s_kappa * s_kappa * r2)));
                double d3term  =  r3inv * ( 1. - expkr * (1. + s_kappa * r));

                pt_comp_    += (rinv * (1. - expkr) * (cluster_q_  [ii])
                                          + d1term1 * (cluster_q_dx[ii] * dx
                                                     + cluster_q_dy[ii] * dy
                                                     + cluster_q_dz[ii] * dz));
                                        
                pt_comp_dx  += (cluster_q_  [ii]  * (d1term2 * dx)
                             - (cluster_q_dx[ii]  * (dx * dx * d2term + d3term)
                             +  cluster_q_dy[ii]  * (dx * dy * d2term)
                             +  cluster_q_dz[ii]  * (dx * dz * d2term)));
                             
                pt_comp_dy  += (cluster_q_  [ii]  *  d1term2 * dy
                             - (cluster_q_dx[ii]  * (dx * dy * d2term)
                             +  cluster_q_dy[ii]  * (dy * dy * d2term + d3term)
                             +  cluster_q_dz[ii]  * (dy * dz * d2term)));
                             
                pt_comp_dz  += (cluster_q_  [ii]  *  d1term2 * dz
                             - (cluster_q_dx[ii]  * (dx * dz * d2term)
                             +  cluster_q_dy[ii]  * (dy * dz * d2term)
                             +  cluster_q_dz[ii]  * (dz * dz * d2term + d3term)));

                ii++;
            }
        }
    }
            
    peng[0] = tempq[0] * pt_comp_;
    peng[1] = tempq[1] * pt_comp_dx  +  tempq[2] * pt_comp_dy  +  tempq[3] * pt_comp_dz;

    return 0;
}
/********************************************************/



/********************************************************/
static int s_ComputeDirectPB(int ibeg, int iend,
                             double *tpoten_old, double peng[2])
{
  /* COMPF_DIRECT directly computes the force on the current target
 * particle determined by the global variable s_target_position.*/
    int j;
    double peng_old[2], L1, L2, L3, L4, area;
    double tp[3], tq[3], sp[3], sq[3], r_s[3];
    double rs, irs, sumrs;
    double G0, kappa_rs, exp_kappa_rs, Gk;
    double cos_theta, cos_theta0, tp1, tp2, dot_tqsq;
    double G10, G20, G1, G2, G3, G4;

    peng[0] = 0.0;
    peng[1] = 0.0;

    tp[0] = s_target_position[0];
    tp[1] = s_target_position[1];
    tp[2] = s_target_position[2];
    tq[0] = s_target_normal[0];
    tq[1] = s_target_normal[1];
    tq[2] = s_target_normal[2];
    
    for (j = ibeg; j < iend+1; j++) {
        sp[0] = s_particle_position_x[j];
        sp[1] = s_particle_position_y[j];
        sp[2] = s_particle_position_z[j];
        sq[0] = s_particle_normal_x[j];
        sq[1] = s_particle_normal_y[j];
        sq[2] = s_particle_normal_z[j];

        r_s[0] = sp[0]-tp[0];  
        r_s[1] = sp[1]-tp[1];  
        r_s[2] = sp[2]-tp[2];
        sumrs = r_s[0]*r_s[0] + r_s[1]*r_s[1] + r_s[2]*r_s[2];
        rs = sqrt(sumrs);
        irs = 1.0/rs;
        G0 = ONE_OVER_4PI * irs;
        kappa_rs = s_kappa * rs;
        exp_kappa_rs = exp(-kappa_rs);
        Gk = exp_kappa_rs * G0;

        cos_theta  = (sq[0]*r_s[0] + sq[1]*r_s[1] + sq[2]*r_s[2]) * irs;
        cos_theta0 = (tq[0]*r_s[0] + tq[1]*r_s[1] + tq[2]*r_s[2]) * irs;
        tp1 = G0* irs;
        tp2 = (1.0 + kappa_rs) * exp_kappa_rs;

        G10 = cos_theta0 * tp1;
        G20 = tp2 * G10;

        G1 = cos_theta * tp1;
        G2 = tp2 * G1;

        dot_tqsq = sq[0]*tq[0] + sq[1]*tq[1] + sq[2]*tq[2];
        G3 = (dot_tqsq - 3.0*cos_theta0*cos_theta) * irs*tp1;
        G4 = tp2*G3 - s_kappa2*cos_theta0*cos_theta*Gk;

        L1 = G1 - s_eps*G2;
        L2 = G0 - Gk;
        L3 = G4 - G3;
        L4 = G10 - G20/s_eps;

        peng_old[0] = tpoten_old[j];  
        peng_old[1] = tpoten_old[j + s_numpars];
        area = s_particle_area[j];

        peng[0] += (L1*peng_old[0] + L2*peng_old[1]) * area;
        peng[1] += (L3*peng_old[0] + L4*peng_old[1]) * area;
    }

    return 0;
}
/********************************************************/


/********************************************************/
static int s_RemoveMoments(struct TreeLinkedListNode *p)
{
/* REMOVE_NODE recursively removes each node from the
 * tree and deallocates its memory for MS array if it exits. */
 
    int i;

    if (p->exist_ms == 1) {
        free_matrix(p->ms);
        free_vector(p->tx);
        free_vector(p->ty);
        free_vector(p->tz);
        p->exist_ms = 0;
    }

    if (p->num_children > 0) {
        for (i = 0; i < p->num_children; i++)
            s_RemoveMoments(p->child[i]);
    }

    return 0;
}
/********************************************************/


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* PRIVATE functions for preconditioning                     * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**********************************************************/
static void leaflength(struct TreeLinkedListNode *p, int idx, int *nrow)
{
/* find the leaf length */

    int i;

    if (idx == p->ibeg && p->num_children == 0) {
        *nrow = p->numpar;
    } else {
        if (p->num_children != 0) {
            for (i = 0; i < p->num_children; i++)
                leaflength(p->child[i], idx, nrow);
        }
     }

}
/**********************************************************/


/**********************************************************/
static int lu_decomp(double **A, int N, int *ipiv)
{
/* we're doing it this way because something is wrong with
 * linking with CMake */
 
  int i, j, k, imax;
  double maxA, *ptr, absA, Tol = 1.0e-14;

  for ( i = 0; i <= N; i++ )
    ipiv[i] = i; // record pivoting number

  for ( i = 0; i < N; i++ ) {
    maxA = 0.0;
    imax = i;
    for (k = i; k < N; k++)
      if ((absA = fabs(A[k][i])) > maxA) {
        maxA = absA;
        imax = k;
      }

    if (maxA < Tol) return 0; //failure, matrix is degenerate

    if (imax != i) {
      //pivoting P
      j = ipiv[i];
      ipiv[i] = ipiv[imax];
      ipiv[imax] = j;

      //pivoting rows of A
      ptr = A[i];
      A[i] = A[imax];
      A[imax] = ptr;

      //counting pivots starting from N (for determinant)
      ipiv[N]++;
    }

    for (j = i + 1; j < N; j++) {
      A[j][i] /= A[i][i];

      for (k = i + 1; k < N; k++)
        A[j][k] -= A[j][i] * A[i][k];
    }
  }

  return 1;
}
/**********************************************************/


/**********************************************************/
static void lu_solve(double **matrixA, int N, int *ipiv, double *rhs)
{
  /* b will contain the solution */
  
  int i, k;
  double *xtemp;

  make_vector(xtemp, N);

  for (i = 0; i < N; i++) {
    xtemp[i] = rhs[ipiv[i]];

    for (k = 0; k < i; k++)
      xtemp[i] -= matrixA[i][k] * xtemp[k];
  }

  for (i = N - 1; i >= 0; i--) {
    for (k = i + 1; k < N; k++)
      xtemp[i] -= matrixA[i][k] * xtemp[k];

    xtemp[i] = xtemp[i] / matrixA[i][i];
  }

  for (i = 0; i < N; i++) {
    rhs[i] = xtemp[i];
  }
  free_vector(xtemp);
}
/**********************************************************/
