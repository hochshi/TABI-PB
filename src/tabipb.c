/*
 * C routine to interface tabipb with apbs
 *
 * C version authored by:
 * Leighton Wilson, University of Michigan, Ann Arbor, MI
 * Jiahui Chen, Southern Methodist University, Dallas, TX
 *
 * Based on package originally written in FORTRAN by:
 * Weihua Geng, Southern Methodist University, Dallas, TX
 * Robery Krasny, University of Michigan, Ann Arbor, MI
 *
 * Works for Sphinx by Jiahui at 7/14/2016
 * Rebuild the architecture of wrapper by Jiahui at 6/30/2016
 * Build matrix free and nanoshaper by Leighton at 6/23/2016
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "tabipb.h"
#include "readin.h"
#include "treecode_tabipb_interface.h"
#include "run_gmres.h"
#include "utilities.h"

#include "global_params.h"
#include "array.h"
#include "TABIPBstruct.h"
#include "particle_struct.h"


/* internal functions */
static int s_ComputeSourceTerm(TABIPBparm *parm, TABIPBvars *vars,
                               TreeParticles *particles);
static int s_ComputePotential(TABIPBparm *parm, TABIPBvars *vars,
                              TreeParticles *particles, double *chrptl);
static int s_OutputPotential(TABIPBvars *vars, TreeParticles *particles);


int TABIPB(TABIPBparm *parm, TABIPBvars *vars) {
  /* Assemble the TABIPBparm out side this subroutine, and pass the three arrays */
  /* TABIPBparm a structure of parameters: file path, file name, density,
     probe radius, epsp, epsw, bulk_strength, treecode order, treecode maxparnode,
     treecode theta, mesh flag, and number of lines in pqr file */
  /* t_chrpos, t_atmchr and t_atmrad are three arrays of position, charges and
     raduis */

  /* variables local to main */
    int i, j;
    double energy_solvation = 0.0;
    double energy_coulomb = 0.0;

    /* variables used to compute potential solution */
    double *chrptl;

    printf("\n              Treecode order: %d", parm->order);
    printf("\n Max # of particles per leaf: %d", parm->maxparnode);
    printf("\n                 MAC (theta): %f\n", parm->theta);
    printf("\n                   Mesh flag: %d\n", parm->mesh_flag);

    printf("\nSetting up the TABI input...\n");

  /**creating particle struct**/
  
    TreeParticles *particles = malloc(sizeof *particles);
    

  /***************constant*****************/
    
    parm->eps = parm->epsw/parm->epsp;
    parm->kappa2 = BULK_COEFF * parm->bulk_strength / parm->epsw / parm->temp;
    parm->kappa = sqrt(parm->kappa2);


    Readin(parm, vars, particles);

    s_ComputeSourceTerm(parm, vars, particles);

    /* set up treecode */
    TreecodeInitialization(parm, vars->nface, particles);
    
    make_vector(particles->xvct, 2 * vars->nface);
    make_vector(vars->xvct, 2 * vars->nface);

    /* call GMRES */
    RunGMRES(vars->nface, particles->source_term, particles->xvct);

    /* the solvation energy computation */
    make_vector(chrptl, vars->nface);
    s_ComputePotential(parm, vars, particles, chrptl);

    energy_solvation = 0.0;
    energy_coulomb = 0.0;

    double r[3], diff[3], dist;

    for (i = 0; i < vars->nface; i++) {
        energy_solvation += chrptl[i];
    }

    for (i = 0; i < vars->natm; i++) {
        r[0] = vars->chrpos[3*i];
        r[1] = vars->chrpos[3*i + 1];
        r[2] = vars->chrpos[3*i + 2];
        
        for (j = i+1; j < vars->natm; j++){
            diff[0] = r[0] - vars->chrpos[3*j];
            diff[1] = r[1] - vars->chrpos[3*j + 1];
            diff[2] = r[2] - vars->chrpos[3*j + 2];
            dist = sqrt(diff[0]*diff[0]
                 + diff[1]*diff[1]
                 + diff[2]*diff[2]);
            energy_coulomb += 1 / parm->epsp / dist * vars->atmchr[i] * vars->atmchr[j];
        }
    }
    free_vector(chrptl);

    energy_solvation = energy_solvation * UNITS_PARA;
    energy_coulomb = energy_coulomb * UNITS_COEFF;
    vars->soleng = energy_solvation;
    vars->couleng = energy_coulomb;

    /* dellocate treecode variables, reorder particles */
    TreecodeFinalization(particles);

    s_OutputPotential(vars, particles);

    free_matrix(particles->position);
    free_matrix(particles->normal);
    free_vector(particles->area);
    free_vector(particles->source_term);
    
    return 0;

}
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */




/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* PRIVATE internal TABIPB functions                         * * * */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**********************************************************/
static int s_ComputeSourceTerm(TABIPBparm *parm, TABIPBvars *vars,
                               TreeParticles *particles)
{
/* this computes the source term where
 * S1=sum(qk*G0)/e1 S2=sim(qk*G0')/e1 */

/* local variables */
    int i, j;
    double sumrs, cos_theta, irs, G0, G1;
    double r_s[3];

    for (i = 0; i < vars->nface; i++) {
        particles->source_term[i] = 0.0;
        particles->source_term[i + vars->nface] = 0.0;
        for (j = 0; j < vars->natm; j++) {

  /* r_s = distance of charge position to triangular */
            r_s[0] = vars->chrpos[3*j] - particles->position[0][i];
            r_s[1] = vars->chrpos[3*j + 1] - particles->position[1][i];
            r_s[2] = vars->chrpos[3*j + 2] - particles->position[2][i];
            sumrs = r_s[0]*r_s[0] + r_s[1]*r_s[1] + r_s[2]*r_s[2];

  /* cos_theta = <tr_q,r_s>/||r_s||_2 */
            cos_theta = particles->normal[0][i] * r_s[0]
                      + particles->normal[1][i] * r_s[1]
                      + particles->normal[2][i] * r_s[2];
            irs = 1 / sqrt(sumrs);
            cos_theta = cos_theta * irs;

  /* G0 = 1/(4pi*||r_s||_2) */
            G0 = ONE_OVER_4PI * irs;

  /* G1 = cos_theta*G0/||r_s||_2 */
            G1 = cos_theta * G0 * irs;

  /* update source term */
            particles->source_term[i] += vars->atmchr[j] * G0 / parm->epsp;
            particles->source_term[vars->nface + i] += vars->atmchr[j]
                                                     * G1 / parm->epsp;
        }
    }

    return 0;
}
/********************************************************/


/********************************************************/
static int s_ComputePotential(TABIPBparm *parm, TABIPBvars *vars,
                              TreeParticles *particles, double *chrptl)
{
  /* local variables */
    int i, j;
    double sumrs, irs, rs, G0, Gk, kappa_rs, exp_kappa_rs;
    double cos_theta, G1, G2, L1, L2;
    double r[3], v[3], s[3], r_s[3];

    for (j = 0; j < vars->nface; j++) {
        chrptl[j] = 0.0;

  /* r[] = tr_xyz[] & v[] = tr_q[] */
        r[0] = particles->position[0][j];
        r[1] = particles->position[1][j];
        r[2] = particles->position[2][j];

        v[0] = particles->normal[0][j];
        v[1] = particles->normal[1][j];
        v[2] = particles->normal[2][j];

        for (i = 0; i < vars->natm; i++) {
  /* s = chrpos[] & r_s = r[]-s[] */
            s[0] = vars->chrpos[3*i];
            s[1] = vars->chrpos[3*i + 1];
            s[2] = vars->chrpos[3*i + 2];

            r_s[0] = r[0] - s[0];
            r_s[1] = r[1] - s[1];
            r_s[2] = r[2] - s[2];

            sumrs = r_s[0]*r_s[0] + r_s[1]*r_s[1] + r_s[2]*r_s[2];
            rs = sqrt(sumrs);
            irs = 1 / rs;

            G0 = ONE_OVER_4PI * irs;
            kappa_rs = parm->kappa * rs;
            exp_kappa_rs = exp(-kappa_rs);
            Gk = exp_kappa_rs * G0;

            cos_theta = (v[0]*r_s[0] + v[1]*r_s[1] + v[2]*r_s[2]) * irs;

            G1 = G0 * cos_theta * irs;
            G2 = G1 * (1.0 + kappa_rs) * exp_kappa_rs;

            L1 = G1 - parm->eps * G2;
            L2 = G0 - Gk;

            chrptl[j] += vars->atmchr[i]
                       * (L1*particles->xvct[j]
                       + L2*particles->xvct[vars->nface+j])
                       * particles->area[j];
        }
    }

    return 0;
}
/********************************************************/


/********************************************************/
static int s_OutputPotential(TABIPBvars *vars, TreeParticles *particles)
{
    int i, j, k, nface_vert;
    double tot_length, loc_length, aa[3], dot_aa, para_temp;
    int **ind_vert;

    nface_vert = 15; /* one vertex could have been involved
                        in at most 11 triangles, so 15 is safe */
    para_temp = UNITS_COEFF * 4 * PI;
    
    
    make_matrix(ind_vert, nface_vert, vars->nspt);
    make_vector(vars->vert_ptl, 2 * vars->nspt);
    
    for (i = 0; i < nface_vert; i++) {
        for (j = 0; j < vars->nspt; j++) {
            ind_vert[i][j] = 0;
        }
    }
    
    for (i = 0; i < 2 * vars->nspt; i++) {
        vars->vert_ptl[i] = 0.0;
    }

    for (i = 0; i < vars->nface; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < nface_vert - 1; k++) {
                if (ind_vert[k][vars->face[j][i] - 1] == 0) {
                    ind_vert[k][vars->face[j][i] - 1] = i + 1;
                    ind_vert[nface_vert - 1][vars->face[j][i] - 1] += 1;
                    break;
                }
            }
        }
    }

    memcpy(vars->xvct, particles->xvct, 2 * vars->nface * sizeof(double));

    
    for (i = 0; i < vars->nspt; i++) {
        tot_length = 0.0;
        for (j = 0; j < ind_vert[nface_vert - 1][i]; j++) {
      /* distance between vertices and centroid */
            aa[0] = particles->position[0][ind_vert[j][i]-1]
                  - vars->vert[0][i];
            aa[1] = particles->position[1][ind_vert[j][i]-1]
                  - vars->vert[1][i];
            aa[2] = particles->position[2][ind_vert[j][i]-1]
                  - vars->vert[2][i];
            dot_aa = aa[0]*aa[0] + aa[1]*aa[1] + aa[2]*aa[2];
            loc_length = sqrt(dot_aa);

            vars->vert_ptl[i] += 1.0/loc_length*vars->xvct[ind_vert[j][i]-1];
            vars->vert_ptl[i + vars->nspt] += 1.0 / loc_length
                                  * vars->xvct[ind_vert[j][i]+vars->nface-1];
            tot_length += 1.0/loc_length;
        }
        vars->vert_ptl[i] /= tot_length;
        vars->vert_ptl[i + vars->nspt] /= tot_length;
    }

    for (i = 0; i < 2 * vars->nface; i++)
        vars->xvct[i] *= para_temp;

    for (i = 0; i < vars->nspt; i++) {
        vars->vert_ptl[i] *= para_temp;
        vars->vert_ptl[i + vars->nspt] *= para_temp;
    }

    vars->max_xvct = MaxVal(vars->xvct, vars->nface);
    vars->min_xvct = MinVal(vars->xvct, vars->nface);
    vars->max_der_xvct = MaxVal(vars->xvct + vars->nface, vars->nface);
    vars->min_der_xvct = MinVal(vars->xvct + vars->nface, vars->nface);

    vars->max_vert_ptl = MaxVal(vars->vert_ptl, vars->nspt);
    vars->min_vert_ptl = MinVal(vars->vert_ptl, vars->nspt);
    vars->max_der_vert_ptl = MaxVal(vars->vert_ptl + vars->nspt, vars->nspt);
    vars->min_der_vert_ptl = MinVal(vars->vert_ptl + vars->nspt, vars->nspt);

    free_matrix(ind_vert);

    return 0;
}
/********************************************************/
