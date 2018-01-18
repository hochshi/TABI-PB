/**************************************************************************
* FILE NAME: main.c                                                       *
*                                                                         *
* PURPOSE: calls primary tabipb routine and prints output when running    *
*          as standalone                                                  *
*                                                                         *
* AUTHORS: Leighton Wilson, University of Michigan, Ann Arbor, MI         *
*          Jiahui Chen, Southern Methodist University, Dallas, TX         *
*                                                                         *
* BASED ON PACKAGE ORIGINALLY WRITTEN IN FORTRAN BY:                      *
*          Weihua Geng, Southern Methodist University, Dallas, TX         *
*          Robery Krasny, University of Michigan, Ann Arbor, MI           *
*                                                                         *
* DEVELOPMENT HISTORY:                                                    *
*                                                                         *
* Date        Author            Description Of Change                     *
* ----        ------            ---------------------                     *
* 01/14/2018  Leighton Wilson   Fixing read in of PQRs                    *
* 07/14/2016  Jiahui Chen       Added Sphinx support                      *
* 06/30/2016  Jiahui Chen       Rebuilt wrapper architecture              *
* 06/23/2016  Leighton Wilson   Added NanoShaper support                  *
*                                                                         *
**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "tabipb.h"
#include "print_output.h"

#include "array.h"
#include "TABIPBstruct.h"

int main(int argc, char **argv)
{
  /* main reads the input file, writes xyzr file for msms and sets up position,
     radius and charges */

    FILE *fp, *wfp;
    char c[16];
    char fname_tp[256];
    char c1[10], c2[10], c3[10], c4[10], c5[10];
    double a1, a2, a3, b1, b2;
    double density, radius, epsp, epsw, bulk_strength, theta, temp;
    int maxparnode, order, mesh_flag, output_datafile, ierr, i;

  /* timing functions for *nix systems */
#ifndef _WIN32                                                                     
    extern void timer_start();
    extern void timer_end();
#endif

#ifndef _WIN32                                                                     
    timer_start("TOTAL_TIME");
#endif  

    TABIPBparm *main_parm = malloc(sizeof *main_parm);
    TABIPBvars *main_vars = malloc(sizeof *main_vars);

/********************************************************/

    fp = fopen("usrdata.in", "r");
    ierr = fscanf(fp, "%s %s", c, main_parm->fname);

    ierr = fscanf(fp, "%s %lf", c, &density);
    main_parm->density = density;

    ierr = fscanf(fp, "%s %lf", c, &radius);
    main_parm->probe_radius = radius;

    ierr = fscanf(fp, "%s %lf", c, &epsp);
    main_parm->epsp = epsp;

    ierr = fscanf(fp, "%s %lf", c, &epsw);
    main_parm->epsw = epsw;

    ierr = fscanf(fp, "%s %lf", c, &bulk_strength);
    main_parm->bulk_strength = bulk_strength;

    ierr = fscanf(fp, "%s %lf", c, &temp);
    main_parm->temp = temp;

    ierr = fscanf(fp, "%s %d", c, &order);
    main_parm->order = order;

    ierr = fscanf(fp, "%s %d", c, &maxparnode);
    main_parm->maxparnode = maxparnode;

    ierr = fscanf(fp, "%s %lf", c, &theta);
    main_parm->theta = theta;

    ierr = fscanf(fp, "%s %d", c, &mesh_flag);
    main_parm->mesh_flag = mesh_flag;

    ierr = fscanf(fp, "%s %d", c, &output_datafile);
    main_parm->output_datafile = output_datafile;
    fclose(fp);

/********************************************************/
    sprintf(main_parm->fpath, "");

    sprintf(fname_tp, "%s%s.pqr", main_parm->fpath, main_parm->fname);
    fp = fopen(fname_tp, "r");

    sprintf(fname_tp, "molecule.xyzr");
    wfp = fopen(fname_tp, "w");

    main_parm->number_of_lines = 0;
    while (fscanf(fp, "%s %s %s %s %s %lf %lf %lf %lf %lf",
           c1, c2, c3, c4, c5, &a1, &a2, &a3, &b1, &b2) != EOF) {
        if (strncmp(c1, "ATOM", 4) == 0) {
            fprintf(wfp, "%f %f %f %f\n", a1, a2, a3, b2);
            main_parm->number_of_lines++;
        }
    }

    fclose(wfp);
    printf("Finished assembling atomic information (.xyzr) file...\n");

    make_vector(main_vars->chrpos, 3 * main_parm->number_of_lines);
    make_vector(main_vars->atmchr, main_parm->number_of_lines);
    make_vector(main_vars->atmrad, main_parm->number_of_lines);

    rewind(fp);
    i = 0;
    
    while (fscanf(fp, "%s %s %s %s %s %lf %lf %lf %lf %lf",
           c1, c2, c3, c4, c5, &a1, &a2, &a3, &b1, &b2) != EOF) {
        if (strncmp(c1, "ATOM", 4) == 0) {
            main_vars->chrpos[3*i] = a1;
            main_vars->chrpos[3*i + 1] = a2;
            main_vars->chrpos[3*i + 2] = a3;
            main_vars->atmchr[i] = b1;
            main_vars->atmrad[i] = b2;
            i++;
        }
    }

    fclose(fp);
    printf("Finished assembling charge structures from .pqr file...\n");

    ierr = TABIPB(main_parm, main_vars);

    ierr = OutputPrint(main_vars);
    if (output_datafile == 1) {
        ierr = OutputVTK(main_parm, main_vars);
    }

    free(main_parm);
    free_vector(main_vars->atmchr);
    free_vector(main_vars->chrpos);
    free_vector(main_vars->atmrad);
    free_vector(main_vars->vert_ptl); // allocate in output_potential()
    free_vector(main_vars->xvct);
    free_matrix(main_vars->vert);
    free_matrix(main_vars->snrm);
    free_matrix(main_vars->face);
    free(main_vars);

#ifndef _WIN32
    timer_end();
#endif

    return 0;
}
