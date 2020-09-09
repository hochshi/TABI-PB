#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "params.h"
#include "molecule.h"
#include "particles.h"
#include "tree.h"
#include "interaction_list.h"
#include "clusters.h"
#include "treecode.h"


int main(int argc, char* argv[])
{
    // set the parameter struct, which is read in from file provided as argv
    if (argc < 2) { std::cout << "No input file set." << std::endl; std::exit(1); }
    struct Params params(argv[1]);
    
    //construct the biomolecule from the provided pqr file
    class Molecule molecule(params);
    
    // output the molecule to an xyzr file for NanoShaper
    molecule.build_xyzr_file();
    
    // build particles from a NanoShaper surface generated by xyzr file
    class Particles particles(molecule, params);
    
    // build a tree on the particles constructed above
    class Tree tree(particles, params);
    
    // build clusters and set interpolation points for the tree constructed above
    class Clusters clusters(particles, tree, params);
    
    // build interaction lists from the tree constructed above
    class InteractionList interaction_list(tree, params);
    
    // initialize the treecode and construct the potential output array
    class Treecode treecode(particles, clusters, 
                            tree, interaction_list, molecule, params);
    
    // run GMRES on the treecode object
    treecode.run_GMRES();
    
    // output resulting potential
    treecode.output();

    return 0;
}