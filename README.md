# TABI-PB
TABI-PB (treecode-accelerated boundary integral) solves the linearized Poisson-Boltzmann equation. The solver employs a well-conditioned boundary integral formulation for the electrostatic potential and its normal derivative on a triangulated molecular surface, and the integral equations are discretized by nodepatch method. The linear system is solved by GMRES iteration, and the matrix-vector product is carried out by a barycentric Lagrange interpolation dual tree traversal (BLDTT) fast summation method which reduces the cost from O(N^2) to O(N), where N is the number elements. This solver also includes NVIDIA GPU support using OpenACC with the PGI/ NVIDIA HPC compilers.

This TABI-PB repo serves as a submodule to APBS, and also works as a standalone distribution. For more information on building APBS, [visit this link](https://apbs.readthedocs.io/en/latest/).

   Authors:  
   - Leighton W. Wilson  (lwwilson@umich.edu) 
   - Robert Krasny  (krasny@umich.edu) 
   - Weihua Geng  (wgeng@smu.edu)
   - Jiahui Chen  (chenj159@msu.edu)


## References
Please refer to the following references for more background:

  - E. Jurrus, D. Engel, K. Star, K. Monson, J. Brandi, L. E. Felberg, D. H. Brookes, L. Wilson, J. Chen, K. Liles, M. Chen, P. Li, D. W. Gohara, T. Dolinsky, R. Konecny, D. R. Koes, J. E. Nielsen, T. Head- Gordon, W. Geng, R. Krasny, G. W. Wei, M. J. Holst, J. A. McCammon, and N. A. Baker, Improvements to the APBS biomolecular solvation software suite, _Protein Sci._ __27__ (2017), 112-128.
   
  - W.H. Geng and R. Krasny, A treecode-accelerated boundary integral Poisson-Boltzmann solver for continuum electrostatics of solvated biomolecules, _J. Comput. Phys._ __247__ (2013), 62-87.

  - L. Wilson, W. Geng and R. Krasny, TABI-PB 2.0: An improved version of the Treecode-Accelerated Boundary Integral Poisson-Boltzmann solver, _J. Phys. Chem. B_ __126__ (2022), 7104-7113.

## Build Instructions

This project uses CMake to manage and configure its build system. In principle, 
building an independent `tabipb` executable is as simple as executing the following 
from the top level directory of TABI-PB:

    mkdir build; cd build; export CC=<CXX compiler>; cmake ..; make

This creates a `tabipb` executable located at `TABI-PB/build/bin/tabipb`.
Compile the OpenACC version with an NVIDIA HPC compiler and
`-DENABLE_OPENACC=ON`. Compile the native CUDA version with
`-DENABLE_CUDA_CC=ON` and an explicit `CMAKE_CUDA_ARCHITECTURES` value.

At startup, the executable reports the selected build backend as `CPU`, `OpenACC`,
or `CUDA`. Native-CUDA builds always use CUDA execution paths and fail instead of
silently falling back to CPU code. Development alternatives are selected at compile
time with the `ENABLE_CUDA_*` CMake options; there are no CUDA runtime environment
switches.

`tabipb` relies on NanoShaper to triangulate the molecular surface. To get a NanoShaper
executable appropriate for your system, invoke `cmake` with the flag `-DGET_NanoShaper=ON`.

## Examples

To run an example, navigate to the examples directory, and run `tabipb` with the 
example input file:
```
cd examples/
../build/bin/tabipb usrdata.in
```

## License
Copyright © 2013-2022, The Regents of the University of Michigan. Released under the [3-Clause BSD License](LICENSE.md).


## Disclaimer
This material is based upon work supported under NSF Grants DMS-0915057, DMS-1418966/1418957, DMS-1819094/1819193, DMS-2110767/2110869 and by the Extreme Science and Engineering Discovery Environment (XSEDE) under grants ACI-1548562 and ASC-190062. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF.
