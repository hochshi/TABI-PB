#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

#include "boundary_element.h"
#include "gmres_backend_cuda.h"

#ifdef USE_CUDA_CC
#include <cuda_runtime.h>
#include "cuda_helpers.h"
#include "gmres_cuda.h"
#endif

/*  -- Iterative template routine --
*     Univ. of Tennessee and Oak Ridge National Laboratory
*     October 1, 1993
*     Details of this algorithm are described in "Templates for the
*     Solution of Linear Systems: Building Blocks for Iterative
*     Methods", Barrett, Berry, Chan, Demmel, Donato, Dongarra,
*     Eijkhout, Pozo, Romine, and van der Vorst, SIAM Publications,
*     1993. (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps).
*
*  Purpose
*  =======
*
*  GMRES solves the linear system Ax = b using the
*  Generalized Minimal Residual iterative method with preconditioning.
*
*  Convergence test: ( norm( b - A*x ) / norm( b ) ) < TOL.
*  For other measures, see the above reference.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER.
*          On entry, the dimension of the matrix.
*          Unchanged on exit.
*
*  B       (input) DOUBLE PRECISION array, dimension N.
*          On entry, right hand side vector B.
*          Unchanged on exit.
*
*  X       (input/output) DOUBLE PRECISION array, dimension N.
*          On input, the initial guess; on exit, the iterated solution.
*
*  RESTRT  (input) INTEGER
*          Restart parameter, <= N. This parameter controls the amount
*          of memory required for matrix H (see WORK and H).
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDW,RESTRT+4).
*
*  LDW     (input) INTEGER
*          The leading dimension of the array WORK. LDW >= max(1,N).
*
*  H       (workspace) DOUBLE PRECISION array, dimension (LDH,RESTRT+2).
*          This workspace is used for constructing and storing the
*          upper Hessenberg matrix. The two extra columns are used to
*          store the Givens rotation matrices.
*
*  LDH    (input) INTEGER
*          The leading dimension of the array H. LDH >= max(1,RESTRT+1).
*
*  ITER    (input/output) INTEGER
*          On input, the maximum iterations to be performed.
*          On output, actual number of iterations performed.
*
*  RESID   (input/output) DOUBLE PRECISION
*          On input, the allowable convergence measure for
*          norm( b - A*x ) / norm( b ).
*          On output, the final value of this measure.
*
*  MATVEC  (external subroutine)
*          The user must provide a subroutine to perform the
*          matrix-vector product
*
*               y := alpha*A*x + beta*y,
*
*          where alpha and beta are scalars, x and y are vectors,
*          and A is a matrix. Vector x must remain unchanged.
*          The solution is over-written on vector y.
*
*  PSOLVE  (external subroutine)
*          The user must provide a subroutine to perform the
*          preconditioner solve routine for the linear system
*
*               M*x = b,
*
*          where x and b are vectors, and M a matrix. Vector b must
*          remain unchanged. The solution is over-written on vector x.
*
*  INFO    (output) INTEGER
*
*          =  0: Successful exit. Iterated approximate solution returned.
*
*          >  0: Convergence to tolerance not achieved.
*
*  BLAS CALLS:   DAXPY, DCOPY, DDOT, DNRM2, DROT, DROTG, DSCAL
*  ============================================================
*/

static double dnrm2_(long int n, const double* w);
static void dscal_(long int n, double alpha, double* x);
static double ddot_(long int n, const double* __restrict x, const double* __restrict y);
static void daxpy_(long int n, double alpha, const double* __restrict x, double* __restrict y);
static void drot_(double& dx, double& dy, double c, double s);
static void drotg_(double da, double db, double& c, double& s);
static void dtrsv_(long int n, const double* a, long int lda, double* x);
static void dgemv_(long int m, long int n, const double* a, long int lda,
                   const double* x, double* y);

static void update_(long int i, long int n, double* x, const double* h, long int ldh,
                    double* y, const double* s, const double* v, long int ldv);
static void basis_(long int i, long int n, double* h, double* v, long int ldv, double* w);
int BoundaryElement::gmres_impl_(const GmresView& view, bool enable_cuda_backend)
{
    long int n = view.n;
    const double* b = view.b;
    double* x = view.x;
    long int restrt = view.restrt;
    double* work = view.work;
    long int ldw = view.ldw;
    double* h = view.h;
    long int ldh = view.ldh;
    long int& iter = *view.iter;
    double& resid = *view.residual;

    long int maxit = iter;
    double tol = resid;

/*     Store the Givens parameters in matrix H. */
/*     Set initial residual (AV is temporary workspace here). */

#ifdef USE_CUDA_CC
    double* work_dev = nullptr;
    double* x_dev = nullptr;
    double* b_dev = nullptr;
    double* h_dev = nullptr;
    double* resid_dev = nullptr;
    double* scalars_dev = nullptr;
    void* cuda_stream = nullptr;
    bool use_cuda_gmres = false;
    bool use_cuda_precond = false;
    const char* gmres_env = std::getenv("TABIPB_CUDA_GMRES");
    const char* require_env = std::getenv("TABIPB_CUDA_REQUIRE_GMRES");
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_all = (require_all_env && std::strcmp(require_all_env, "0") != 0);
    const bool require_gmres = require_all ||
                               (require_env && std::strcmp(require_env, "0") != 0);
    use_cuda_gmres = enable_cuda_backend &&
                     (require_gmres || (gmres_env && std::strcmp(gmres_env, "0") != 0));
    (void)std::getenv("TABIPB_CUDA_GMRES_DEBUG_MAP");
    if (use_cuda_gmres) {
        cuda_stream = nullptr;
        use_cuda_precond = !params_.precondition_;
        if (require_gmres && params_.precondition_) {
            std::cerr << "[CUDA_GMRES] CUDA GMRES requires diagonal precondition; block precondition is CPU-only. "
                      << "Disable block precondition or implement GPU block precondition.\n";
            std::exit(1);
        }

        const std::size_t work_len = static_cast<std::size_t>(ldw) * static_cast<std::size_t>(restrt + 4);
        cudaError_t err = cudaMalloc(&work_dev, work_len * sizeof(double));
        if (err != cudaSuccess) {
            if (require_gmres) {
                std::cerr << "[CUDA_GMRES] cudaMalloc(work) failed: "
                          << cudaGetErrorString(err) << "\n";
                std::exit(1);
            }
            use_cuda_gmres = false;
        } else {
            err = cudaMalloc(&x_dev, static_cast<std::size_t>(n) * sizeof(double));
            if (err != cudaSuccess) {
                if (require_gmres) {
                    std::cerr << "[CUDA_GMRES] cudaMalloc(x) failed: "
                              << cudaGetErrorString(err) << "\n";
                    std::exit(1);
                }
                cudaFree(work_dev);
                work_dev = nullptr;
                use_cuda_gmres = false;
            } else {
                CUDA_MEMCPY_ASYNC(x_dev, x, static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
                err = cudaMalloc(&b_dev, static_cast<std::size_t>(n) * sizeof(double));
                if (err != cudaSuccess) {
                    if (require_gmres) {
                        std::cerr << "[CUDA_GMRES] cudaMalloc(b) failed: "
                                  << cudaGetErrorString(err) << "\n";
                        std::exit(1);
                    }
                    cudaFree(x_dev);
                    cudaFree(work_dev);
                    x_dev = nullptr;
                    work_dev = nullptr;
                    use_cuda_gmres = false;
                } else {
                    CUDA_MEMCPY_ASYNC(b_dev, b, static_cast<std::size_t>(n) * sizeof(double),
                                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
                }
                if (use_cuda_gmres) {
                    const std::size_t h_cols = static_cast<std::size_t>(restrt + 2);
                    const std::size_t h_bytes = static_cast<std::size_t>(ldh) * h_cols * sizeof(double);
                    err = cudaMalloc(&h_dev, h_bytes);
                    if (err != cudaSuccess) {
                        if (require_gmres) {
                            std::cerr << "[CUDA_GMRES] cudaMalloc(h) failed: "
                                      << cudaGetErrorString(err) << "\n";
                            std::exit(1);
                        }
                        if (b_dev) cudaFree(b_dev);
                        cudaFree(x_dev);
                        cudaFree(work_dev);
                        b_dev = nullptr;
                        x_dev = nullptr;
                        work_dev = nullptr;
                        use_cuda_gmres = false;
                    } else {
                        err = cudaMemsetAsync(h_dev, 0, h_bytes,
                                              reinterpret_cast<cudaStream_t>(cuda_stream));
                        if (err != cudaSuccess) {
                            if (require_gmres) {
                                std::cerr << "[CUDA_GMRES] cudaMemset(h) failed: "
                                          << cudaGetErrorString(err) << "\n";
                                std::exit(1);
                            }
                            cudaFree(h_dev);
                            if (b_dev) cudaFree(b_dev);
                            cudaFree(x_dev);
                            cudaFree(work_dev);
                            h_dev = nullptr;
                            b_dev = nullptr;
                            x_dev = nullptr;
                            work_dev = nullptr;
                            use_cuda_gmres = false;
                        }
                    }
                    if (use_cuda_gmres) {
                        err = cudaMalloc(&resid_dev, sizeof(double));
                        if (err != cudaSuccess) {
                            if (require_gmres) {
                                std::cerr << "[CUDA_GMRES] cudaMalloc(resid) failed: "
                                          << cudaGetErrorString(err) << "\n";
                                std::exit(1);
                            }
                            if (h_dev) cudaFree(h_dev);
                            if (b_dev) cudaFree(b_dev);
                            cudaFree(x_dev);
                            cudaFree(work_dev);
                            resid_dev = nullptr;
                            h_dev = nullptr;
                            b_dev = nullptr;
                            x_dev = nullptr;
                            work_dev = nullptr;
                            use_cuda_gmres = false;
                        } else {
                            err = cudaMalloc(&scalars_dev, 3 * sizeof(double));
                            if (err != cudaSuccess) {
                                if (require_gmres) {
                                    std::cerr << "[CUDA_GMRES] cudaMalloc(scalars) failed: "
                                              << cudaGetErrorString(err) << "\n";
                                    std::exit(1);
                                }
                                cudaFree(resid_dev);
                                if (h_dev) cudaFree(h_dev);
                                if (b_dev) cudaFree(b_dev);
                                cudaFree(x_dev);
                                cudaFree(work_dev);
                                scalars_dev = nullptr;
                                resid_dev = nullptr;
                                h_dev = nullptr;
                                b_dev = nullptr;
                                x_dev = nullptr;
                                work_dev = nullptr;
                                use_cuda_gmres = false;
                            }
                        }
                    }
                }
            }
        }
    }
#else
    const bool use_cuda_gmres = false;
    (void)use_cuda_gmres;
#endif

#ifdef USE_CUDA_CC
    if (use_cuda_gmres) {
        if (use_cuda_precond && b_dev) {
            CUDA_MEMCPY_ASYNC(work_dev + 2 * ldw, b_dev,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
        } else if (b_dev) {
            CUDA_MEMCPY_ASYNC(work_dev + 2 * ldw, b_dev,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
        } else {
            CUDA_MEMCPY_ASYNC(work_dev + 2 * ldw, b,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
        }
        if (dnrm2_(n, x) != 0.) {
            BoundaryElement::matrix_vector_cuda(-1., x_dev, 1., work_dev + 2 * ldw, cuda_stream);
            if (!use_cuda_precond) {
                CUDA_MEMCPY_ASYNC(work + 2 * ldw, work_dev + 2 * ldw,
                                static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
                CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
            }
        }
    } else
#endif
    {
        for (long int idx = 0; idx < n; ++idx) work[2 * ldw + idx] = b[idx];
        if (dnrm2_(n, x) != 0.) {
            BoundaryElement::matrix_vector(-1., x, 1., &work[2 * ldw]);
        }
    }

    #ifdef USE_CUDA_CC
    if (use_cuda_precond) {
        BoundaryElement::precondition_diagonal_cuda(work_dev, work_dev + 2 * ldw, cuda_stream);
    } else
    #endif
    {
        if (params_.precondition_) BoundaryElement::precondition_block   (work, &work[2 * ldw]);
        else                       BoundaryElement::precondition_diagonal(work, &work[2 * ldw]);
    }

#ifdef USE_CUDA_CC
    if (use_cuda_gmres && !use_cuda_precond) {
        CUDA_MEMCPY_ASYNC(work_dev + 2 * ldw, work + 2 * ldw, static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
        CUDA_MEMCPY_ASYNC(work_dev, work, static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
    }
#endif

    double bnrm2 = 0.0;
#ifdef USE_CUDA_CC
    if (use_cuda_gmres && b_dev && scalars_dev) {
        gmres_cuda_dnrm2(b_dev, static_cast<std::size_t>(n),
                         scalars_dev, 1, cuda_stream);
        CUDA_MEMCPY_ASYNC(&bnrm2, scalars_dev, sizeof(double),
                        cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
        CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
    } else
#endif
    {
        bnrm2 = dnrm2_(n, b);
    }
    if (bnrm2 == 0.) bnrm2 = 1.;
    
#ifdef USE_CUDA_CC
    if (use_cuda_gmres) {
        double wnorm = 0.0;
        if (scalars_dev) {
            gmres_cuda_dnrm2(work_dev, static_cast<std::size_t>(n),
                             scalars_dev + 1, 1, cuda_stream);
            CUDA_MEMCPY_ASYNC(&wnorm, scalars_dev + 1, sizeof(double),
                            cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
            CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
        }
        if (wnorm / bnrm2 < tol) {
            if (work_dev) cudaFree(work_dev);
            if (x_dev) cudaFree(x_dev);
            if (b_dev) cudaFree(b_dev);
            if (h_dev) cudaFree(h_dev);
            if (resid_dev) cudaFree(resid_dev);
            if (scalars_dev) cudaFree(scalars_dev);
            return 0;
        }
    } else
#endif
    if (dnrm2_(n, work) / bnrm2 < tol) {
        return 0;
    }

    iter = 0;

    while (true) {

    /*        Construct the first column of V. */

        double rnorm = 0.0;
#ifdef USE_CUDA_CC
        if (use_cuda_gmres) {
            gmres_cuda_copy(work_dev + 3 * ldw, work_dev, static_cast<std::size_t>(n), cuda_stream);
            if (scalars_dev) {
                gmres_cuda_dnrm2(work_dev + 3 * ldw, static_cast<std::size_t>(n),
                                 scalars_dev + 2, 1, cuda_stream);
                CUDA_MEMCPY_ASYNC(&rnorm, scalars_dev + 2, sizeof(double),
                                cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
                CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
            }
            gmres_cuda_dscal(work_dev + 3 * ldw, 1. / rnorm, static_cast<std::size_t>(n), cuda_stream);
            CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
        } else
#endif
        {
            for (long int idx = 0; idx < n; ++idx) work[3 * ldw + idx] = work[idx];
            rnorm = dnrm2_(n, &work[3 * ldw]);
            dscal_(n, 1. / rnorm, &work[3 * ldw]);
        }

    /*        Initialize S to the elementary vector E1 scaled by RNORM. */

        #ifdef USE_CUDA_CC
        if (use_cuda_gmres) {
            const std::size_t s_len = static_cast<std::size_t>(restrt + 1);
            cudaMemsetAsync(work_dev + ldw, 0, s_len * sizeof(double),
                            reinterpret_cast<cudaStream_t>(cuda_stream));
            CUDA_MEMCPY_ASYNC(work_dev + ldw, &rnorm, sizeof(double),
                            cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
        } else
        #endif
        {
            work[ldw] = rnorm;
            for (long int k = 1; k < n; ++k) work[k + ldw] = 0.;
        }

        for (long int i = 0; i < restrt; ++i) {
            ++iter;

            #ifdef USE_CUDA_CC
        if (use_cuda_gmres) {
            BoundaryElement::matrix_vector_cuda(1., work_dev + static_cast<std::size_t>(3 + i) * static_cast<std::size_t>(ldw),
                                                0., work_dev + 2 * ldw, cuda_stream);
            if (!use_cuda_precond) {
                CUDA_MEMCPY_ASYNC(work + 2 * ldw, work_dev + 2 * ldw,
                                static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
                    CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
                }
            } else
            #endif
            {
                BoundaryElement::matrix_vector(1., &work[(3 + i) * ldw], 0., &work[2 * ldw]);
            }
            #ifdef USE_CUDA_CC
            if (use_cuda_precond) {
                BoundaryElement::precondition_diagonal_cuda(work_dev + 2 * ldw, work_dev + 2 * ldw, cuda_stream);
            } else
            #endif
            {
                if (params_.precondition_) BoundaryElement::precondition_block   (&work[2 * ldw], &work[2 * ldw]);
                else                       BoundaryElement::precondition_diagonal(&work[2 * ldw], &work[2 * ldw]);
            }

        /*           Construct I-th column of H orthnormal to the previous */
        /*           I-1 columns. */
#ifdef USE_CUDA_CC
            if (use_cuda_gmres) {
                if (!use_cuda_precond) {
                CUDA_MEMCPY_ASYNC(work_dev + 2 * ldw, work + 2 * ldw,
                                static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
                }
                gmres_cuda_basis(i + 1, n,
                                 h_dev + static_cast<std::size_t>(i) * static_cast<std::size_t>(ldh),
                                 work_dev + 3 * ldw, static_cast<std::size_t>(ldw),
                                 work_dev + 2 * ldw, cuda_stream);
                CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
            } else
#endif
            {
                basis_(i+1, n, &h[i * ldh], &work[3 * ldw], ldw, &work[2 * ldw]);
            }

        /*           Apply Givens rotations to the I-th column of H. This */
        /*           "updating" of the QR factorization effectively reduces */
        /*           the Hessenberg matrix to upper triangular form during */
        /*           the RESTRT iterations. */
#ifdef USE_CUDA_CC
            if (use_cuda_gmres) {
                gmres_cuda_apply_prev_givens(h_dev, static_cast<std::size_t>(ldh),
                                             i, restrt, cuda_stream);
                gmres_cuda_apply_givens(h_dev, static_cast<std::size_t>(ldh),
                                        work_dev + ldw, i, restrt,
                                        resid_dev, cuda_stream);
                double resid_dev_host = 0.0;
                CUDA_MEMCPY_ASYNC(&resid_dev_host, resid_dev, sizeof(double),
                                cudaMemcpyDeviceToHost,
                                reinterpret_cast<cudaStream_t>(cuda_stream));
                CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
                resid = std::fabs(resid_dev_host) / bnrm2;
            } else
#endif
            {
                for (long int k = 0; k < i; ++k) {
                    drot_(h[k + i * ldh],      h[k + 1 + i * ldh],
                          h[k + restrt * ldh], h[k + (restrt + 1) * ldh]);
                }

            /*           Construct the I-th rotation matrix, and apply it to H so that */
            /*           H(I+1,I) = 0. */
                                      
                drotg_(h[i * (ldh + 1)],      h[i * (ldh + 1) + 1],
                       h[i + (restrt) * ldh], h[i + (restrt + 1) * ldh]);
                                 
                drot_ (h[i * (ldh + 1)],      h[i * (ldh + 1) + 1],
                       h[i + (restrt) * ldh], h[i + (restrt + 1) * ldh]);

            /*           Apply the I-th rotation matrix to [ S(I), S(I+1) ]'. This */
            /*           gives an approximation of the residual norm. If less than */
            /*           tolerance, update the approximation vector X and quit. */
                                
                drot_(work[i + ldw], work[i + ldw + 1],
                      h[i + (restrt) * ldh], h[i + (restrt + 1) * ldh]);
                                
                resid = std::fabs(work[i + 1 + ldw]) / bnrm2;
            }
            std::cout << "GMRES iteration " << std::setw(3) << iter
                      << ": error = " << std::scientific << resid << std::endl;

            if (resid <= tol) {

#ifdef USE_CUDA_CC
                if (use_cuda_gmres) {
                    gmres_update_cuda(i + 1, n, x,
                                      work_dev + 3 * ldw, ldw,
                                      x_dev, h_dev, ldh,
                                      work_dev + ldw,
                                      cuda_stream);
                    CUDA_MEMCPY_ASYNC(x, x_dev, static_cast<std::size_t>(n) * sizeof(double),
                                    cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
                    CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
                } else
#endif
                {
                    update_(i+1, n, x, h, ldh, &work[2 * ldw], &work[ldw], &work[3 * ldw], ldw);
                }

#ifdef USE_CUDA_CC
                if (work_dev) cudaFree(work_dev);
                if (x_dev) cudaFree(x_dev);
                if (b_dev) cudaFree(b_dev);
                if (h_dev) cudaFree(h_dev);
                if (resid_dev) cudaFree(resid_dev);
                if (scalars_dev) cudaFree(scalars_dev);
#endif
                return 0;
            }
        }

    /*        Compute current solution vector X. */

#ifdef USE_CUDA_CC
        if (use_cuda_gmres) {
            gmres_update_cuda(restrt, n, x,
                              work_dev + 3 * ldw, ldw,
                              x_dev, h_dev, ldh,
                              work_dev + ldw,
                              cuda_stream);
        } else
#endif
        {
            update_(restrt, n, x, h, ldh, &work[2 * ldw], &
                    work[ldw], &work[3 * ldw], ldw);
        }

    /*        Compute residual vector R, find norm, then check for tolerance. */

        #ifdef USE_CUDA_CC
        if (use_cuda_gmres) {
            if (b_dev) {
                CUDA_MEMCPY_ASYNC(work_dev + 2 * ldw, b_dev,
                                static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
            } else {
                CUDA_MEMCPY_ASYNC(work_dev + 2 * ldw, b,
                                static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
            }
            BoundaryElement::matrix_vector_cuda(-1., x_dev, 1., work_dev + 2 * ldw, cuda_stream);
            if (!use_cuda_precond) {
                CUDA_MEMCPY_ASYNC(work + 2 * ldw, work_dev + 2 * ldw,
                                static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
                CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
            }
        } else
        #endif
        {
            for (long int idx = 0; idx < n; ++idx) work[2 * ldw + idx] = b[idx];
            BoundaryElement::matrix_vector(-1., x, 1., &work[2 * ldw]);
        }
        #ifdef USE_CUDA_CC
        if (use_cuda_precond) {
            BoundaryElement::precondition_diagonal_cuda(work_dev, work_dev + 2 * ldw, cuda_stream);
        } else
        #endif
        {
            if (params_.precondition_) BoundaryElement::precondition_block   (work, &work[2 * ldw]);
            else                       BoundaryElement::precondition_diagonal(work, &work[2 * ldw]);
        }

#ifdef USE_CUDA_CC
        if (use_cuda_gmres && !use_cuda_precond) {
            CUDA_MEMCPY_ASYNC(work_dev + 2 * ldw, work + 2 * ldw,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
            CUDA_MEMCPY_ASYNC(work_dev, work,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
        }
#endif
        #ifdef USE_CUDA_CC
        if (use_cuda_gmres && scalars_dev) {
            double rnorm = 0.0;
            gmres_cuda_dnrm2(work_dev, static_cast<std::size_t>(n),
                             scalars_dev + 2, 1, cuda_stream);
            CUDA_MEMCPY_ASYNC(&rnorm, scalars_dev + 2, sizeof(double),
                            cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
            CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
            resid = rnorm / bnrm2;
        } else
#endif
        {
            work[restrt + ldw] = dnrm2_(n, work);
            resid = work[restrt + ldw] / bnrm2;
        }

        if (resid <= tol) {
#ifdef USE_CUDA_CC
            if (use_cuda_gmres) {
                CUDA_MEMCPY_ASYNC(x, x_dev, static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
                CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
            }
            if (work_dev) cudaFree(work_dev);
            if (x_dev) cudaFree(x_dev);
            if (b_dev) cudaFree(b_dev);
            if (h_dev) cudaFree(h_dev);
            if (resid_dev) cudaFree(resid_dev);
            if (scalars_dev) cudaFree(scalars_dev);
#endif
            return 0;
        }
        
        if (iter == maxit) {
#ifdef USE_CUDA_CC
            if (use_cuda_gmres) {
                CUDA_MEMCPY_ASYNC(x, x_dev, static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
                CUDA_STREAM_SYNC_AND_CHECK(reinterpret_cast<cudaStream_t>(cuda_stream));
            }
            if (work_dev) cudaFree(work_dev);
            if (x_dev) cudaFree(x_dev);
            if (b_dev) cudaFree(b_dev);
            if (h_dev) cudaFree(h_dev);
            if (resid_dev) cudaFree(resid_dev);
            if (scalars_dev) cudaFree(scalars_dev);
#endif
            return 1;
        }
    } /* Restart. */
}


/*     =============================================================== */
static void update_(long int i, long int n, double* x, const double* h, long int ldh,
                    double* y, const double* s, const double* v, long int ldv)
{
/*     This routine updates the GMRES iterated solution approximation. */
/*     Solve H*Y = S for upper triangualar H. */
/*     Compute current solution vector X = X + V*Y. */

    for (long int idx = 0; idx < i; ++idx) y[idx] = s[idx];
    
    dtrsv_(i, h, ldh, y);
    dgemv_(n, i, v, ldv, y, x);
}


/*     ========================================================= */
static void basis_(long int i, long int n, double* h, double* v, long int ldv, double* w)
{
/*     Construct the I-th column of the upper Hessenberg matrix H */
/*     using the Gram-Schmidt process on V and W. */

    for (long int k = 0; k < i; ++k) {
        h[k] = ddot_(n, w, &v[k * ldv]);
        daxpy_(n, -h[k], &v[k * ldv], w);
    }
    h[i] = dnrm2_(n, w);
    
    for (long int idx = 0; idx < n; ++idx) v[i * ldv + idx] = w[idx];
    dscal_(n, 1. / h[i], &v[i * ldv]);
}


static double dnrm2_(long int n, const double* x)
{
    double norm = 0.;
    for (long int idx = 0; idx < n; ++idx) {
        norm += x[idx] * x[idx];
    }
    return std::sqrt(norm);
}


static void dscal_(long int n, double alpha, double* x)
{
    for (long int idx = 0; idx < n; ++idx) {
        x[idx] *= alpha;
    }
}

static double ddot_(long int n, const double* __restrict x,
                    const double* __restrict y)
{
    double ddot = 0.;
    for (long int idx = 0; idx < n; ++idx) {
        ddot += x[idx] * y[idx];
    }
    return ddot;
}


static void daxpy_(long int n, double alpha, const double* __restrict x,
                   double* __restrict y)
{
    for (long int idx = 0; idx < n; ++idx) {
        y[idx] += alpha * x[idx];
    }
}


static void drot_(double& dx, double& dy, double c, double s)
{
/*  applies a plane rotation. */
    double dtemp = c * dx + s * dy;
    dy = c * dy - s * dx;
    dx = dtemp;
}


static void drotg_(double da, double db, double& c, double& s)
{
/*  construct givens plane rotation. */

    double roe = db;
    if (std::abs(da) > std::abs(db)) roe = da;
    double scale = std::abs(da) + std::abs(db);
    
    if (scale != 0.) {
        double d__1 = da / scale;
        double d__2 = db / scale;
        
        double r = scale * std::sqrt(d__1 * d__1 + d__2 * d__2)
                * (roe >= 0. ? 1. : -1.);

        c = da / r;
        s = db / r;
        
    } else {
        c = 1.;
        s = 0.;
    }
}


static void dtrsv_(long int n, const double* a, long int lda,
                   double* x)
{
/*  solve A*x = b, where A is upper triangular */

    for (long int j = n - 1; j >= 0; --j) {
        if (x[j] != 0.) {
            x[j] /= a[j + j*lda];
            double temp = x[j];
            for (long int i = j - 1; i >= 0; --i) {
                x[i] -= temp * a[i + j*lda];
            }
        }
    }
}


static void dgemv_(long int m, long int n, const double* a, long int lda,
                   const double* x, double* y)
{
/*  Form  y = A*x + y */

    for (long int j = 0; j < n; ++j) {
        if (x[j] != 0.) {
            double temp = x[j];
            for (long int i = 0; i < m; ++i) {
                y[i] += temp * a[i + j*lda];
            }
        }
    }
}
