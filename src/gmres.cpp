#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

#include "boundary_element.h"

#ifdef USE_CUDA_CC
#include <openacc.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gmres_cuda.h"
extern "C" {
    CUcontext acc_get_cuda_context(void) __attribute__((weak));
}
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

#ifdef USE_CUDA_CC
static void basis_cuda_(long int i, long int n, double* h,
                        double* v_dev, long int ldv,
                        double* w_dev,
                        double* partials_dev, std::size_t partials_len,
                        void* stream)
{
    for (long int k = 0; k < i; ++k) {
        h[k] = gmres_cuda_ddot(w_dev, v_dev + k * ldv,
                               static_cast<std::size_t>(n),
                               partials_dev, partials_len, stream);
        gmres_cuda_daxpy(w_dev, v_dev + k * ldv, -h[k],
                         static_cast<std::size_t>(n), stream);
    }
    h[i] = gmres_cuda_dnrm2(w_dev, static_cast<std::size_t>(n),
                            partials_dev, partials_len, stream);
    gmres_cuda_copy(v_dev + i * ldv, w_dev, static_cast<std::size_t>(n), stream);
    gmres_cuda_dscal(v_dev + i * ldv, 1. / h[i], static_cast<std::size_t>(n), stream);
}

static void update_cuda_(long int i, long int n, double* x,
                         const double* h, long int ldh,
                         const double* s,
                         double* v_dev, long int ldv,
                         double* work_dev, long int ldw,
                         double* x_dev, double* h_dev,
                         void* stream)
{
    if (i <= 0) return;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaMemcpyAsync(work_dev + ldw, s, static_cast<std::size_t>(i) * sizeof(double),
                    cudaMemcpyHostToDevice, cuda_stream);
    cudaMemcpyAsync(h_dev, h,
                    static_cast<std::size_t>(ldh) * static_cast<std::size_t>(i) * sizeof(double),
                    cudaMemcpyHostToDevice, cuda_stream);
    gmres_cuda_dtrsv_upper(h_dev, static_cast<std::size_t>(ldh),
                           work_dev + ldw, static_cast<std::size_t>(i), stream);
    gmres_cuda_dgemv(v_dev, static_cast<std::size_t>(ldv),
                     work_dev + ldw, x_dev,
                     static_cast<std::size_t>(n), static_cast<std::size_t>(i), stream);
    cudaMemcpyAsync(x, x_dev, static_cast<std::size_t>(n) * sizeof(double),
                    cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
}
#endif

//*****************************************************************
int BoundaryElement::gmres_(long int n, const double *b, double *x, long int restrt,
                     double* work, long int ldw, double* h, long int ldh,
                     long int& iter, double& resid)
{
    long int maxit = iter;
    double tol = resid;

/*     Store the Givens parameters in matrix H. */
/*     Set initial residual (AV is temporary workspace here). */

#ifdef USE_CUDA_CC
    double* work_dev = nullptr;
    double* partials_dev = nullptr;
    std::size_t partials_len = 0;
    double* x_dev = nullptr;
    double* h_dev = nullptr;
    bool mapped_work = false;
    std::size_t mapped_work_segments = 0;
    bool mapped_x = false;
    void* cuda_stream = nullptr;
    bool use_cuda_gmres = false;
    const char* gmres_env = std::getenv("TABIPB_CUDA_GMRES");
    const char* require_env = std::getenv("TABIPB_CUDA_REQUIRE_GMRES");
    const char* require_all_env = std::getenv("TABIPB_CUDA_REQUIRE_ALL");
    const bool require_gmres = (require_all_env && std::strcmp(require_all_env, "0") != 0) ||
                               (require_env && std::strcmp(require_env, "0") != 0);
    const char* verbose_env = std::getenv("TABIPB_CUDA_GMRES_VERBOSE");
    const bool verbose_gmres = require_gmres || (verbose_env && std::strcmp(verbose_env, "0") != 0);
    use_cuda_gmres = require_gmres || (gmres_env && std::strcmp(gmres_env, "0") != 0);
    const char* debug_map_env = std::getenv("TABIPB_CUDA_GMRES_DEBUG_MAP");
    const bool debug_map = debug_map_env && std::strcmp(debug_map_env, "0") != 0;
    if (use_cuda_gmres) {
        static bool cu_inited = false;
        if (!cu_inited) {
            cuInit(0);
            cu_inited = true;
        }
        acc_wait(acc_async_sync);
        cuda_stream = acc_get_cuda_stream(acc_async_sync);
        CUcontext acc_ctx = nullptr;
        if (acc_get_cuda_context) {
            acc_ctx = acc_get_cuda_context();
        }
        if (acc_ctx == nullptr) {
            cuCtxGetCurrent(&acc_ctx);
        }
        if (acc_ctx != nullptr) {
            cuCtxSetCurrent(acc_ctx);
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
            constexpr std::size_t kBlock = 256;
            partials_len = (static_cast<std::size_t>(n) + kBlock - 1) / kBlock;
            err = cudaMalloc(&partials_dev, partials_len * sizeof(double));
            if (err != cudaSuccess) {
                if (require_gmres) {
                    std::cerr << "[CUDA_GMRES] cudaMalloc(partials) failed: "
                              << cudaGetErrorString(err) << "\n";
                    std::exit(1);
                }
                cudaFree(work_dev);
                work_dev = nullptr;
                use_cuda_gmres = false;
            }
            if (use_cuda_gmres) {
                err = cudaMalloc(&x_dev, static_cast<std::size_t>(n) * sizeof(double));
                if (err != cudaSuccess) {
                    if (require_gmres) {
                        std::cerr << "[CUDA_GMRES] cudaMalloc(x) failed: "
                                  << cudaGetErrorString(err) << "\n";
                        std::exit(1);
                    }
                    cudaFree(partials_dev);
                    cudaFree(work_dev);
                    partials_dev = nullptr;
                    work_dev = nullptr;
                    use_cuda_gmres = false;
                } else {
                    cudaMemcpyAsync(x_dev, x, static_cast<std::size_t>(n) * sizeof(double),
                                    cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
                    const std::size_t h_cols = static_cast<std::size_t>(restrt + 2);
                    const std::size_t h_bytes = static_cast<std::size_t>(ldh) * h_cols * sizeof(double);
                    err = cudaMalloc(&h_dev, h_bytes);
                    if (err != cudaSuccess) {
                        if (require_gmres) {
                            std::cerr << "[CUDA_GMRES] cudaMalloc(h) failed: "
                                      << cudaGetErrorString(err) << "\n";
                            std::exit(1);
                        }
                        cudaFree(x_dev);
                        cudaFree(partials_dev);
                        cudaFree(work_dev);
                        x_dev = nullptr;
                        partials_dev = nullptr;
                        work_dev = nullptr;
                        use_cuda_gmres = false;
                    }
                }
            }
            if (use_cuda_gmres) {
                for (long int k = 0; k < restrt + 4; ++k) {
                    double* seg = work + k * ldw;
                    acc_map_data(seg,
                                 work_dev + static_cast<std::size_t>(k) * static_cast<std::size_t>(ldw),
                                 static_cast<std::size_t>(n) * sizeof(double));
                    ++mapped_work_segments;
                }
                mapped_work = mapped_work_segments > 0;
                acc_map_data(x, x_dev, static_cast<std::size_t>(n) * sizeof(double));
                mapped_x = true;
                if (verbose_gmres) {
                    static bool step2_logged = false;
                    if (!step2_logged) {
                        std::cerr << "[CUDA_GMRES] Step 2 complete: Krylov vectors mapped on device across iterations.\n";
                        step2_logged = true;
                    }
                }
            }
        }
    }
#else
    const bool use_cuda_gmres = false;
#endif

    for (long int idx = 0; idx < n; ++idx) work[2 * ldw + idx] = b[idx];
#ifdef USE_CUDA_CC
    if (use_cuda_gmres) {
        cudaMemcpyAsync(work_dev + 2 * ldw, work + 2 * ldw, static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
    }
#endif

    if (dnrm2_(n, x) != 0.) {
        for (long int idx = 0; idx < n; ++idx) work[2 * ldw + idx] = b[idx];
#ifdef USE_CUDA_CC
        if (use_cuda_gmres) {
            cudaMemcpyAsync(work_dev + 2 * ldw, work + 2 * ldw,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
            BoundaryElement::matrix_vector(-1., x, 1., &work[2 * ldw], true);
            cudaMemcpyAsync(work + 2 * ldw, work_dev + 2 * ldw,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
        } else
#endif
        {
            BoundaryElement::matrix_vector(-1., x, 1., &work[2 * ldw]);
        }
    }

    if (params_.precondition_) BoundaryElement::precondition_block   (work, &work[2 * ldw]);
    else                       BoundaryElement::precondition_diagonal(work, &work[2 * ldw]);

#ifdef USE_CUDA_CC
    if (use_cuda_gmres) {
        cudaMemcpyAsync(work_dev + 2 * ldw, work + 2 * ldw, static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
        cudaMemcpyAsync(work_dev, work, static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
    }
#endif

    double bnrm2 = dnrm2_(n, b);
    if (bnrm2 == 0.) bnrm2 = 1.;
    
#ifdef USE_CUDA_CC
    if (use_cuda_gmres) {
        const double wnorm = gmres_cuda_dnrm2(work_dev, static_cast<std::size_t>(n),
                                              partials_dev, partials_len, cuda_stream);
        if (wnorm / bnrm2 < tol) {
            if (mapped_work) {
                for (long int k = 0; k < restrt + 4; ++k) {
                    double* seg = work + k * ldw;
                    acc_unmap_data(seg);
                }
                mapped_work = false;
                mapped_work_segments = 0;
            }
            if (mapped_x) {
                acc_unmap_data(x);
                mapped_x = false;
            }
            if (work_dev) cudaFree(work_dev);
            if (partials_dev) cudaFree(partials_dev);
            if (x_dev) cudaFree(x_dev);
            if (h_dev) cudaFree(h_dev);
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

#ifdef USE_CUDA_CC
        double rnorm = 0.0;
        if (use_cuda_gmres) {
            gmres_cuda_copy(work_dev + 3 * ldw, work_dev, static_cast<std::size_t>(n), cuda_stream);
            rnorm = gmres_cuda_dnrm2(work_dev + 3 * ldw, static_cast<std::size_t>(n),
                                     partials_dev, partials_len, cuda_stream);
            gmres_cuda_dscal(work_dev + 3 * ldw, 1. / rnorm, static_cast<std::size_t>(n), cuda_stream);
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
        } else
#endif
        {
            for (long int idx = 0; idx < n; ++idx) work[3 * ldw + idx] = work[idx];
            rnorm = dnrm2_(n, &work[3 * ldw]);
            dscal_(n, 1. / rnorm, &work[3 * ldw]);
        }

    /*        Initialize S to the elementary vector E1 scaled by RNORM. */

        work[ldw] = rnorm;
        for (long int k = 1; k < n; ++k) work[k + ldw] = 0.;

        for (long int i = 0; i < restrt; ++i) {
            ++iter;

            #ifdef USE_CUDA_CC
            if (use_cuda_gmres) {
                if (debug_map) {
                    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
                    const bool v_ok = acc_is_present(&work[(3 + i) * ldw], bytes);
                    const bool w_ok = acc_is_present(&work[2 * ldw], bytes);
                    const bool x_ok = acc_is_present(x, bytes);
                    if (!v_ok || !w_ok || !x_ok) {
                        std::cerr << "[CUDA_GMRES] missing mapping before matrix_vector"
                                  << " iter=" << iter << " i=" << i
                                  << " v=" << v_ok << " w=" << w_ok << " x=" << x_ok
                                  << " v_ptr=" << static_cast<const void*>(&work[(3 + i) * ldw])
                                  << " w_ptr=" << static_cast<const void*>(&work[2 * ldw])
                                  << " x_ptr=" << static_cast<const void*>(x)
                                  << " bytes=" << bytes << "\n";
                        std::exit(1);
                    }
                }
                BoundaryElement::matrix_vector(1., &work[(3 + i) * ldw], 0., &work[2 * ldw], true);
                cudaMemcpyAsync(work + 2 * ldw, work_dev + 2 * ldw,
                                static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
                cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
            } else
            #endif
            {
                BoundaryElement::matrix_vector(1., &work[(3 + i) * ldw], 0., &work[2 * ldw]);
            }
            if (params_.precondition_) BoundaryElement::precondition_block   (&work[2 * ldw], &work[2 * ldw]);
            else                       BoundaryElement::precondition_diagonal(&work[2 * ldw], &work[2 * ldw]);

        /*           Construct I-th column of H orthnormal to the previous */
        /*           I-1 columns. */
#ifdef USE_CUDA_CC
            if (use_cuda_gmres) {
                cudaMemcpyAsync(work_dev + 2 * ldw, work + 2 * ldw,
                                static_cast<std::size_t>(n) * sizeof(double),
                                cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
                basis_cuda_(i + 1, n, &h[i * ldh], work_dev + 3 * ldw, ldw,
                            work_dev + 2 * ldw, partials_dev, partials_len, cuda_stream);
                cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
            } else
#endif
            {
                basis_(i+1, n, &h[i * ldh], &work[3 * ldw], ldw, &work[2 * ldw]);
            }

        /*           Apply Givens rotations to the I-th column of H. This */
        /*           "updating" of the QR factorization effectively reduces */
        /*           the Hessenberg matrix to upper triangular form during */
        /*           the RESTRT iterations. */

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
            std::cout << "GMRES iteration " << std::setw(3) << iter
                      << ": error = " << std::scientific << resid << std::endl;

            if (resid <= tol) {

#ifdef USE_CUDA_CC
                if (use_cuda_gmres) {
                    update_cuda_(i + 1, n, x, h, ldh,
                                 &work[ldw],
                                 work_dev + 3 * ldw, ldw,
                                 work_dev, ldw,
                                 x_dev, h_dev,
                                 cuda_stream);
                } else
#endif
                {
                    update_(i+1, n, x, h, ldh, &work[2 * ldw], &work[ldw], &work[3 * ldw], ldw);
                }

#ifdef USE_CUDA_CC
                if (mapped_work) {
                    for (long int k = 0; k < restrt + 4; ++k) {
                        double* seg = work + k * ldw;
                        acc_unmap_data(seg);
                    }
                    mapped_work = false;
                    mapped_work_segments = 0;
                }
                if (mapped_x) {
                    acc_unmap_data(x);
                    mapped_x = false;
                }
                if (work_dev) cudaFree(work_dev);
                if (partials_dev) cudaFree(partials_dev);
                if (x_dev) cudaFree(x_dev);
                if (h_dev) cudaFree(h_dev);
#endif
                return 0;
            }
        }

    /*        Compute current solution vector X. */

#ifdef USE_CUDA_CC
        if (use_cuda_gmres) {
            update_cuda_(restrt, n, x, h, ldh,
                         &work[ldw],
                         work_dev + 3 * ldw, ldw,
                         work_dev, ldw,
                         x_dev, h_dev,
                         cuda_stream);
        } else
#endif
        {
            update_(restrt, n, x, h, ldh, &work[2 * ldw], &
                    work[ldw], &work[3 * ldw], ldw);
        }

    /*        Compute residual vector R, find norm, then check for tolerance. */

        for (long int idx = 0; idx < n; ++idx) work[2 * ldw + idx] = b[idx];
        
        #ifdef USE_CUDA_CC
        if (use_cuda_gmres) {
            if (debug_map) {
                const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
                const bool x_ok = acc_is_present(x, bytes);
                const bool w_ok = acc_is_present(&work[2 * ldw], bytes);
                if (!x_ok || !w_ok) {
                    std::cerr << "[CUDA_GMRES] missing mapping before residual matrix_vector"
                              << " iter=" << iter << " x=" << x_ok << " w=" << w_ok
                              << " x_ptr=" << static_cast<const void*>(x)
                              << " w_ptr=" << static_cast<const void*>(&work[2 * ldw])
                              << " bytes=" << bytes << "\n";
                    std::exit(1);
                }
            }
            cudaMemcpyAsync(work_dev + 2 * ldw, work + 2 * ldw,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
            BoundaryElement::matrix_vector(-1., x, 1., &work[2 * ldw], true);
            cudaMemcpyAsync(work + 2 * ldw, work_dev + 2 * ldw,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream));
            cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream));
        } else
        #endif
        {
            BoundaryElement::matrix_vector(-1., x, 1., &work[2 * ldw]);
        }
        if (params_.precondition_) BoundaryElement::precondition_block   (work, &work[2 * ldw]);
        else                       BoundaryElement::precondition_diagonal(work, &work[2 * ldw]);

#ifdef USE_CUDA_CC
        if (use_cuda_gmres) {
            cudaMemcpyAsync(work_dev + 2 * ldw, work + 2 * ldw,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
            cudaMemcpyAsync(work_dev, work,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(cuda_stream));
        }
#endif
        work[restrt + ldw] = dnrm2_(n, work);
        resid = work[restrt + ldw] / bnrm2;

        if (resid <= tol) {
#ifdef USE_CUDA_CC
            if (mapped_work) {
                for (long int k = 0; k < restrt + 4; ++k) {
                    double* seg = work + k * ldw;
                    acc_unmap_data(seg);
                }
                mapped_work = false;
                mapped_work_segments = 0;
            }
            if (mapped_x) {
                acc_unmap_data(x);
                mapped_x = false;
            }
            if (work_dev) cudaFree(work_dev);
            if (partials_dev) cudaFree(partials_dev);
            if (x_dev) cudaFree(x_dev);
            if (h_dev) cudaFree(h_dev);
#endif
            return 0;
        }
        
        if (iter == maxit) {
#ifdef USE_CUDA_CC
            if (mapped_work) {
                for (long int k = 0; k < restrt + 4; ++k) {
                    double* seg = work + k * ldw;
                    acc_unmap_data(seg);
                }
                mapped_work = false;
                mapped_work_segments = 0;
            }
            if (mapped_x) {
                acc_unmap_data(x);
                mapped_x = false;
            }
            if (work_dev) cudaFree(work_dev);
            if (partials_dev) cudaFree(partials_dev);
            if (x_dev) cudaFree(x_dev);
            if (h_dev) cudaFree(h_dev);
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
