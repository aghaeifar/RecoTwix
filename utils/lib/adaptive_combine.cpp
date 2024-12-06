#include <iostream>
#include <algorithm>    // std::max
#include <complex>
#include <vector>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

#define NDEBUG 1 //disable Eigen's debug assertions (slight speed improvement)

// compile with
// "mex CXXOPTIMFLAGS="\$CXXOPTIMFLAGS -Ofast" adaptiveCombine_mex.cpp"
// for multi-threading support compile with // optimization flags to consider: O2/O3/Ofast
// "mex CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" CXXOPTIMFLAGS="\$CXXOPTIMFLAGS -Ofast" adaptiveCombine_mex.cpp" 

extern "C" {

void adaptive_combine(std::complex<float> *sig, std::complex<float> *wi, float *norm, int *n, int *bs, int *st, int nc_svd, bool modeSliBySli) 
{
    int nc    =  n[0];
    int nx    =  n[1];
    int ny    =  n[2];
    int nz    =  n[3];
    int bsx   = bs[0];
    int bsy   = bs[1];
    int bsz   = bs[2];
    int stx   = st[0];
    int sty   = st[1];
    int stz   = st[2];
    int nxsmall = int(double(nx)/stx + 0.5);
    int nysmall = int(double(ny)/stx + 0.5);
    int nzsmall = int(double(nz)/stz + 0.5);
    long nvox      = long(nx)*long(ny)*long(nz);
    long nvoxsmall = long(nxsmall)*long(nysmall)*long(nzsmall);
    
    if (modeSliBySli)
        bsz = 1;
           
    vector<int> maxCoilSli;
    Array< MatrixXcf,Dynamic,Dynamic > Vsli;
    if (nc_svd) {
        if (nc_svd<0) 
            nc_svd = min(max(9,nc/2),nc);//auto mode
        else
            nc_svd = min(nc_svd,nc);//fail-safe
        
        if (modeSliBySli) {
            Vsli = Array< MatrixXcf,Dynamic,Dynamic > ((nc,nc_svd),nz);
          #pragma omp parallel for
            for (long z=0; z<nzsmall; ++z) {
                MatrixXcf msig(nc,nvox/nz);
                for (long k=0; k<nvox/nz; ++k) {
                    for (int c=0; c<nc; c++) {
                        long ix = c+nc*(k+(nvox/nz)*z);
                        msig(c,k) = sig[ix];
                    }
                }
                JacobiSVD< MatrixXcf > svd(msig*msig.adjoint(), ComputeThinU | ComputeThinV);                
                Vsli(z)=svd.matrixV().block(0,0,nc,nc_svd);
            }
        } else {
            Vsli = Array< MatrixXcf,Dynamic,Dynamic> ((nc,nc_svd),1);
            MatrixXcf msig(nc,nvox);
          #pragma omp parallel for
            for (long k=0; k<nvox; ++k)
                for (int c=0; c<nc; c++)
                    msig(c,k) = sig[c+nc*k];
            JacobiSVD< MatrixXcf > svd(msig*msig.adjoint(), ComputeThinU | ComputeThinV);
            Vsli(0)=svd.matrixV().block(0,0,nc,nc_svd);
        }
    } else { //we need to find maxcoilSli
        if (modeSliBySli) {
            maxCoilSli.resize(nz,0);
          #pragma omp parallel for
            for (long z=0; z<nz; ++z) {
                vector<float> sigSum(nc,0.);
                for (long k=0; k<nvox/nz; ++k) {
                    for (int c=0; c<nc; c++) {
                        long ix = c+nc*(k+(nvox/nz)*z);
                        sigSum[c] += std::abs(sig[ix]);
                    }
                }
                float maxSum = 0.;
                for (int c=0; c<nc; c++) {
                    if (sigSum[c]>maxSum) {
                        maxCoilSli[z] = c;
                        maxSum        = sigSum[c];
                    }
                }
            }
        } else {
            maxCoilSli.resize(1,0);
            vector<float> sigSum(nc,0.);
          #pragma omp parallel for
            for (long k=0; k<nvox; ++k) {
                for (int c=0; c<nc; c++) {
                    long ix = c+nc*k;
                    sigSum[c] += std::abs(sig[ix]);
                }
            }
            float maxSum = 0.;
            for (int c=0; c<nc; c++) {
                if (sigSum[c]>maxSum) {
                    maxCoilSli[0] = c;
                    maxSum        = sigSum[c];
                }
            }
        }
    }
    
    #pragma omp parallel for schedule(dynamic,1) collapse(2)
    for (long z=0; z<nzsmall; ++z) {
        for (long y=0; y<nysmall; ++y) {
            long zmin = max(stz*z-(bsz/2),          0L);
            long zmax = min(stz*z+((bsz+1)/2)-1, nz-1L);
            long lz   = zmax-zmin+1;
            long ymin = max(sty*y-(bsy/2),          0L);
            long ymax = min(sty*y+((bsy+1)/2)-1, ny-1L);
            long ly   = ymax-ymin+1;
            
            MatrixXcf  V;
            MatrixXcf  Vt;
            int maxCoil = 0; //maxCoil==0 in svd mode!
            if (nc_svd) {
                if (modeSliBySli)
                    V = Vsli(z);
                else
                    V = Vsli(0);
                Vt = V.adjoint();
            } else {
                if (modeSliBySli)
                    maxCoil = maxCoilSli[z];
                else
                    maxCoil = maxCoilSli[0];
            }
                
            for (long x=0; x<nxsmall; ++x) {
                long xmin = max(stx*x-(bsx/2),          0L);
                long xmax = min(stx*x+((bsx+1)/2)-1, nx-1L);
                long lx   = xmax-xmin+1;
                
                long cnt = 0;
                MatrixXcf m(nc,lx*ly*lz);
                for (long kz=zmin; kz<=zmax; ++kz) {
                    for (long ky=ymin; ky<=ymax; ++ky) {
                        for (long kx=xmin; kx<=xmax; ++kx) {
                            long ix = kx + nx*(ky + ny*kz);                            
                            for (int c=0; c<nc; ++c)
                                m(c,cnt) = sig[nc*ix+c];
                            cnt++;
                        }
                    }
                }
                
                SelfAdjointEigenSolver< MatrixXcf > eigensolver;
                if (nc_svd) {
                    m = Vt*m; //transform to SVD-space
                    eigensolver.compute(m*m.adjoint());                    
                } else {
                    eigensolver.compute(m*m.adjoint());
                }
                if (eigensolver.info() != Success)
                    abort();
                
                // calc current index
                long ix = x + nxsmall*(y + nysmall*z);
                
                //Eigenvector with max eigenval (=last) gives the correct combination coeffs
                Matrix<complex<float>,Dynamic,1> eigvec = eigensolver.eigenvectors().rightCols(1);
                
                //Correct phase based on coil with max intensity
                eigvec*= exp(complex<float>(0.,-arg(eigvec(maxCoil))));
                
                if (nc_svd) //transform back to original coil space
                    eigvec = V*eigvec;
                
                // calculate norm:
                norm[ix] = 0.;
                for (int c=0; c<nc; ++c)
                    norm[ix] += abs(eigvec(c));
                norm[ix]*=norm[ix];
                                    
                for (int c=0; c<nc; ++c) {
                    wi[nc*ix+c] =  eigvec(c); // std::conj(eigvec(c));
                }
            }
        }
    }
}


} // extern "C" 
