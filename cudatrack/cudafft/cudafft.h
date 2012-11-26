/*

KISS FFT modified to run inside a CUDA kernel

*/
#pragma once
#include <complex>
#include <vector>

namespace kissfft_utils {

#define CFFT_BOTH __device__ __host__

template<typename T>
struct cpx
{
	T x, y;
	CFFT_BOTH T& imag() { return y; }
	CFFT_BOTH T& real() { return x; }
	CFFT_BOTH const T& imag() const { return y; }
	CFFT_BOTH const T& real() const { return x; }
	CFFT_BOTH cpx() : x(0.0f), y(0.0f) {}
	CFFT_BOTH cpx(T a, T b) : x(a), y(b) {}

	CFFT_BOTH cpx operator*(const T& b) const { return cpx(x*b,y*b); }
	CFFT_BOTH cpx& operator*=(const T& b) { x*=b; y*=b; return *this; }
	CFFT_BOTH cpx operator*(const cpx& b) const { return cpx(x*b.x - y*b.y, x*b.y + y*b.x); }
	CFFT_BOTH cpx operator-(const cpx& b) const { return cpx(x-b.x, y-b.y); }
	CFFT_BOTH cpx operator+(const cpx& b) const { return cpx(x+b.x, y+b.y); }
	CFFT_BOTH cpx& operator+=(const cpx& b) { x+=b.x; y+=b.y; return *this; }

	cpx(const std::complex<T>& a) : x(a.real()), y(a.imag()) {}
};

template <typename T_scalar>
struct traits
{
    typedef T_scalar scalar_type;
    typedef cpx<scalar_type> cpx_type;
    void fill_twiddles( cpx_type* dst ,int nfft,bool inverse)const
    {
        T_scalar phinc =  (inverse?2:-2)* acos( (T_scalar) -1)  / nfft;
        for (int i=0;i<nfft;++i)
            dst[i] = exp( std::complex<T_scalar>(0,i*phinc) );
    }

    void prepare(
            std::vector< cpx_type > & twiddles,
            int nfft,bool inverse, 
            std::vector<int> & stageRadix, 
            std::vector<int> & stageRemainder )const
    {
        twiddles.resize(nfft);
        fill_twiddles( &twiddles[0],nfft,inverse);

        //factorize
        //start factoring out 4's, then 2's, then 3,5,7,9,...
        int n= nfft;
        int p=4;
        do {
            while (n % p) {
                switch (p) {
                    case 4: p = 2; break;
                    case 2: p = 3; break;
                    default: p += 2; break;
                }
                if (p*p>n)
                    p=n;// no more factors
            }
            n /= p;
            stageRadix.push_back(p);
            stageRemainder.push_back(n);
        }while(n>1);
    }
};

}



template <typename T,
         typename T_traits=kissfft_utils::traits<T> 
         >
class cudafft
{
public:
    typedef T_traits traits_type;
	typedef typename kissfft_utils::cpx<T> cpx_type;
    typedef typename traits_type::scalar_type scalar_type;

	struct KernelParams 
	{
		char* data;
		int twiddles_offset, remainder_offset, scratchbuf_offset;
		bool inverse;
		int memsize;
		int nfft;

		CFFT_BOTH cpx_type& twiddles(int i) { return ((cpx_type*) (data + twiddles_offset))[i]; }
		CFFT_BOTH int& radix(int i) { return ((int*)data) [i]; }
		CFFT_BOTH int& remainder(int i) { return ((int*)(data + remainder_offset)) [i]; }
		CFFT_BOTH cpx_type* scratchbuf() { return (cpx_type*)(data + scratchbuf_offset); }
	};


    cudafft(int nfft,bool inverse,const traits_type & traits=traits_type() ) 
        :traits(traits)
    {
		std::vector<cpx_type> twiddles;
		std::vector<int> stageRadix;
		std::vector<int> stageRemainder;
		kparams.inverse = inverse;

		traits.prepare(twiddles, nfft, inverse, stageRadix, stageRemainder);
		int maxRadix = *std::max_element(stageRadix.begin(), stageRadix.end());
		int scratchbufsize = maxRadix * sizeof(cpx_type);

		// parameter memory layout: [radix remainders scratchbuf twiddles]
		kparams.remainder_offset = sizeof(int)*stageRadix.size();
		kparams.scratchbuf_offset = kparams.remainder_offset + sizeof(int)*stageRemainder.size();
		kparams.twiddles_offset = kparams.scratchbuf_offset + scratchbufsize;

		kparams_size = kparams.twiddles_offset + sizeof(cpx_type)*twiddles.size();
		
		hostbuf = new char[kparams_size];
		memcpy(hostbuf, &stageRadix[0], sizeof(int) * stageRadix.size());
		memcpy(hostbuf+ kparams.remainder_offset, &stageRemainder[0], sizeof(int) * stageRemainder.size());
		memcpy(hostbuf+ kparams.twiddles_offset, &twiddles[0], sizeof(cpx_type) * twiddles.size()); 

		// Copy to device memory
		cudaMalloc(&kparams.data, kparams_size);
		cudaMemcpy(kparams.data, hostbuf, kparams_size, cudaMemcpyHostToDevice);
		kparams.memsize = kparams_size;
		kparams.nfft = nfft;
    }

	~cudafft() {
		delete[] hostbuf;
		cudaFree(kparams.data);
	}

	static CFFT_BOTH void transform(const cpx_type * src , cpx_type * dst, KernelParams kparm)
	{
		kf_work(0, dst, src, 1,1, &kparm);
	}

	void host_transform(const cpx_type * src , cpx_type * dst)
	{
		KernelParams kp = kparams;
		kp.data = hostbuf;
		kf_work(0, dst, src, 1,1, &kp);
	}

    static CFFT_BOTH void kf_work( int stage,cpx_type * Fout, const cpx_type * f, size_t fstride,size_t in_stride, KernelParams* kparm)
    {
        int p = kparm->radix(stage);
        int m = kparm->remainder(stage);
        cpx_type * Fout_beg = Fout;
        cpx_type * Fout_end = Fout + p*m;

        if (m==1) {
            do{
                *Fout = *f;
                f += fstride*in_stride;
            }while(++Fout != Fout_end );
        }else{
            do{
                // recursive call:
                // DFT of size m*p performed by doing
                // p instances of smaller DFTs of size m, 
                // each one takes a decimated version of the input
                kf_work(stage+1, Fout , f, fstride*p,in_stride,kparm);
                f += fstride*in_stride;
            }while( (Fout += m) != Fout_end );
        }

        Fout=Fout_beg;

        // recombine the p smaller DFTs 
        switch (p) {
            case 2:  
				kf_bfly2(Fout,fstride,m, kparm); 
				break;
           case 3: kf_bfly3(Fout,fstride,m, kparm); break;
           case 4: kf_bfly4(Fout,fstride,m, kparm); break;
           case 5: kf_bfly5(Fout,fstride,m, kparm); break;
           default: kf_bfly_generic(Fout,fstride,m,p,kparm); break;
        }
    }

    // these were #define macros in the original kiss_fft
    static CFFT_BOTH void C_ADD( cpx_type & c,const cpx_type & a,const cpx_type & b) { 
		c.x=a.x+b.x;
		c.y=a.y+b.y;
	}
    static CFFT_BOTH void C_MUL( cpx_type & c,const cpx_type & a,const cpx_type & b) { 
		c=a*b;
	}
    static CFFT_BOTH void C_SUB( cpx_type & c,const cpx_type & a,const cpx_type & b) { 
		c.x=a.x-b.x;
		c.y=a.y-b.y;
	}
    static CFFT_BOTH void C_ADDTO( cpx_type & c,const cpx_type & a) { 
		c.x+=a.x;
		c.y+=a.y;
	}
    static CFFT_BOTH scalar_type S_MUL( const scalar_type & a,const scalar_type & b) { return a*b;}
    static CFFT_BOTH scalar_type HALF_OF( const scalar_type & a) { return a*.5;}
    static CFFT_BOTH void C_MULBYSCALAR(cpx_type & c,const scalar_type & a) {c*=a;}

    static CFFT_BOTH void kf_bfly2( cpx_type * Fout, const size_t fstride, int m, KernelParams* kparm)
    {
        for (int k=0;k<m;++k) {
            cpx_type t = Fout[m+k] * kparm->twiddles(k*fstride);
            Fout[m+k] = Fout[k] - t;
            Fout[k] += t;
        }
    }

	static CFFT_BOTH void kf_bfly4( cpx_type * Fout, const size_t fstride, const size_t m, KernelParams* kparm)
    {
        cpx_type scratch[7];
        int negative_if_inverse = kparm->inverse ? -1 : 1;
        for (size_t k=0;k<m;++k) {
            scratch[0] = Fout[k+m] * kparm->twiddles(k*fstride);
            scratch[1] = Fout[k+2*m] * kparm->twiddles(k*fstride*2);
            scratch[2] = Fout[k+3*m] * kparm->twiddles(k*fstride*3);
            scratch[5] = Fout[k] - scratch[1];

            Fout[k] += scratch[1];
            scratch[3] = scratch[0] + scratch[2];
            scratch[4] = scratch[0] - scratch[2];
            scratch[4] = cpx_type( scratch[4].imag()*negative_if_inverse , -scratch[4].real()* negative_if_inverse );

            Fout[k+2*m]  = Fout[k] - scratch[3];
            Fout[k] += scratch[3];
            Fout[k+m] = scratch[5] + scratch[4];
            Fout[k+3*m] = scratch[5] - scratch[4];
        }
    }

    static CFFT_BOTH void kf_bfly3( cpx_type * Fout, const size_t fstride, const size_t m, KernelParams* kparm)
    {
        size_t k=m;
        const size_t m2 = 2*m;
        cpx_type *tw1,*tw2;
        cpx_type scratch[5];
        cpx_type epi3;
        epi3 = kparm->twiddles(fstride*m);

        tw1=tw2=&kparm->twiddles(0);

        do{
			scratch[1] = Fout[m]* *tw1;
			scratch[2] = Fout[m2]* *tw2;

            C_ADD(scratch[3],scratch[1],scratch[2]);
            C_SUB(scratch[0],scratch[1],scratch[2]);
            tw1 += fstride;
            tw2 += fstride*2;

            Fout[m] = cpx_type( Fout->real() - HALF_OF(scratch[3].real() ) , Fout->imag() - HALF_OF(scratch[3].imag() ) );

            C_MULBYSCALAR( scratch[0] , epi3.imag() );

            C_ADDTO(*Fout,scratch[3]);

            Fout[m2] = cpx_type(  Fout[m].real() + scratch[0].imag() , Fout[m].imag() - scratch[0].real() );

            C_ADDTO( Fout[m] , cpx_type( -scratch[0].imag(),scratch[0].real() ) );
            ++Fout;
        }while(--k);
    }

    static CFFT_BOTH void kf_bfly5( cpx_type * Fout, const size_t fstride, const size_t m, KernelParams* kparm)
    {
        cpx_type *Fout0,*Fout1,*Fout2,*Fout3,*Fout4;
        size_t u;
        cpx_type scratch[13];
        cpx_type * twiddles = &kparm->twiddles(0);
        cpx_type *tw;
        cpx_type ya,yb;
        ya = twiddles[fstride*m];
        yb = twiddles[fstride*2*m];

        Fout0=Fout;
        Fout1=Fout0+m;
        Fout2=Fout0+2*m;
        Fout3=Fout0+3*m;
        Fout4=Fout0+4*m;

        tw=twiddles;
        for ( u=0; u<m; ++u ) {
            scratch[0] = *Fout0;

            C_MUL(scratch[1] ,*Fout1, tw[u*fstride]);
            C_MUL(scratch[2] ,*Fout2, tw[2*u*fstride]);
            C_MUL(scratch[3] ,*Fout3, tw[3*u*fstride]);
            C_MUL(scratch[4] ,*Fout4, tw[4*u*fstride]);

            C_ADD( scratch[7],scratch[1],scratch[4]);
            C_SUB( scratch[10],scratch[1],scratch[4]);
            C_ADD( scratch[8],scratch[2],scratch[3]);
            C_SUB( scratch[9],scratch[2],scratch[3]);

            C_ADDTO( *Fout0, scratch[7]);
            C_ADDTO( *Fout0, scratch[8]);

            scratch[5] = scratch[0] + cpx_type(
                    S_MUL(scratch[7].real(),ya.real() ) + S_MUL(scratch[8].real() ,yb.real() ),
                    S_MUL(scratch[7].imag(),ya.real()) + S_MUL(scratch[8].imag(),yb.real())
                    );

            scratch[6] =  cpx_type( 
                    S_MUL(scratch[10].imag(),ya.imag()) + S_MUL(scratch[9].imag(),yb.imag()),
                    -S_MUL(scratch[10].real(),ya.imag()) - S_MUL(scratch[9].real(),yb.imag()) 
                    );

            C_SUB(*Fout1,scratch[5],scratch[6]);
            C_ADD(*Fout4,scratch[5],scratch[6]);

            scratch[11] = scratch[0] + 
                cpx_type(
                        S_MUL(scratch[7].real(),yb.real()) + S_MUL(scratch[8].real(),ya.real()),
                        S_MUL(scratch[7].imag(),yb.real()) + S_MUL(scratch[8].imag(),ya.real())
                        );

            scratch[12] = cpx_type(
                    -S_MUL(scratch[10].imag(),yb.imag()) + S_MUL(scratch[9].imag(),ya.imag()),
                    S_MUL(scratch[10].real(),yb.imag()) - S_MUL(scratch[9].real(),ya.imag())
                    );

            C_ADD(*Fout2,scratch[11],scratch[12]);
            C_SUB(*Fout3,scratch[11],scratch[12]);

            ++Fout0;++Fout1;++Fout2;++Fout3;++Fout4;
        }
    }

    /* perform the butterfly for one stage of a mixed radix FFT */
    static CFFT_BOTH void kf_bfly_generic( cpx_type * Fout, const size_t fstride, int m, int p, KernelParams* kparm)
    {
        int u,k,q1,q;
        cpx_type * twiddles = &kparm->twiddles(0);
        cpx_type t;
        int Norig = kparm->nfft;
        cpx_type *scratchbuf = kparm->scratchbuf();

        for ( u=0; u<m; ++u ) {
            k=u;
            for ( q1=0 ; q1<p ; ++q1 ) {
                scratchbuf[q1] = Fout[ k  ];
                k += m;
            }

            k=u;
            for ( q1=0 ; q1<p ; ++q1 ) {
                int twidx=0;
                Fout[ k ] = scratchbuf[0];
                for (q=1;q<p;++q ) {
                    twidx += fstride * k;
                    if (twidx>=Norig) twidx-=Norig;
                    C_MUL(t,scratchbuf[q] , twiddles[twidx] );
                    C_ADDTO( Fout[ k ] ,t);
                }
                k += m;
            }
        }
    }

    int kparams_size;
	KernelParams kparams;
	char* hostbuf;
    traits_type traits;
};


