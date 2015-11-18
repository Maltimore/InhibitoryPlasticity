
        #ifndef SENSE_H
        #define SENSE_H
        #include <mkl_service.h>
        #include <mkl_vml.h>
        #include <mkl_dfti.h>
        #include <cstring>
        extern DFTI_DESCRIPTOR_HANDLE hand;
        extern MKL_Complex16 in_cmplx[N_CMPLX], out_cmplx[N_CMPLX], k_cmplx[N_CMPLX];
        DFTI_DESCRIPTOR_HANDLE init_dfti();
        #endif