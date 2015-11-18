
        #include <sense.h>
        DFTI_DESCRIPTOR_HANDLE hand;
        MKL_Complex16 in_cmplx[N_CMPLX], out_cmplx[N_CMPLX], k_cmplx[N_CMPLX];
        DFTI_DESCRIPTOR_HANDLE init_dfti()
        {
            DFTI_DESCRIPTOR_HANDLE hand = 0;
            mkl_set_num_threads(1);
            DftiCreateDescriptor(&hand, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N_REAL); //MKL_LONG status
            DftiSetValue(hand, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            DftiSetValue(hand, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            DftiSetValue(hand, DFTI_BACKWARD_SCALE, 1. / N_REAL);
            //if (0 == status) status = DftiSetValue(hand, DFTI_THREAD_LIMIT, 1);
            DftiCommitDescriptor(hand); //if (0 != status) cout << "ERROR, status = " << status << "\n";
            return hand;
        } 