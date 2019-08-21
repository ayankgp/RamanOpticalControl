//============================================================//
//                                                            //
//   This codes provides parallel omp codes to compute the    //
//  optimal fit for given molecular spectra, using the NLOPT  //
//     package. In addition it computes the optimal Raman     //
//  assisted excitation field to maximize discrimination in   //
//   a given ensemble of biological molecules for which the   //
//      excitation and Raman spectra are known or can be      //
//                  approximately obtained.                   //
//                                                            //
//                @author  A. Chattopadhyay                   //
//    @affiliation Princeton University, Dept. of Chemistry   //
//           @version Updated last on Dec 14 2018             //
//                                                            //
//============================================================//

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <nlopt.h>
#include <omp.h>
#include <time.h>
#define ENERGY_FACTOR 1. / 27.211385
#define ERROR_BOUND 1.0E-8
#define WAVELENGTH2FREQ 1239.84
#define NLOPT_XTOL 1.0E-6

typedef double complex cmplx;

typedef struct parameters{

    cmplx* rho_0;
    int nDIM;
    double* time_A;
    double* time_R;
    int timeDIM_A;
    int timeDIM_R;
    double field_amp_A;
    double field_amp_R;
    double omega_R;
    double omega_v;
    double omega_e;
    int thread_num;
} parameters;

typedef struct molecule{
    int nDIM;
    double* energies;
    double* matrix_gamma_pd;
    double* matrix_gamma_dep;
    double gamma_dep;
    cmplx* rho_0;
    cmplx* mu;
    double d_mu_dx;
    cmplx* field_A;
    cmplx* field_R;
    cmplx* rho;
    cmplx* dyn_rho_A;
    cmplx* dyn_rho_R;
} molecule;

typedef struct mol_system{
    molecule** ensemble;
    molecule* original;
    parameters* params;
    int* count;
} mol_system;

typedef struct supersystem{
    mol_system* mol_system_A;
    mol_system* mol_system_B;
    int* count;
} supersystem;


//====================================================================================================================//
//                                                                                                                    //
//                                        AUXILIARY FUNCTIONS FOR MATRIX OPERATIONS                                   //
//   ---------------------------------------------------------------------------------------------------------------  //
//    Given a matrix or vector and their dimensionality, these routines perform the operations of printing, adding,   //
//        scaling, copiesing to another compatible data structure, finding trace, or computing the maximum element.   //
//                                                                                                                    //
//====================================================================================================================//


void print_complex_mat(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX MATRIX                 //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e + %3.3eJ  ", creal(A[i * nDIM + j]), cimag(A[i * nDIM + j]));
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_complex_vec(const cmplx *A, const int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX VECTOR                 //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e + %3.3eJ  ", creal(A[i]), cimag(A[i]));
	}
	printf("\n");
}

void print_double_mat(const double *A, const int nDIM)
//----------------------------------------------------//
// 	            PRINTS A REAL MATRIX                  //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e  ", A[i * nDIM + j]);
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_double_vec(const double *A, const int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A REAL VECTOR                    //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e  ", A[i]);
	}
	printf("\n");
}


void copy_mat(const cmplx *A, cmplx *B, const int nDIM)
//----------------------------------------------------//
// 	        COPIES MATRIX A ----> MATRIX B            //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] = A[i * nDIM + j];
        }
    }
}

void add_mat(const cmplx *A, cmplx *B, const int nDIM1, const int nDIM2)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM1; i++)
    {
        for(j=0; j<nDIM2; j++)
        {
            B[i * nDIM2 + j] += A[i * nDIM2 + j];
        }
    }
}


void add_vec(const double *A, double *B, const int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        B[i] += A[i];
    }
}

void scale_mat(cmplx *A, const double factor, const int nDIM1, const int nDIM2)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> MATRIX B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM1; i++)
    {
        for(int j=0; j<nDIM2; j++)
        {
            A[i * nDIM2 + j] *= factor;
        }
    }
}


void scale_vec(double *A, const double factor, const int nDIM)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> VECTOR B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        A[i] *= factor;
    }
}


cmplx complex_trace(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	                 RETURNS TRACE[A]                 //
//----------------------------------------------------//
{
    cmplx trace = 0.0 + I * 0.0;
    for(int i=0; i<nDIM; i++)
    {
        trace += A[i*nDIM + i];
    }
    printf("Trace = %3.3e + %3.3eJ  \n", creal(trace), cimag(trace));

    return trace;
}


double complex_abs(cmplx z)
//----------------------------------------------------//
// 	            RETURNS ABSOLUTE VALUE OF Z           //
//----------------------------------------------------//
{

    return sqrt((creal(z)*creal(z) + cimag(z)*cimag(z)));
}


double complex_max_element(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            if(complex_abs(A[i * nDIM + j]) > max_el)
            {
                max_el = complex_abs(A[i * nDIM + j]);
            }
        }
    }
    return max_el;
}

double vec_max(const double *const A, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    for(int i=0; i<nDIM; i++)
    {
        if(A[i] > max_el)
        {
            max_el = A[i];
        }
    }
    return max_el;
}

double vec_sum(const double *const A, const int nDIM)
//----------------------------------------------------//
// 	            RETURNS SUM OF VECTOR ELEMENTS        //
//----------------------------------------------------//
{
    double sum = 0.0;
    for(int i=0; i<nDIM; i++)
    {
        sum += A[i];
    }
    return sum;
}


void CalculateRamanControlField(molecule* mol, const parameters *const params)
//---------------------------------------------------------------------------------//
//                 RETURNS THE RAMAN CONTROL FIELD AS A FUNCTION OF TIME           //
//---------------------------------------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;
    int timeDIM_vib = params->timeDIM_R;

    double* t = params->time_R;
    const double dt = fabs(t[1] - t[0]);

    double A_R = params->field_amp_R;

    for(i=0; i<timeDIM_vib; i++)
    {
        const double time = t[i] + 0.5 * dt;
        mol->field_R[i] = A_R * pow(cos(M_PI*time/(fabs(2*t[0]))), 2) *
                          ((cos((params->omega_R + params->omega_v) * time)) + cos(params->omega_R * time));
    }

}


//====================================================================================================================//
//                                                                                                                    //
//                                CALCULATION OF OPEN QUANTUM SYSTEM DYNAMICS                                         //
//                                                                                                                    //
//====================================================================================================================//


void L_operate(cmplx* Qmat, const cmplx field_ti, molecule* mol)
//----------------------------------------------------//
// 	    RETURNS Q <-- L[Q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
    const int nDIM = mol->nDIM;
    double* matrix_gamma_dep = mol->matrix_gamma_dep;
    double* matrix_gamma_pd = mol->matrix_gamma_pd;
    cmplx* mu = mol->mu;
    double* energies = mol->energies;

    cmplx* Lmat = (cmplx*)calloc(nDIM * nDIM,  sizeof(cmplx));

    for(int m = 0; m < nDIM; m++)
        {
        for(int n = 0; n < nDIM; n++)
            {
                Lmat[m * nDIM + n] -= I * (energies[m] - energies[n]) * Qmat[m * nDIM + n];
                for(int k = 0; k < nDIM; k++)
                {
                    Lmat[m * nDIM + n] += mol->d_mu_dx * I * field_ti * field_ti * (mu[m * nDIM + k] * Qmat[k * nDIM + n] - Qmat[m * nDIM + k] * mu[k * nDIM + n]);

//                    Lmat[m * nDIM + n] -= 0.5 * (matrix_gamma_pd[k * nDIM + n] + matrix_gamma_pd[k * nDIM + m]) * Qmat[m * nDIM + n];
//                    Lmat[m * nDIM + n] += matrix_gamma_pd[m * nDIM + k] * Qmat[k * nDIM + k];
                }

                Lmat[m * nDIM + n] -= matrix_gamma_dep[m * nDIM + n] * Qmat[m * nDIM + n];
            }

        }

    for(int m = 0; m < nDIM; m++)
        {
        for(int n = 0; n < nDIM; n++)
            {
                Qmat[m * nDIM + n] = Lmat[m * nDIM + n];
            }
        }
    free(Lmat);

}

void RamanTransfer(molecule* mol, const parameters *const params)
//---------------------------------------------------------------------------------------------------------------------//
// 	 	 		     CALCULATES FULL LINDBLAD DYNAMICS  DUE TO THE RAMAN CONTROL FIELD FROM TIME 0 to T               //
//--------------------------------------------------------------------------------------------------------------------//
{
    int i, k, t_index;
    const int nDIM = mol->nDIM;

    cmplx *rho_0 = mol->rho_0;
    double *time = params->time_R;

    CalculateRamanControlField(mol, params);
    cmplx* field = mol->field_R;

    double dt = fabs(time[1] - time[0]);

    cmplx* L_rho_func = (cmplx*)calloc(nDIM * nDIM, sizeof(cmplx));
    copy_mat(rho_0, L_rho_func, nDIM);
    copy_mat(rho_0, mol->rho, nDIM);

    for(t_index=0; t_index<params->timeDIM_R; t_index++)
    {
        k=1;
        do
        {
            L_operate(L_rho_func, field[t_index], mol);
            scale_mat(L_rho_func, dt/k, nDIM, nDIM);
            add_mat(L_rho_func, mol->rho, nDIM, nDIM);
            k+=1;
        }while(complex_max_element(L_rho_func, nDIM) > ERROR_BOUND);

        for(i=0; i<nDIM; i++)
        {
            mol->dyn_rho_R[i*params->timeDIM_R + t_index] = mol->rho[i*nDIM + i];
        }

        copy_mat(mol->rho, L_rho_func, nDIM);
    }

    free(L_rho_func);
}