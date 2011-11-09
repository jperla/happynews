#ifndef UTILS_H
#define UTILS_H

#include "typedefs.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_sort.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <sys/types.h>

#ifdef USE_ACML
#include <acml.h>
#include <acml_mv.h>
#endif

#define GAUSS_CONST -0.91893853320467
#define VERBOSE 0
#define SIZE_OF_TEMP_VECTOR 1000

void outlog(char* fmt, ...);

// get and set from vectors and matrices

static inline double vget(const gsl_vector* v, int i)
{ return(gsl_vector_get(v, i)); };

static inline void vset(gsl_vector* v, int i, double x)
{ gsl_vector_set(v, i, x); };

static inline double mget(const gsl_matrix* m, int i, int j)
{ return(gsl_matrix_get(m, i, j)); };

static inline void mset(gsl_matrix* m, int i, int j, double x)
{ gsl_matrix_set(m, i, j, x); };

// add to matrices and vectors

static inline void madd(gsl_matrix* m, int i, int j, double val)
{ gsl_matrix_set(m, i, j, gsl_matrix_get(m, i, j) + val); };

static inline void vadd(gsl_vector* v, int i, double val)
{ gsl_vector_set(v, i, gsl_vector_get(v, i) + val); };

// get a random number between 0 and 1
double runif();

// normalize log proportions
void log_normalize(gsl_vector* x);

// given log(a) and log(b), return log(a+b)
double log_sum(double log_a, double log_b);

// take the exponent of a vector
void vexp_in_place(gsl_vector* v);
void vexp(gsl_vector* v, gsl_vector* exp_v);

double vsum(gsl_vector* v);

// log of the density of a gaussian
double log_dgauss(double x, double mean, double var);

// square a value
static inline double square(double x)
{ return(x * x); }

// take the log (handle 0 elegantly)
static inline double safe_log(double v)
{
    if (v == 0)
        return(-100);
    else
        return(log(v));
};

static inline void vct_outlog(gsl_vector* v)
{
    int i;
    for (i = 0; i < v->size; i++)
        outlog("[%3d] % 5.3f", i, vget(v, i));
};

// wrappers around gsl functions

static inline double digamma(double x)
{ return(gsl_sf_psi(x)); };

static inline double lngamma(double x)
{ return(gsl_sf_lngamma(x)); };

void vct_scanf(const char* filename, gsl_vector* v);
void mtx_scanf(const char* filename, gsl_matrix * m);
void vct_printf(const char* filename, gsl_vector* v);
void mtx_printf(const char* filename, const gsl_matrix * m);

static inline double dot(gsl_vector* x, gsl_vector* y)
{
    double v;
    gsl_blas_ddot(x, y, &v);
    return(v);
};

double quadratic_form(gsl_vector* x, gsl_matrix* A, gsl_vector* y);
void matrix_inverse(gsl_matrix* m, gsl_matrix* inverse);

// load an int array
int* load_int_array(char* filename, int size);

// os functions
void make_directory(char* name);

// get temporary work space
gsl_vector_view get_temp_vector(int size);

// permute integer vector
void permute_int_array(int* v, int size);

// SD of a vector
double std_dev(gsl_vector* v);

// mean of a vector
double mean(gsl_vector* v);

// center and scale a vector
void center_and_scale(gsl_vector* v);

// entropy of the expectation of a dirichlet
double entropy_expected_p(gsl_vector* v);

// entropy of log p
double entropy_log_p(gsl_vector* v);

static inline void blas_dcopy(const gsl_vector* src,
                              gsl_vector* dest)
{
    // dcopy(src->size, src->data, 1, dest->data, 1);
    gsl_blas_dcopy(src, dest);
};

static inline void blas_daxpy(double a,
                              const gsl_vector* x,
                              gsl_vector* y)
{
    // daxpy(x->size, a, x->data, 1, y->data, 1);
    gsl_blas_daxpy(a, x, y);
};

static inline void blas_dscal(double a, gsl_vector* y)
{
    // dscal(y->size, a, y->data, 1);
    gsl_blas_dscal(a, y);
};

static inline double blas_dasum(const gsl_vector* y)
{
    // return(dasum(y->size, y->data, 1));
    return(gsl_blas_dasum(y));
};

void compute_column_sums(gsl_matrix* m, gsl_vector* v);

// sample K random integers between 0 and N-1
void sample_integers(int k, int n, int* result);

// compute the ordering of a vector
void order(size_t* ordering, const gsl_vector* v);

void compute_mean_and_sd(gsl_vector* v, double* mean, double* sd);
double predictive_r2(gsl_vector* y, gsl_vector* pred);

// return the maximum value in the int array
int max_int_array(int* array, int size);

// count the number of lines in a file
int count_lines(char* filename);

#endif
