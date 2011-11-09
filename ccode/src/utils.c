#include "utils.h"

static gsl_rng* RANDOM_NUMBER_GENERATOR = NULL;
static gsl_vector* TMP_VECTOR = NULL;
static time_t _time_value;
static struct tm* _time_struct = NULL;
char _time_string[13];

gsl_vector_view get_temp_vector(int size)
{
    if (TMP_VECTOR==NULL)
        TMP_VECTOR = gsl_vector_calloc(SIZE_OF_TEMP_VECTOR);
    gsl_vector_view v = gsl_vector_subvector(TMP_VECTOR, 0, size);
    gsl_vector_set_zero(&v.vector);
    return(v);
}

/* log of the density of a gaussian */

double log_dgauss(double x,
                  double mean,
                  double var)
{
    return(GAUSS_CONST - 0.5 * log(var) - pow((x - mean), 2) / (2 * var));
}

/* seed the random number geneartor with the time */

void seed_random_number_generator()
{
    gsl_rng* random_number_generator = gsl_rng_alloc(gsl_rng_taus);
    long t1;
    (void) time(&t1);
    // !!! DEBUG
    // t1 = 1147530551;
    // t1 = 1178113763;
    // t1 = 1203973386;
    // t1 = 1262233184;
    outlog("random seed = %ld\n", t1);
    gsl_rng_set(random_number_generator, t1);
    RANDOM_NUMBER_GENERATOR = random_number_generator;
}


/* draw a uniform random number */

double runif()
{
    if (RANDOM_NUMBER_GENERATOR == NULL)
        seed_random_number_generator();
    return(gsl_rng_uniform (RANDOM_NUMBER_GENERATOR));
}


/* normalize log proportions */

void log_normalize(gsl_vector* x)
{
    double sum = vget(x, 0);
    int i;
    for (i = 1; i < x->size; i++)
    {
        sum = log_sum(sum, vget(x, i));
    }
    for (i = 0; i < x->size; i++)
    {
        vset(x, i, vget(x, i) - sum);
    }
}


/* given log(a) and log(b), compute log(a+b) */

double log_sum(double log_a, double log_b)
{
  double v;

  if (log_a == -1) return(log_b);

  if (log_a < log_b)
  {
      v = log_b+log(1 + exp(log_a-log_b));
  }
  else
  {
      v = log_a+log(1 + exp(log_b-log_a));
  }
  return(v);
}

/* read/write vectors/matrices from files */


void vct_scanf(const char* filename, gsl_vector* v)
{
    FILE* fileptr;
    fileptr = fopen(filename, "r");
    gsl_vector_fscanf(fileptr, v);
    fclose(fileptr);
    if (VERBOSE) { outlog("read %ld vector from %s", v->size, filename); }
}

void mtx_scanf(const char* filename, gsl_matrix * m)
{
    FILE* fileptr = fopen(filename, "r");
    if (VERBOSE)
    { outlog("reading %ld x %ld matrix from %s",
             m->size1, m->size2, filename); }
    gsl_matrix_fscanf(fileptr, m);
    fclose(fileptr);
}

void vct_printf(const char* filename, gsl_vector* v)
{
    FILE* fileptr;
    fileptr = fopen(filename, "w");
    gsl_vector_fprintf(fileptr, v, "%20.17e");
    fclose(fileptr);
    if (VERBOSE) { outlog( "wrote %ld vector to %s", v->size, filename); }
}


void mtx_printf(const char* filename, const gsl_matrix * m)
{
    if (VERBOSE)
    { outlog( "writing %ld x %ld matrix to %s",
              m->size1, m->size2, filename); }
    FILE* fileptr;
    fileptr = fopen(filename, "w");
    gsl_matrix_fprintf(fileptr, m, "%20.17e");
    fclose(fileptr);
}


/* make a directory */

void make_directory(char* name)
{
    outlog("making directory %s", name);
    mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
}


/* take the exponent of a vector */

void vexp_in_place(gsl_vector* v)
{
    int i;
    for (i = 0; i < v->size; i++)
        vset(v, i, exp(vget(v, i)));
}

void vexp(gsl_vector* v, gsl_vector* exp_v)
{
    // vrda_exp(v->size, v->data, exp_v->data);

    int i;
    for (i = 0; i < v->size; i++)
        vset(exp_v, i, exp(vget(v, i)));
}


/* load an integer array and return the length */

int* load_int_array(char* filename, int size)
{
    int* result = malloc(sizeof(int) * size);
    int v;
    FILE* fileptr = fopen(filename, "r");
    int i = 0;
    while ((fscanf(fileptr, "%10d", &v) != EOF))
    {
        result[i] = v;
        i = i + 1;
    }
    assert(i == size);
    return(result);
}


/* return the maximum value in the int array */

int max_int_array(int* array, int size)
{
  int i;
  int max = array[0];

  for (i = 0; i < size; i++)
    if (array[i] > max) max = array[i];

  return(max);
}

/* compute the quadratic form x A^T y */

double quadratic_form(gsl_vector* x, gsl_matrix* A, gsl_vector* y)
{
    gsl_vector_view tmp = get_temp_vector(x->size);
    gsl_blas_dgemv(CblasNoTrans, 1.0, A, y, 0.0, &tmp.vector);
    return(dot(&tmp.vector, x));
}


/* invert a matrix */

void matrix_inverse(gsl_matrix* m, gsl_matrix* inverse)
{
    gsl_matrix *lu;
    gsl_permutation* p;
    int signum;

    p = gsl_permutation_alloc(m->size1);
    lu = gsl_matrix_alloc(m->size1, m->size2);

    gsl_matrix_memcpy(lu, m);
    gsl_linalg_LU_decomp(lu, p, &signum);
    gsl_linalg_LU_invert(lu, p, inverse);

    gsl_matrix_free(lu);
    gsl_permutation_free(p);
}


/* permute an integer array */

void permute_int_array(int* v, int size)
{
    if (RANDOM_NUMBER_GENERATOR==NULL)
        seed_random_number_generator();

    gsl_ran_shuffle(RANDOM_NUMBER_GENERATOR, v, size, sizeof(int));
}

/* compute the ordering of a vector */

void order(size_t* ordering, const gsl_vector* v)
{
    gsl_sort_index(ordering, v->data, v->stride, v->size);
}


/* standard deviation */

double std_dev(gsl_vector* v)
{
    double ret = gsl_stats_sd(v->data, v->stride, v->size);
    return(ret);
}


/* mean of a vector */

double mean(gsl_vector* v)
{
    double ret = gsl_stats_mean(v->data, v->stride, v->size);
    return(ret);
}

double vsum(gsl_vector* v)
{
  int i;
  double sum = 0;
  for (i = 0; i < v->size; i++)
    sum += vget(v, i);
  return(sum);
}

void compute_mean_and_sd(gsl_vector* v, double* mean, double* sd)
{
    int i;
    double sum_x = 0;
    double sum_x2 = 0;
    for (i = 0; i < v->size; i++)
    {
        sum_x += vget(v, i);
        sum_x2 += square(vget(v, i));
    }
    *mean = sum_x / v->size;
    *sd = sum_x2 - 2 * (*mean) * sum_x + square(*mean) * v->size;
    *sd = sqrt(*sd / (v->size - 1));
}


/* center and scale a vector */

void center_and_scale(gsl_vector* v)
{
    double mean_v, sd_v;
    compute_mean_and_sd(v, &mean_v, &sd_v);
    // center
    gsl_vector_add_constant(v, -mean_v);
    // scale
    gsl_vector_scale(v, 1.0/sd_v);
}


/* entropy of the expectation of a dirichlet */

double entropy_expected_p(gsl_vector* v)
{
    int i;
    double h = 0;

    for (i = 0; i < v->size; i++)
        if (vget(v, i) > 0)
            h -= vget(v, i) * log(vget(v, i));

    // its OK to compute the sum of absolute because v is positive
    double sum = gsl_blas_dasum(v);
    h = h / sum;
    h = h + log(sum);
    return(h);
}


/* entropy of log p */

double entropy_log_p(gsl_vector* v)
{
    int i;
    double h = 0;
    for (i = 0; i < v->size; i++)
        h -= exp(vget(v, i)) * vget(v, i);
    return(h);
}


/* sample K random integers between 0 and N-1 */

void sample_integers(int k, int n, int* result)
{
    int i;
    if (RANDOM_NUMBER_GENERATOR == NULL) seed_random_number_generator();

    gsl_permutation * p = gsl_permutation_alloc(n);
    gsl_permutation_init(p);
    gsl_ran_shuffle(RANDOM_NUMBER_GENERATOR, p->data, n, sizeof(size_t));
    for (i = 0; i < k; i++)
        result[i] = p->data[i];
    gsl_permutation_free(p);
}


/* compute column sums of a matrix */

void compute_column_sums(gsl_matrix* m, gsl_vector* v)
{
    assert(m->size2 == v->size);

    gsl_vector_set_zero(v);
    int r;
    for (r = 0; r < m->size1; r++)
    {
        gsl_vector_view view = gsl_matrix_row(m, r);
        blas_daxpy(1.0, &view.vector, v);
    }
}


void outlog(char* fmt, ...)
{
    if (LOG==NULL) { LOG=stdout; }

    _time_value = time(NULL);
    _time_struct = gmtime(&(_time_value));
    strftime(_time_string, 13, "%Y%m%d|%H%M", _time_struct);

    fprintf(LOG, "[%13s] ", _time_string);

    va_list args;
    va_start(args, fmt);
    vfprintf(LOG, fmt, args);
    fprintf(LOG, "\n");
    va_end(args);
    fflush(LOG);
}

/* predictive r2 */

double predictive_r2(gsl_vector* y, gsl_vector* pred)
{
    assert(y->size == pred->size);

    int i;
    double m = mean(y);
    double num = 0;
    double den = 0;
    for (i = 0; i < y->size; i++)
    {
        num += pow(vget(y,i) - vget(pred, i), 2);
        den += pow(vget(y,i) - m, 2);
    }
    return(1 - num/den);
}


int count_lines(char* filename)
{
  outlog("reading %s", filename);
  FILE* fileptr = fopen(filename, "r");
  int count = 0;
  char line[1000];
  while(fgets(line, 1000, fileptr) != 0) {
    ++count;
  }
  outlog("The number of lines in %s is %d\n", filename, count);
  fclose(fileptr);

  outlog("Closed the file %s", filename);

  return count;
}

