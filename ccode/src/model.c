#include "model.h"

#define WRITE_PHI 0

extern settings* OPTS;

/* -------------------------------------------------------------------
 * reading, writing, allocating, freeing, resetting
 * ------------------------------------------------------------------- */


/* allocate a model */

model* model_alloc(int type,
                   int ntopics,
                   int nterms)
{
  model* mod = malloc(sizeof(model));
  mod->type = type;
  mod->ntopics = ntopics;
  mod->nterms = nterms;
  mod->alpha = gsl_vector_calloc(ntopics);
  mod->log_beta = gsl_matrix_calloc(ntopics, nterms);
  mod->sigma2 = 0;
  mod->coeff = NULL;
  mod->coeff2 = NULL;
  mod->cov = NULL;
  mod->chisq = 0;
  if (mod->type == SLDA_MODEL)
  {
    mod->coeff = gsl_vector_calloc(ntopics);
    mod->coeff2 = gsl_vector_calloc(ntopics);
    gsl_vector_set_zero(mod->coeff);
    gsl_vector_set_zero(mod->coeff2);
    // !!! why is this hardcoded here?
    mod->sigma2 = 0.1;
  }
  mod->elbo_topics = gsl_vector_alloc(ntopics);
  gsl_vector_set_all(mod->elbo_topics, 0);
  mod->elbo_words = gsl_vector_alloc(ntopics);
  gsl_vector_set_all(mod->elbo_words, 0);
  return(mod);
}


void model_memcpy(model* mod, model* src)
{
  mod->type = src->type;
  mod->ntopics = src->ntopics;
  mod->nterms = src->nterms;
  gsl_vector_memcpy(mod->alpha, src->alpha);
  gsl_matrix_memcpy(mod->log_beta, src->log_beta);
  mod->sigma2 = src->sigma2;
  mod->chisq = src->chisq;
  if (mod->type == SLDA_MODEL)
  {
    gsl_vector_memcpy(mod->coeff, src->coeff);
    gsl_vector_memcpy(mod->coeff2, src->coeff2);
  }
  mod->elbo_topics = gsl_vector_alloc(mod->ntopics);
  gsl_vector_memcpy(mod->elbo_topics, src->elbo_topics);
  mod->elbo_words = gsl_vector_alloc(mod->ntopics);
  gsl_vector_memcpy(mod->elbo_words, src->elbo_words);
}


/* free a model */

void free_model(model* mod)
{
  gsl_vector_free(mod->alpha);
  gsl_matrix_free(mod->log_beta);
  if (mod->coeff != NULL)
  {
    gsl_vector_free(mod->coeff);
    gsl_vector_free(mod->coeff2);
    // !!! why is this commented out?
    // gsl_matrix_free(mod->cov);
  }
  free(mod);
}



/* allocate sufficient statistics */

suff_stats* suff_stats_alloc(int type,
                             int ntopics,
                             int nterms,
                             int ndocs)
{
  suff_stats* ss = malloc(sizeof(suff_stats));
  ss->beta = gsl_matrix_calloc(ntopics, nterms);
  ss->alpha = gsl_vector_calloc(ntopics);
  if (type == SLDA_MODEL)
  {
    ss->sum_y_e_bar_z = gsl_vector_calloc(ntopics);
    ss->sum_e_outer_bar_z = gsl_matrix_calloc(ntopics, ntopics);
    ss->sum_e_bar_z = gsl_vector_calloc(ntopics);
    ss->sum_e_bar_z2 = gsl_vector_calloc(ntopics);
    ss->sum_y_2 = 0;
  }
  ss->ndocs = 0;
  return(ss);
}


/* reset sufficient statistics */

void reset_suff_stats(int type,
                      suff_stats* ss)
{
  ss->ndocs = 0;
  gsl_matrix_set_all(ss->beta, 0.001);
  gsl_vector_set_zero(ss->alpha);
  if (type == SLDA_MODEL)
  {
    gsl_vector_set_zero(ss->sum_y_e_bar_z);
    gsl_matrix_set_zero(ss->sum_e_outer_bar_z);
    gsl_vector_set_zero(ss->sum_e_bar_z);
    gsl_vector_set_zero(ss->sum_e_bar_z2);
    ss->sum_y_2 = 0;
  }
}


/* write sufficient statistics */

void write_suff_stats(char* root,
                      suff_stats* ss,
                      int type)
{
  char filename[100];
  make_directory(root);
  sprintf(filename, "%s/SS-BETA.dat", root);
  mtx_printf(filename, ss->beta);
  if (type == SLDA_MODEL)
  {
    sprintf(filename, "%s/SS-SUM-E-OUTER-BAR-Z.dat", root);
    mtx_printf(filename, ss->sum_e_outer_bar_z);
    sprintf(filename, "%s/SS-SUM-Y-E-BAR-Z.dat", root);
    vct_printf(filename, ss->sum_y_e_bar_z);
    sprintf(filename, "%s/SS-SUM-E-BAR-Z.dat", root);
    vct_printf(filename, ss->sum_e_bar_z);
    sprintf(filename, "%s/SS-SUM-E-BAR-Z2.dat", root);
    vct_printf(filename, ss->sum_e_bar_z2);
  }
}


/* write the model */

void write_model(model* mod,
                 char* directory)
{
  char filename[100];

  outlog("writing model to %s", directory);

  make_directory(directory);

  sprintf(filename, "%s/log-beta.dat", directory);
  mtx_printf(filename, mod->log_beta);

  if (mod->type == SLDA_MODEL)
  {
    sprintf(filename, "%s/coeff.dat", directory);
    vct_printf(filename, mod->coeff);
  }

  sprintf(filename, "%s/alpha.dat", directory);
  vct_printf(filename, mod->alpha);

  sprintf(filename, "%s/info.dat", directory);
  FILE* file = fopen(filename, "w");
  fprintf(file, "%d\n%d\n%d\n", mod->type, mod->ntopics, mod->nterms);
  fclose(file);
}


/* read a model */

model* read_model(char* dir)
{
  char fname[300];
  model* mod;
  int ntopics, nterms, type, i;

  sprintf(fname, "%s/info.dat", dir);
  FILE* file = fopen(fname, "r");
  fscanf(file, "%d\n%d\n%d\n", &type, &ntopics, &nterms);
  mod = model_alloc(type, ntopics, nterms);

  outlog("reading model from %s\n  type %d\n  %d topics\n  %d terms",
         dir, mod->type, mod->ntopics, mod->nterms);

  sprintf(fname, "%s/log-beta.dat", dir);
  mtx_scanf(fname, mod->log_beta);

  if (mod->type == SLDA_MODEL)
  {
    sprintf(fname, "%s/coeff.dat", dir);
    vct_scanf(fname, mod->coeff);
    for (i = 0; i < mod->ntopics; i++)
      vset(mod->coeff2, i, square(vget(mod->coeff, i)));
  }

  sprintf(fname, "%s/alpha.dat", dir);
  vct_scanf(fname, mod->alpha);
  mod->sum_alpha = blas_dasum(mod->alpha);

  return(mod);
}


/* -------------------------------------------------------------------
 * fitting the model and updating sufficient statistics
 * ------------------------------------------------------------------- */


/* update sufficient statistics from a variational posterior */

void update_suff_stats(suff_stats* ss, posterior* post,
                       document* doc, model* mod)
{
  int i, n;
  int ntopics = ss->beta->size1;

  for (n = 0; n < doc->length; n++)
  {
    for (i = 0; i < ntopics; i++)
    {
      // topic multinomial distribution update
      double phi;
      phi = mget(post->phi, n, i);
      madd(ss->beta, i, doc->words[n], vget(doc->counts, n) * phi);
    }
  }
  if (mod->type == SLDA_MODEL)
  {
    gsl_blas_daxpy(doc->label/doc->total, post->sum_phi, ss->sum_y_e_bar_z);
    gsl_matrix_scale(post->e_outer_sum_phi_n, 1.0/square(doc->total));
    gsl_matrix_add(ss->sum_e_outer_bar_z, post->e_outer_sum_phi_n);
    for (i = 0; i < mod->ntopics; i++)
    {
      double v = vget(post->sum_phi, i) / doc->total;
      vadd(ss->sum_e_bar_z, i, v);
      vadd(ss->sum_e_bar_z2, i, square(v));
      gsl_blas_daxpy(1.0/doc->total, post->sum_phi, ss->sum_e_bar_z);
    }
    ss->sum_y_2 += square(doc->label);
  }
  // dirichlet
  gsl_vector_add_constant(post->dig_gam, -digamma(doc->total + mod->sum_alpha));
  gsl_vector_add(ss->alpha, post->dig_gam);
  ss->ndocs = ss->ndocs + 1;
}


/*
 * update a model from sufficient statistics.
 *
 * this function sets a model to its maximum likelihood estimates or
 * variational bayes estimates under the sufficient statistics.
 * returns a likelihood adjustment, if using variational bayes.
 *
 */

void update_topics(model* mod, gsl_matrix* ss,
                     short use_var_bayes, double vb_smoothing)
{
  int i, j;
  double sum, elbo;
  for (i = 0; i < mod->ntopics; i++) {
    elbo = 0;
    sum = 0;
    for (j = 0; j < mod->nterms; j++) {
      sum += mget(ss, i, j);
    }
    if (sum < 1) outlog("warning: topic %d has total count < 1", i);
    if (use_var_bayes) {
      elbo -= lgamma(sum + vb_smoothing * mod->nterms);
      sum = digamma(sum + vb_smoothing * mod->nterms);
    } else {
      sum = safe_log(sum);
    }
    for (j = 0; j < mod->nterms; j++) {
      if (use_var_bayes) {
        double gam_j = mget(ss,i,j) + vb_smoothing;
        double dg_gam = digamma(gam_j);
        elbo += (vb_smoothing-gam_j)*(dg_gam-sum)+lgamma(gam_j);
        mset(mod->log_beta, i, j, dg_gam - sum);
      }
      else {
        mset(mod->log_beta, i, j,
             safe_log(mget(ss, i, j)) - sum);
      }
    }
    gsl_vector_set(mod->elbo_topics, i, elbo);
  }
}


/* update the linear model */

void update_linear_model(model* mod, suff_stats* ss)
{
  assert(mod->type == SLDA_MODEL);

  int i;
  gsl_matrix* inverse = gsl_matrix_alloc(mod->ntopics, mod->ntopics);
  matrix_inverse(ss->sum_e_outer_bar_z, inverse);
  gsl_blas_dgemv(CblasNoTrans, 1.0, inverse, ss->sum_y_e_bar_z, 0, mod->coeff);

  for (i = 0; i < mod->ntopics; i++)
  {
    vset(mod->coeff2, i, square(vget(mod->coeff, i)));
  }

  if (OPTS->fit_sigma2 == TRUE)
    mod->sigma2 =
      (ss->sum_y_2 - dot(ss->sum_y_e_bar_z, mod->coeff))/
      ss->ndocs;

  if (0) {
    outlog("coefficients:%s", "");
    vct_outlog(mod->coeff);

    outlog("sigma2: %4.2e", mod->sigma2);
    for (i = 0; i < mod->ntopics; i++) {
      double mean = vget(ss->sum_e_bar_z, i) / ss->ndocs;
      double sd = sqrt((vget(ss->sum_e_bar_z2, i) - mean) / ss->ndocs);
      outlog("E[bar{z}]_%d = %8.4f; SD = %8.4f", i, mean, sd);
    }
  }
}


/* update the slda model */

void update_model(model* mod, suff_stats* ss, short use_var_bayes,
                  double vb_smoothing, short topics_update,
                  short linear_update)
{
  if ((mod->type == SLDA_MODEL) && (linear_update == 1)) {
    update_linear_model(mod, ss);
  }
  if (topics_update) {
    update_topics(mod, ss->beta, use_var_bayes, vb_smoothing);
  }
  // !!! update the dirichlet
}


// !!! this belongs somewhere else

/*
 * write predictions
 *
 * writes predicted values (if appropriate) and posterior dirichlets
 */

void write_inference(corpus* dat, model* mod, char* name)
{
  char filename[100];
  gsl_vector_view row;

  gsl_vector* lhood = gsl_vector_calloc(dat->ndocs);
  gsl_vector* fcast = gsl_vector_calloc(dat->ndocs);
  gsl_matrix* gammas = gsl_matrix_calloc(dat->ndocs, mod->ntopics);
  gsl_vector* response = gsl_vector_calloc(dat->ndocs);

  posterior* post = posterior_alloc(mod->type, mod->ntopics, max_nterms(dat));
  int d;
  make_directory(name);
  if (WRITE_PHI)
  {
    sprintf(filename, "%s/phi/", name);
    make_directory(filename);
  }
  for (d = 0; d < dat->ndocs; d++)
  {
    if (mod->type == SLDA_MODEL)
      vset(lhood, d, slda_var_inference(post, mod, dat->docs[d]));
    else
      vset(lhood, d, lda_var_inference(post, mod, dat->docs[d]));

    if (mod->coeff != NULL)
    {
      post->fcast = dot(post->sum_phi, mod->coeff) / dat->docs[d]->total;
      vset(fcast, d, post->fcast);
      vset(response, d, dat->docs[d]->label);
    }
    row = gsl_matrix_row(gammas, d);
    blas_daxpy(+1, mod->alpha, post->sum_phi);
    gsl_vector* gamma = post->sum_phi;
    gsl_vector_memcpy(&row.vector, gamma);
    blas_daxpy(-1, mod->alpha, post->sum_phi);
    if (WRITE_PHI)
    {
      gsl_matrix_view myphi =
        gsl_matrix_submatrix(post->phi, 0, 0,
                             dat->docs[d]->length, mod->ntopics);

      sprintf(filename, "%s/phi/phi-doc-%03d.dat", name, d);
      mtx_printf(filename, &myphi.matrix);
    }
  }
  make_directory(name);
  sprintf(filename, "%s/gamma.dat", name);
  mtx_printf(filename, gammas);
  sprintf(filename, "%s/lhood.dat", name);
  vct_printf(filename, lhood);
  if (mod->coeff != NULL)
  {
    sprintf(filename, "%s/fcast.dat", name);
    vct_printf(filename, fcast);
  }
}
