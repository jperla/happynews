#include "inference.h"
#include "typedefs.h"

extern settings* OPTS;

#define DEBUG_INFERENCE 0

gsl_vector* SLDATEMP0 = NULL;
gsl_vector* SLDATEMP1 = NULL;
gsl_vector* SLDATEMP2 = NULL;

/* ---------------------------------------------------------------------- */
/* allocation and initialization                                          */
/* ---------------------------------------------------------------------- */

/* allocate the variational posterior */

posterior* posterior_alloc(int type, int ntopics, int nterms)
{
  posterior* post = malloc(sizeof(posterior));
  post->phi = gsl_matrix_calloc(nterms, ntopics);
  post->log_phi = gsl_matrix_calloc(nterms, ntopics);
  post->sum_phi = gsl_vector_calloc(ntopics);
  post->dig_gam = gsl_vector_calloc(ntopics);
  post->e_log_theta = gsl_vector_calloc(ntopics);
  post->niter    = -1;
  post->elbo     =  0;
  post->sq_err   =  0;
  post->fcast    =  0;
  post->elbo_words = gsl_vector_alloc(ntopics);
  if (type == SLDA_MODEL)
  {
    post->e_outer_sum_phi_n = gsl_matrix_calloc(ntopics, ntopics);
  }
  post->elbo_resp = 0;
  return(post);
}


/* initialize the posterior before coordinate ascent. */

void initialize_posterior(posterior* post, model* mod, document* doc)
{
  gsl_vector_set_all(post->sum_phi, doc->total/mod->ntopics);
  gsl_matrix_set_all(post->phi, 1.0/mod->ntopics);
  gsl_matrix_set_all(post->log_phi, -log(mod->ntopics));
  compute_digamma(post, mod);
  if (mod->type == SLDA_MODEL)
  {
    compute_e_outer_sum_phi_n(post, doc);
    // !!! this *workspace* should be part of the posterior
    // !!! and allocated when we allocate the posterior
    if (SLDATEMP0 == NULL)
    {
      SLDATEMP0 = gsl_vector_alloc(mod->ntopics);
      SLDATEMP1 = gsl_vector_alloc(mod->ntopics);
      SLDATEMP2 = gsl_vector_alloc(mod->ntopics);
    }
  }

  post->elbo = 0;
  post->niter = 0;
  post->was_short_circuited = 0;
}


/* ----------------------------------------------------------------------
 * variational inference
 * ---------------------------------------------------------------------- */


/* check if coordinate ascent is stopped */

short check_convergence(posterior* post, double prev_iter_elbo)
{
  double converged = (prev_iter_elbo - post->elbo) / prev_iter_elbo;

  // output if necessary
  if (DEBUG_INFERENCE)
  {
    outlog("iter=%3d; elbo=%1.13e (%1.5e)", post->niter, post->elbo, converged);
  }
  if (((post->elbo - prev_iter_elbo) < -1e-8) &&
      (post->niter > 1))
  {
    outlog("warning: elbo down by %1.13e at %d",
           post->elbo-prev_iter_elbo, post->niter);
  }
  // check the usual convergence criterion
  if (converged <= OPTS->var_conv)
  {
    return(1);
  }
  // check if we are greater than max per-doc change from last EM iteration
  if ((OPTS->max_doc_change > 0) &&
      (post->prev_elbo != 0) &&
      (post->niter > 1) &&
      ((post->elbo - post->prev_elbo) /
       fabs(post->elbo) > OPTS->max_doc_change))
  {
    post->was_short_circuited = 1;
    return(1);
  }
  // check if we've exceeded the maximum number of iterations
  if ((OPTS->var_maxiter > 0) && (post->niter > OPTS->var_maxiter))
  {
    post->was_short_circuited = 1;
    return(1);
  }
  return(0);
}


/* variational inference for LDA */

double lda_var_inference(posterior* post, model* mod, document* doc)
{
  assert(mod->type == LDA_MODEL);
  int n;
  if (doc->total == 0) return(0);

  initialize_posterior(post, mod, doc);
  while (1)
  {
    // check convergence
    post->niter = post->niter + 1;
    double prev_iter_elbo = post->elbo;
    post->elbo = elbo(post, mod, doc);
    if (check_convergence(post, prev_iter_elbo)) break;
    // iterate throught words
    for (n = 0; n < doc->length; n++)
    {
      // update phi_n and phi counts (i.e., gamma)
      gsl_vector_view phi_n = gsl_matrix_row(post->phi, n);
      gsl_vector_view log_phi_n = gsl_matrix_row(post->log_phi, n);
      gsl_vector_view beta_n = gsl_matrix_column(mod->log_beta, doc->words[n]);
      double count = vget(doc->counts, n);
      blas_daxpy(-count, &phi_n.vector, post->sum_phi);
      blas_dcopy(post->dig_gam, &log_phi_n.vector);
      blas_daxpy(1, &beta_n.vector, &log_phi_n.vector);
      log_normalize(&log_phi_n.vector);
      vexp(&log_phi_n.vector, &phi_n.vector);
      blas_daxpy(count, &phi_n.vector, post->sum_phi);
      // if phi_only, then update gamma (via digamma) after each word
      if (OPTS->phi_only) compute_digamma(post, mod);
    }
    // if "blocked", then update gamma (via digamma) after words
    if (!OPTS->phi_only)  compute_digamma(post, mod);
  }
  return(post->elbo);
}


/* supervised LDA variational inference */

double slda_var_inference(posterior* post, model* mod, document* doc)
{
  assert(mod->type == SLDA_MODEL);
  int k, n;
  initialize_posterior(post, mod, doc);
  if (doc->total == 0) return(0);

  // precompute: Y / (N \sigma^2) * \eta
  blas_dcopy(mod->coeff, SLDATEMP0);
  blas_dscal(doc->label/(doc->total*mod->sigma2), SLDATEMP0);
  // precompute: - \eta^2 / (2 N^2 * sigma^2)
  double denom = square(doc->total)*mod->sigma2;
  blas_dcopy(mod->coeff2, SLDATEMP1);
  blas_dscal(-1.0/(2.0*denom), SLDATEMP1);
  while (1)
  {
    post->niter = post->niter + 1;
    double prev_iter_elbo = post->elbo;
    compute_e_outer_sum_phi_n(post, doc); // !!! fold this into elbo?
    post->elbo = elbo(post, mod, doc);
    if (check_convergence(post, prev_iter_elbo)) break;
    for (n = 0; n < doc->length; n++)
    {
      gsl_vector_view phi_n = gsl_matrix_row(post->phi, n);
      gsl_vector_view log_phi_n = gsl_matrix_row(post->log_phi, n);
      gsl_vector_view beta_n = gsl_matrix_column(mod->log_beta, doc->words[n]);
      // remove phi_n from sum phi
      blas_daxpy(-1.0, &phi_n.vector, post->sum_phi);
      // compute proportional to log_&phi_n.vector
      blas_dcopy(post->dig_gam, &log_phi_n.vector); // prior
      blas_daxpy(1, &beta_n.vector, &log_phi_n.vector); // topic
      blas_daxpy(1.0, SLDATEMP0, &log_phi_n.vector); //  (y / N \sigma^2) * \eta
      // (1 / 2 N^2 \sigma^2) (2 (\eta^T \bar{\phi}_{-j}) + eta ** 2)
      blas_daxpy(-dot(post->sum_phi, mod->coeff)/denom,
                 mod->coeff, &log_phi_n.vector);
      blas_daxpy(1.0, SLDATEMP1, &log_phi_n.vector);
      // normalize phi_n
      log_normalize(&log_phi_n.vector);
      vexp(&log_phi_n.vector, &phi_n.vector);
      // add phi_n to sum phi
      blas_daxpy(+1.0, &phi_n.vector, post->sum_phi);;
      // update digamma(gamma)---if non-"blocked"
      if (OPTS->phi_only) compute_digamma(post, mod);
    }
    // update digamma(gamma) if "blocked"
    if (!OPTS->phi_only) compute_digamma(post, mod);
  }
  if (DEBUG_INFERENCE)
  {
    outlog("response = %5.3f", doc->label);
    outlog("nwords   = %5.3f", doc->total);
    outlog("final phi bar:");
    for (k = 0; k < mod->ntopics; k++)
      outlog("[%02d] %5.3f", k, vget(post->sum_phi,k)/doc->total);
  }
  post->fcast = dot(post->sum_phi, mod->coeff) / doc->total;
  assert(!isnan(post->fcast));
  return(post->elbo);
}


/* compute the digamma function of gamma (the variational dirichlet) */

void compute_digamma(posterior* post, model* mod)
{
  int k;
  double dig_sum_gam = digamma(blas_dasum(post->sum_phi) + mod->sum_alpha);
  for (k = 0; k < mod->ntopics; k++) {
    double dig_gam = digamma(vget(post->sum_phi, k) + vget(mod->alpha, k));
    vset(post->dig_gam, k, dig_gam);
    vset(post->e_log_theta, k, dig_gam - dig_sum_gam);
  }
}


/* ---------------------------------------------------------------------- */
/* code for computing the elbo in LDA and sLDA                            */
/* ---------------------------------------------------------------------- */


/* compute the portion of the elbo for the topic proportions */

void set_elbo_prop(posterior* post, model* mod, document* doc)
{
  int k;

  post->elbo_prop = 0;
  post->elbo_prop_ent = 0;

  // assumes gamma = alpha + \sum phi
  // assumes E[log theta] is computed

  post->elbo_prop += lgamma(mod->sum_alpha);
  double sum_gamma = doc->total + mod->sum_alpha;
  post->elbo_prop_ent += - lgamma(sum_gamma);
  for (k = 0; k < mod->ntopics; k++)
  {
    double alpha_k = vget(mod->alpha, k);
    double gamma_k = vget(post->sum_phi, k) + alpha_k;
    double e_log_theta_k = vget(post->e_log_theta, k);
    post->elbo_prop -= lgamma(alpha_k);
    post->elbo_prop += (alpha_k - 1) * e_log_theta_k;
    post->elbo_prop_ent += lgamma(gamma_k);
    post->elbo_prop_ent -= (gamma_k - 1) * e_log_theta_k;
  }
}


/* compute the portion of the elbo for the topic assignments and words */

void set_elbo_z_and_words(posterior* post, model* mod, document* doc)
{
  int n, k;

  post->elbo_z = 0;
  post->elbo_z_ent = 0;
  gsl_vector_set_zero(post->elbo_words);

  for (n = 0; n < doc->length; n++)
  {
    gsl_vector_view phi_n = gsl_matrix_row(post->phi, n);
    gsl_vector_view log_phi_n = gsl_matrix_row(post->log_phi, n);
    gsl_vector_view beta_n = gsl_matrix_column(mod->log_beta, doc->words[n]);
    double cnt = vget(doc->counts, n);
    post->elbo_z_ent += - cnt * dot(&phi_n.vector, &log_phi_n.vector);
    post->elbo_z += cnt * dot(&phi_n.vector, post->e_log_theta);
    for (k = 0; k < mod->ntopics; k++)
    {
      vadd(post->elbo_words, k,
           cnt * vget(&phi_n.vector, k) * vget(&beta_n.vector, k));
    }
  }
}


/* compute the portion of the elbo for the response variable */

void set_elbo_resp(posterior* post, model* mod, document* doc)
{
  post->elbo_resp = square(doc->label);
  post->elbo_resp -=
    2.0 * (doc->label * dot(mod->coeff, post->sum_phi)) / doc->total;
  post->elbo_resp +=
    quadratic_form(mod->coeff, post->e_outer_sum_phi_n, mod->coeff) /
    square(doc->total);
  post->sq_err = post->elbo_resp;
  post->elbo_resp = - post->elbo_resp/ (2 * mod->sigma2);
}


/* compute the elbo */

double elbo(posterior* post, model* mod, document* doc)
{
  set_elbo_prop(post, mod, doc);
  set_elbo_z_and_words(post, mod, doc);
  if (mod->type == SLDA_MODEL)
    set_elbo_resp(post, mod, doc);

  post->elbo =
    post->elbo_prop + post->elbo_z + vsum(post->elbo_words) +
    post->elbo_resp + post->elbo_prop_ent + post->elbo_z_ent;

  assert(!isnan(post->elbo));
  return(post->elbo);
}


// !!! write optimized elbo that assumes that gamma = phi + alpha
// !!! blasify the elbo using a workspace


/* in sLDA, compute the sum of E[X] */

void compute_sum_phi(posterior* post, document* doc)
{
  int n;
  gsl_vector_view row;
  gsl_vector_set_zero(post->sum_phi);
  for (n = 0; n < doc->length; n++)
  {
    row = gsl_matrix_row(post->phi, n);
    blas_daxpy(1.0, &row.vector, post->sum_phi);
  }
}


/* in sLDA, compute E[(sum X_n) (sum X_n)^2] */

void compute_e_outer_sum_phi_n(posterior* post, document* doc)
{
  int n;
  gsl_matrix_set_zero(post->e_outer_sum_phi_n);
  for (n = 0; n < doc->length; n++)
  {
    gsl_vector_view phi_n = gsl_matrix_row(post->phi, n);
    blas_daxpy(-1.0, &phi_n.vector, post->sum_phi);
    gsl_blas_dger(1.0, &phi_n.vector, post->sum_phi,
                  post->e_outer_sum_phi_n);
    blas_daxpy(+1.0, &phi_n.vector, post->sum_phi);
  }
  gsl_vector_view diag = gsl_matrix_diagonal(post->e_outer_sum_phi_n);
  blas_daxpy(+1.0, post->sum_phi, &diag.vector);
}
