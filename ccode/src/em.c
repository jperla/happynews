#include "em.h"

// !!! initialization should be turned into case statements
// !!! initialization should be factored out of this file

#define DEBUG_EM 0
#define WRITE_LAG 10

extern settings* OPTS;

/**************************************************************************
 * functions for initializing the model
 *
 *
 */

/* initialize dirichlet hyperparameters to a constant */

void init_alpha(model* mod)
{
  gsl_vector_set_all(mod->alpha, OPTS->init_alpha_scale / mod->ntopics);
  mod->sum_alpha = blas_dasum(mod->alpha);
}


/* initialize GLM coefficients to a constant */

void init_coeff(model* mod, corpus* dat)
{
  assert(mod->type == SLDA_MODEL);

  double resp_sd;
  double resp_mean;
  compute_response_mean_and_sd(dat, &resp_mean, &resp_sd);

  if (OPTS->coef_init_type == COEF_INIT_ZERO) {
    gsl_vector_set_all(mod->coeff, 0.0);
  }
  else if (OPTS->coef_init_type == COEF_INIT_GRIDDED) {
    int i;
    double coeff = OPTS->min_coef;
    double incr = 2 * fabs(OPTS->min_coef) / mod->ntopics;
    for (i = 0; i < mod->ntopics; i++) {
      vset(mod->coeff, i, coeff);
      coeff = coeff + incr;
    }
  }

  // start off sigma2
  if (OPTS->init_sigma2 > 0)
    mod->sigma2 = OPTS->init_sigma2;
  else
    mod->sigma2 = square(resp_sd);
}



/* initialize topics from document counts */

void init_topics_from_docs(model* mod, corpus* dat)
{
  outlog("initializing from documents");

  int k, d, n;
  gsl_matrix* topics_ss = gsl_matrix_calloc(mod->ntopics, mod->nterms);
  gsl_matrix_set_all(topics_ss, OPTS->init_smooth);
  for (k = 0; k < mod->ntopics; k++)
  {
    outlog("  initializing topic %03d:", k);
    for (d = 0; d < OPTS->init_num_docs; d++)
    {
      int chosen = floor(runif() * dat->ndocs);
      if (dat->selected[chosen] == 0) {
        d = d - 1;
        continue;
      }
      document* doc = dat->docs[chosen];
      outlog("  d = %6d ; y=% 5.3f", chosen, doc->label);
      for (n = 0; n < doc->length; n++)
      {
        int term  = doc->words[n];
        int count = vget(doc->counts, n);
        madd(topics_ss, k, term, count);
      }
    }
  }
  update_topics(mod, topics_ss, OPTS->use_var_bayes, OPTS->vb_smooth);
  gsl_matrix_free(topics_ss);
}


/* initialize topics from other topics */

void init_topics_from_topics(model* mod)
{
  gsl_matrix* beta = gsl_matrix_calloc(mod->ntopics, mod->nterms);
  mtx_scanf(OPTS->init_topics, beta);
  int i, j;
  for (i = 0; i < mod->ntopics; i++)
  {
    for (j = 0; j < mod->nterms; j++)
    {
      mset(mod->log_beta, i, j, safe_log(mget(beta, i, j)));
    }
  }
}


/* initialize topics randomly */

void init_topics_from_random(model* mod)
{
  int i, j;
  gsl_matrix* topics_ss = gsl_matrix_calloc(mod->ntopics, mod->nterms);
  for (i = 0; i < mod->ntopics; i++)
  {
    for (j = 0; j < mod->nterms; j++) {
      double random_number = runif();
      mset(topics_ss, i, j, OPTS->init_smooth + random_number);
    }
  }
  update_topics(mod, topics_ss, OPTS->use_var_bayes, OPTS->vb_smooth);
  gsl_matrix_free(topics_ss);
}


/* initialize topics from the uniform distribution */

void init_topics_from_uniform(model* mod)
{
  gsl_matrix* topics_ss = gsl_matrix_calloc(mod->ntopics, mod->nterms);
  gsl_matrix_set_all(topics_ss, 1.0);
  update_topics(mod, topics_ss, OPTS->use_var_bayes, OPTS->vb_smooth);
  gsl_matrix_free(topics_ss);
}



/* top level: initializing topics, dirichlet, and slda coefficients */

// !!! this should be a very simple initialization.  the complex
// initializations should be explored with R.

void init_model(model* mod, corpus* dat)
{
  // initialize dirichlet parameters
  init_alpha(mod);

  // initialize coefficients
  if (mod->type == SLDA_MODEL) init_coeff(mod, dat);

  // initialize topics
  if (OPTS->topic_init_type == TOPIC_INIT_FROM_TOPICS)
  {
    init_topics_from_topics(mod);
  }
  else if (OPTS->topic_init_type == TOPIC_INIT_FROM_RANDOM)
  {
    init_topics_from_random(mod);
  }
  else if (OPTS->topic_init_type == TOPIC_INIT_FROM_DOCS)
  {
    init_topics_from_docs(mod, dat);
  }
  else if ((OPTS->topic_init_type == TOPIC_INIT_FROM_UNIFORM) &&
           (mod->type == SLDA_MODEL))
  {
    init_topics_from_uniform(mod);
  }
  else
  {
    outlog("ERROR: invalid initialization type/model combination");
    assert(0);
  }
}


/**************************************************************************
 * functions for fitting a model
 *
 *
 */


/* check the convergence of the EM algorithm */

short check_em_convergence(em_fit* the_fit)
{
  the_fit->conv =
    (the_fit->mod->elbo - the_fit->prev_elbo) / fabs(the_fit->prev_elbo);

  outlog("conv calc = %g - %g / %g",
         the_fit->mod->elbo, the_fit->prev_elbo, fabs(the_fit->prev_elbo));

  outlog("iter=%03d ; conv=%5.3e ; elbo=%5.3e",
         the_fit->iter, the_fit->conv, the_fit->mod->elbo);

  // check if the elbo decreased
  if ((the_fit->conv < 0) && (the_fit->iter > 1))
  {
    // backtrack and adjust variational iterations
    // this implements a specific schedule for increasing the # of iterations
    outlog("warning: elbo decreased");
    outlog("backtracking to the previous model");
    model_memcpy(the_fit->mod, the_fit->prev_mod);
    if (OPTS->var_maxiter > -1)
    {
      if (OPTS->var_maxiter > 20) OPTS->var_maxiter += 10;
      else OPTS->var_maxiter += OPTS->var_maxiter;
      outlog("increased max var niter to %d", OPTS->var_maxiter);
    }
  }
  // check if sigma2 has increased
  if ((OPTS->backtrack_sigma2 == TRUE) &&
      (the_fit->prev_mod->sigma2 < the_fit->mod->sigma2))
  {
    outlog("declaring convergence: sigma2 has increased");
    outlog("backtracking to the previous model");
    model_memcpy(the_fit->mod, the_fit->prev_mod);
    return(TRUE);
  }
  // check if EM has converged
  if (((the_fit->conv <= OPTS->em_conv) &&
       (the_fit->iter > OPTS->em_min_iter)) ||
      (the_fit->iter > OPTS->em_max_iter))
  {
    outlog("em convergence criterion is met");
    // check if we are already doing inference until convergence
    if (OPTS->var_maxiter == -1)
    {
      return(TRUE);
    }
    // otherwise, set up final EM iterations
    else {
      outlog("running final EM iterations");
      // set up final EM iterations parameters
      // !!! this is done in a bad hacky way
      OPTS->var_maxiter = -1;
      OPTS->max_doc_change = -1;
      OPTS->em_conv = 1e-4;
      OPTS->em_max_iter = the_fit->iter + 100;
    }
  }
  return(FALSE);
}


/* write various logs and diagnostics after an iteration of EM */

void log_em_iteration(em_fit* the_fit)
{
  time_t t;
  time(&t);

  write_lhood_log(the_fit->lhood_log, the_fit->mod, the_fit->iter,
                  the_fit->conv, t - the_fit->last_log_time);
  the_fit->last_log_time = t;

  if (DEBUG_EM || ((the_fit->iter % WRITE_LAG) == 0))
  {
    char latest_model_dir[100];
    sprintf(latest_model_dir, "%s/latest/", the_fit->dir);
    write_model(the_fit->mod, latest_model_dir);
  }
  if (DEBUG_EM)
  {
    char ss_string[100];
    sprintf(ss_string, "%s/latest/ss", the_fit->dir);
    save_suff_stats(ss_string, the_fit->ss, the_fit->mod->type);
  }
}


/* set up the EM algorithm */

em_fit* setup_em(corpus* dat, char* dir)
{
  char filename[100], last_model_dir[100];
  em_fit* the_fit = malloc(sizeof(em_fit));
  // set directory and data
  the_fit->dir = dir;
  the_fit->dat = dat;
  // make directories and log files
  make_directory(the_fit->dir);
  sprintf(last_model_dir, "%s/latest/", dir);
  sprintf(filename, "%s/lhood.dat", dir);
  the_fit->lhood_log = fopen(filename, "w");
  write_lhood_log_header(the_fit->lhood_log, OPTS->ntopics, OPTS->type);
  // set up sufficient statistics
  the_fit->ss =
    suff_stats_alloc(OPTS->type, OPTS->ntopics,
                     the_fit->dat->nterms, the_fit->dat->ndocs);
  // trim documents

  // !!! i am on the fence about whether to trim within this code
  // !!! this feels like s/t we should do with R
  trim_docs(the_fit->dat, OPTS->trim_prop);

  // allocate the model
  the_fit->mod = model_alloc(OPTS->type, OPTS->ntopics, dat->nterms);
  the_fit->prev_mod = model_alloc(OPTS->type, OPTS->ntopics, dat->nterms);
  // set up various parameters
  the_fit->elbo_vect = gsl_vector_calloc(dat->ndocs);
  the_fit->prev_elbo = 0;
  the_fit->conv = 0;
  the_fit->iter = 0;
  // set up options for variational iterations and document scaling
  if (OPTS->adapt_var_maxiter == TRUE) {
    outlog("adaptively increasing the maximum variational iterations");
    OPTS->var_maxiter = 2;
    OPTS->max_doc_change = -1;
    outlog("beginning number of maximum iterations = %d", OPTS->var_maxiter);
  }
  // initialize the model
  init_model(the_fit->mod, the_fit->dat);
  if (DEBUG_EM)
  {
    sprintf(filename, "%s/initial", dir);
    write_model(the_fit->mod, filename);
  }
  // initialize the time
  time_t t;
  time(&t);
  the_fit->last_log_time = t;

  return(the_fit);
}


/* finish up the EM algorithm */

void finish_em(em_fit* the_fit)
{
  // write the final model
  char final_model_dir[100];
  sprintf(final_model_dir, "%s/final-model", the_fit->dir);
  write_model(the_fit->mod, final_model_dir);
  write_selected_array(the_fit->dat, final_model_dir);
  // close the log file
  fclose(the_fit->lhood_log);
}


/* high level: fitting a model with EM */

model* fit(corpus* dat, char* dir)
{
  em_fit* the_fit = setup_em(dat, dir);
  do {
    // next iteration
    the_fit->iter++;
    outlog("EM iteration %03d", the_fit->iter);
    the_fit->prev_elbo = the_fit->mod->elbo;

    // e step
    reset_suff_stats(the_fit->mod->type, the_fit->ss);
    inference(the_fit->mod, the_fit->dat, the_fit->ss, the_fit->elbo_vect);
    if (check_em_convergence(the_fit)) {
      log_em_iteration(the_fit);
      break;
    }
    log_em_iteration(the_fit);

    // m step
    model_memcpy(the_fit->prev_mod, the_fit->mod);
    update_model(the_fit->mod, the_fit->ss, OPTS->use_var_bayes,
                 OPTS->vb_smooth, TRUE, TRUE);

  } while (TRUE);

  finish_em(the_fit);

  return(the_fit->mod);
}

/*************************************************************************/

/* perform inference on a data set */

// !!! this belongs somewhere else.

void inference(model* mod, corpus* dat, suff_stats* ss,
               gsl_vector* elbo_vect)
{
  posterior* post =
    posterior_alloc(OPTS->type, OPTS->ntopics, max_nterms(dat));

  mod->elbo_prop = 0; mod->elbo_z = 0; mod->elbo_resp = 0; mod->elbo_ent = 0;
  mod->rss = 0; mod->elbo = 0; mod->mean_niter = 0; mod->mean_prop_ent = 0;
  gsl_vector_set_all(mod->elbo_words, 0);

  int nused = 0, nshortcircuits = 0;
  int d, k;
  for (d = 0; d < dat->ndocs; d++) {
    if (dat->selected[d] == 0) continue;
    nused++;
    document* doc_d = dat->docs[d];

    if (OPTS->permute_words)
      permute_int_array(doc_d->words, doc_d->length);

    if (mod->type == LDA_MODEL) {
      if (elbo_vect != NULL) post->prev_elbo = vget(elbo_vect, d);
      lda_var_inference(post, mod, doc_d);
    }
    else {
      slda_var_inference(post, mod, doc_d);
      mod->rss += post->sq_err;
    }

    if (elbo_vect != NULL) vset(elbo_vect, d, post->elbo);
    mod->elbo += post->elbo;
    assert(!isnan(mod->elbo));

    mod->elbo_prop += post->elbo_prop;
    mod->elbo_z += post->elbo_z;
    for (k = 0; k < mod->ntopics; k++)
      vadd(mod->elbo_words, k, vget(post->elbo_words, k));
    mod->elbo_resp += post->elbo_resp;
    mod->elbo_ent += post->elbo_z_ent + post->elbo_prop_ent;
    if (post->was_short_circuited == 1) nshortcircuits += 1;

    mod->mean_niter += post->niter;
    mod->mean_prop_ent += entropy_expected_p(post->sum_phi);

    if (doc_d->total > 0) update_suff_stats(ss, post, doc_d, mod);
  }
  mod->mean_niter /= nused;
  outlog("average number of variational iterations : %0.3g", mod->mean_niter);
  mod->mean_prop_ent /= nused;
  outlog("number of short circuits : %d", nshortcircuits);
}


/* write the likelihood log file header */

void write_lhood_log_header(FILE* lhood_log, int ntopics, short type)
{
  fprintf(lhood_log,
          "%10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s",
          "iter", "time", "conv", "elbo", "elbo.prop", "elbo.z", "elbo.words",
          "elbo.topics", "elbo.ent","rss", "v.niter", "doc.ent");
  int i;
  for (i = 0; i < ntopics; i++)
    fprintf(lhood_log, " %6s.%03d", "ent", i);
  if (type == SLDA_MODEL)
  {
    for (i = 0; i < ntopics; i++)
      fprintf(lhood_log, " %6s.%03d", "eta", i);
    fprintf(lhood_log, " %10s", "sigma2");
  }
  fprintf(lhood_log, "\n");
}


/* write a line of the likelihood log */

void write_lhood_log(FILE* lhood_log, model* mod, int iter,
                     double conv, int time)
{
  int i;

  // iteration number and time
  fprintf(lhood_log, "% 10d % 10d", iter, time);

  // convergence criterion
  if (isinf(conv))
    fprintf(lhood_log, " %10s", "INF");
  else
    fprintf(lhood_log, " % 8.3e", conv);

  // total elbo, LDA elbo, and RSS
  fprintf(lhood_log, " % 8.3e", mod->elbo);
  fprintf(lhood_log, " % 8.3e % 8.3e % 8.3e % 8.3e % 8.3e", mod->elbo_prop,
          mod->elbo_z, vsum(mod->elbo_words), vsum(mod->elbo_topics),
          mod->elbo_ent);
  fprintf(lhood_log, " % 8.3e", mod->rss);

  // mean # variational iterations; mean entropy of \theta
  fprintf(lhood_log, " % 10.0f % 10.2f", mod->mean_niter, mod->mean_prop_ent);

  // topic entropies
  for (i = 0; i < mod->ntopics; i++) {
    gsl_vector_view beta_i = gsl_matrix_row(mod->log_beta, i);
    fprintf(lhood_log, " % 10.2f", entropy_log_p(&beta_i.vector));
  }
  if (mod->type == SLDA_MODEL) {
    // coefficients
    for (i = 0; i < mod->ntopics; i++)
      fprintf(lhood_log, " % 10.3f", vget(mod->coeff, i));
    // sigma^2
    fprintf(lhood_log, " % 10.3f", mod->sigma2);
  }
  fprintf(lhood_log, "\n");
  fflush(lhood_log);
}
