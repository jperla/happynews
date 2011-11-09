#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multifit.h>
#include "assert.h"
#include "utils.h"
#include <string.h>
#include <getopt.h>

void set_default_settings();
void print_settings();
void set_settings_from_args(int argc, char* argv[]);

// parameters

#define LDA_MODEL 0
#define SLDA_MODEL 1

#define TOPIC_INIT_FROM_DOCS 0
#define TOPIC_INIT_FROM_TOPICS 3
#define TOPIC_INIT_FROM_RANDOM 4
#define TOPIC_INIT_FROM_UNIFORM 5

#define COEF_INIT_ZERO 0
#define COEF_INIT_GRIDDED 1

#define VOCABULARY_OFFSET 0 // useful for data from R with min vocab of 1

// !!! these should be options

#define TERM_OCC 0
#define LOG_OCC 0
#define SCALE_OCC 0
#define SCALE_OCC_VAL 10.0

#define TRUE 1
#define FALSE 0

// for use in various functions
char TMP_STRING[100];
FILE* LOG;

// a single document and response
typedef struct document
{
  int* words;          // unique terms in a document
  gsl_vector* counts;  // counts of each word
  int length;          // number of terms
  double total;        // sum of counts
  double label;        // annotation
} document;


// a collection of annotated documents
typedef struct corpus
{
  int nterms;      // size of the vocabulary
  int ndocs;       // number of documents in the corpus
  document** docs; // the documents
  short* selected; // which documents to analyze
  int nselected;   // number of documents selected
  int collapsed;   // whether the counts of the documents are collapsed
} corpus;


// !!! do we want to divide this up into a model and estimation
// summary statistics?

// a model
typedef struct model
{
  short type;            // model type (LDA_MODEL,...)
  int ntopics;           // number of topics
  int nterms;            // vocabulary size
  gsl_vector* alpha;     // dirichlet hyperparameter
  gsl_matrix* log_beta;  // per topic word distribution
  double sum_alpha;      // sum of the alpha
  gsl_vector* coeff;     // linear model coefficients
  gsl_vector* coeff2;    // linear model coefficients squared
  double sigma2;         // linear model variance
  gsl_matrix* cov;       // covariance matrix of linear model !!! unused?
  double chisq;          // chi squared statistic of linear model

  double elbo_prop;        // the elbo terms on the topic proportions
  double elbo_z;           // the elbo terms for the prob of the z vars
  gsl_vector* elbo_words;  // the elbo terms for the prob of the words
  double elbo_resp;        // the elbo terms for the prob of the response
  gsl_vector* elbo_topics; // the elbo terms for the prob of the topics
  double elbo_ent;         // elbo entropy terms
  double elbo;             // total elbo
  double mean_niter;       // mean number of variational iterations
  double mean_prop_ent;    // mean entropy of document proportions
  double rss;              // residual sum of squares
} model;


// sufficient statistics
typedef struct suff_stats
{
  gsl_matrix* beta;           // suff stats for per topic word dist
  gsl_vector* alpha;          // suff stats for dirichlet
  int ndocs;                  // number of documents (needed for dirichlet)

  // for sLDA
  gsl_vector* sum_y_e_bar_z;
  gsl_matrix* sum_e_outer_bar_z;
  gsl_vector* sum_e_bar_z;
  gsl_vector* sum_e_bar_z2;
  double sum_y_2;

} suff_stats;


// posterior distribution
typedef struct posterior
{
  gsl_matrix* phi;               // per word multinomial
  gsl_matrix* log_phi;           // log of the per-word multinomial
  gsl_vector* dig_gam;           // digamma(gamma)
  gsl_vector* sum_phi;           // sum of the phi's (+ alpha = gamma)
  gsl_matrix* e_outer_sum_phi_n; // E[(sum phi_n) (sum phi_n)^T]
  gsl_vector* e_log_theta;       // E[log theta]

  int niter;                     // number of iterations

  double elbo_prop;              // the elbo terms on the topic proportions
  double elbo_z;                 // elbo terms for the prob of the z vars
  gsl_vector* elbo_words;        // elbo terms for the prob of the words
  double elbo_resp;              // elbo terms for the prob of the response
  double elbo_z_ent;             // elbo entropy term for z
  double elbo_prop_ent;          // elbo entropy term for prop
  double elbo;                   // total elbo
  double sq_err;                 // squared error of the prediction
  short was_short_circuited;     // was variational inference short circuited?
  double prev_elbo;              // the previous iteration's elbo for this doc

  double fcast;                  // the forecast
} posterior;


// EM fit object
typedef struct em_fit
{
  model* mod;
  model* prev_mod;
  corpus* dat;
  char* dir;
  double prev_elbo;
  double conv;
  int iter;
  gsl_vector* elbo_vect;
  suff_stats* ss;
  time_t last_log_time;
  FILE* lhood_log;
} em_fit;


// global settings for fitting and inference

typedef struct settings
{
  // settings specified by the user

  int type;                 // type of model (lda or slda)
  int ntopics;              // number of topics
  int task;                 // inference or estimation
  char* docs_file;          // the file for the corpus
  char* vocab_file;         // the file for the vocabulary
  char* resp_file;          // the file for the response variable
  char* model_dir;          // the location of the model (for in or out)
  char* inf_dir;            // where to put the inference files

  // settings for running EM

  double em_conv;           // EM convergence criterion
  int permute_words;        // should we permute the words at each iteration?
  int em_max_iter;          // max number of iterations of EM
  int em_min_iter;          // min number of iterations of EM
  int use_var_bayes;        // should we fit with variational bayes?
  double vb_smooth;         // variational bayes smoothing parameter
  double trim_prop;         // proportion of documents to trim
  short adapt_var_maxiter;  // should we adapt the maximum iterations in EM
  short fit_sigma2;         // should we fit sigma^2
  short backtrack_sigma2;   // declare convergence when sigma2 goes up?

  // settings associated with initializing EM

  short coef_init_type;     // type of coefficient initialiation (see consts)
  short topic_init_type;    // type of topic initialization (see consts)
  char* init_topics;        // filename of initial topics
  double init_alpha_scale;  // initial scaling of the topics dirichlet
  int init_num_docs;        // number of documents to initialize with
  double init_smooth;       // initial smoothing of topics
  double min_coef;          // minimum coefficient of linear model
  double init_sigma2;       // initial sigma^2; <= 0 -> "use sample sigma^2"

  // settings associated with variational inference

  double var_conv;          // variational convergence criterion
  int var_maxiter;          // max number of variational iterations
  int var_miniter;          // min number of variational iterations
  int phi_only;             // should we only optimize phi?
  int collapsed;            // should we employ collapsed inference?
  double max_doc_change;    // maximimum inference change per doc

} settings;

#endif
