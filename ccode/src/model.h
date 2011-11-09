#ifndef MODEL_H
#define MODEL_H

#include <gsl/gsl_multifit.h>
#include "typedefs.h"
#include "utils.h"
#include "inference.h"
#include "corpus.h"

model* model_alloc(int type,
                   int ntopics,
                   int nterms);

void model_memcpy(model* dest, model* src);

void free_model(model* mod);

void save_suff_stats(char* root,
                     suff_stats* ss,
                     int type);

suff_stats* suff_stats_alloc(int type,
                             int ntopics,
                             int nterms,
                             int ndocs);

void reset_suff_stats(int type,
                      suff_stats* ss);

void update_suff_stats(suff_stats* ss,
                       posterior* post,
                       document* doc,
                       model* mod);

void update_model(model* mod, suff_stats* ss, short use_var_bayes,
                  double vb_smoothing, short update_topics,
                  short update_linear);

void update_topics(model* mod, gsl_matrix* ss, short use_var_bayes,
                   double vb_smoothing);

void write_model(model* mod,
                 char* directory);

model* read_model(char* dir);

void write_inference(corpus* dat,
                     model* mod,
                     char* name);

double scaled_var_dir(const gsl_vector* counts, double prior, double prev_elbo,
                      int ngrid, double min_increase, gsl_vector* param,
                      gsl_vector* e_logprob);


#endif
