#ifndef EM_H
#define EM_H

#include <stdio.h>
#include "utils.h"
#include "typedefs.h"
#include "math.h"
#include "model.h"
#include "inference.h"
#include "corpus.h"

void init_model(model* mod, corpus* dat);

model* fit(corpus* dat, char* dir);

void write_lhood_log(FILE* lhood_log, model* mod, int iter, double conv,
                     int time_diff);

void write_lhood_log_header(FILE* lhood_log, int ntopics, short type);

void inference(model* mod, corpus* dat, suff_stats* ss,
               gsl_vector* elbo_vect);

#endif
