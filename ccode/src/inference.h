#ifndef INFERENCE_H
#define INFERENCE_H

#include "typedefs.h"
#include "utils.h"


posterior* posterior_alloc(int type, int ntopics, int nterms);

double lda_var_inference(posterior* post, model* mod, document* doc);

double slda_var_inference(posterior* post, model* mod, document* doc);

double elbo(posterior* post, model* mod, document* doc);

void compute_e_outer_sum_phi_n(posterior* post, document* doc);

void compute_digamma(posterior* post, model* mod);

#endif
