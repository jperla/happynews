#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "typedefs.h"
#include "em.h"

// !!! CHECK THIS AGAINST THE CODE ON V

void fit_to_selected(double* p,
                     int* select,
                     int* nselect,
                     char** docs,
                     char** resp,
                     char** fit_dir)
{
    fit_param fitp;
    inf_param infp;

    fitp.type = (int) p[0];
    fitp.ntopics = (int) p[1];
    fitp.nrestarts = (double) p[2];
    fitp.em_conv = (double) p[3];
    fitp.permute_words = (int)p[4];
    fitp.em_max_iter = (int)p[5];
    fitp.em_min_iter = (int)p[6];
    fitp.use_var_bayes = (int)p[7];
    fitp.vb_smooth = (double)p[8];
    fitp.trim_prop = (double)p[9];

    fitp.init_type = (int)p[10];
    fitp.init_alpha_scale = (int)p[11];
    fitp.init_num_docs = (int)p[12];
    fitp.init_smooth = (double)p[13];
    fitp.init_coeff = (double)p[14];

    infp.var_conv = (double)p[15];
    infp.var_maxiter = (int) p[16];
    infp.var_miniter = (int) p[17];
    infp.phi_only = (int) p[18];
    infp.collapsed = (int) p[19];

    corpus* dat = read_corpus(*docs, *resp, select, TRUE, FALSE);
    fit(&fitp, &infp, dat, *fit_dir);
}


void predict_selected(double* p,
                      int* select,
                      int* nselect,
                      char** docs,
                      char** resp,
                      char** fit_dir,
                      char** pred_dir,
                      double* predictions,
                      int* npredictions)
{
    inf_param infp;
    infp.var_conv = p[0];
    infp.var_maxiter = (int) p[1];
    infp.var_miniter = (int) p[2];
    infp.phi_only = (int) p[3];
    infp.collapsed = (int) p[4];

    corpus* dat = read_corpus(*docs, *resp, select, TRUE, FALSE);
    assert(dat->ndocs == *npredictions);
    model* mod = read_model(*fit_dir);

    gsl_vector_view row;
    gsl_vector* lhood = gsl_vector_calloc(dat->ndocs);
    gsl_matrix* phibar = gsl_matrix_calloc(dat->ndocs, mod->ntopics);
    posterior* post = posterior_alloc(mod->type,
                                      mod->ntopics,
                                      max_nterms(dat));
    int d;
    for (d = 0; d < dat->ndocs; d++)
    {
        if ((d % 1000) == 0) fprintf(stderr, ".");
        document* doc = dat->docs[d];
        // note: lda inference ignores the response
        vset(lhood, d,
             lda_var_inference(&infp, post, mod, doc));
        row = gsl_matrix_row(phibar, d);
        blas_dscal(1.0/doc->total, post->sum_phi);
        predictions[d] = dot(mod->coeff, post->sum_phi);
        blas_dcopy(post->sum_phi, &row.vector);
    }
    printf("\n");
    make_directory(*pred_dir);
    char filename[100];
    sprintf(filename, "%s/phibar.dat", *pred_dir);
    mtx_printf(filename, phibar);
    sprintf(filename, "%s/lhood.dat", *pred_dir);
    vct_printf(filename, lhood);

    gsl_vector_free(lhood);
    gsl_matrix_free(phibar);
    free_corpus(dat);
    free_model(mod);
}



void fan_lv(int* select,
            int* nselect,
            char** docs,
            char** resp,
            double* fan_lv_score,
            double* term_count,
            int* len_score,
            double* trim)
{

    corpus* dat = read_corpus(*docs, *resp, select, TRUE, FALSE);
    trim_docs(dat, *trim);
    assert(dat->nterms <= *len_score);
    gsl_vector* score = gsl_vector_calloc(dat->nterms);
    gsl_vector* count = gsl_vector_calloc(dat->nterms);
    compute_fan_lv_score(dat, score, count);
    int v;
    for (v = 0; v < dat->nterms; v++)
    {
        fan_lv_score[v] = vget(score, v);
        term_count[v] = vget(count, v);
    }
    gsl_vector_free(score);
    gsl_vector_free(count);
}
