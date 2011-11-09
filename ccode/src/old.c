// UNUSED: initializing sLDA from a grid

void init_slda_grid(model* mod,
                    corpus* dat,
                    double resp_sd,
                    double init_alpha,
                    double init_canonical_coeff)
{
    int k;
    mod->sigma2 = resp_sd * resp_sd;
    outlog("scaling coefficients by response SD %5.3g\n", resp_sd);
    double canon_coeff = - init_canonical_coeff;
    double step = (2.0 * init_canonical_coeff)/(mod->ntopics - 1);
    for (k = 0; k < mod->ntopics; k++)
    {
        vset(mod->coeff, k, canon_coeff * resp_sd);
        vset(mod->coeff2, k, square(vget(mod->coeff, k)));
        canon_coeff += step;
    }
}


    /*
    // alternative way to bin the terms in fan lv initialization
    int groupsize = floor(dat->nterms / mod->ntopics);
    gsl_matrix_set_zero(mod->log_beta);
    int idx = 0;
    int current_topic = 0;
    for (v = 0; v < dat->nterms; v++)
    {
        int term = ordering[v];
        // if the term_count for this word is 0 then skip it
        if (vget(term_count, v) == 0) continue;
        madd(mod->log_beta, current_topic, term, vget(term_count, term));
        idx = idx + 1;
        if ((idx % groupsize)==0) { current_topic++; }
    }
    */


// weird experimental initialization from docs
void init_from_docs(model* mod,
                    corpus* dat,
                    double init_alpha,
                    int num_init,
                    short use_var_bayes,
                    double init_smoothing)

{
    // !!! NOTE: THIS DOES NOT TAKE INTO ACCOUNT TRIMMING

    int k, d;

    gsl_vector_set_all(mod->alpha, init_alpha / mod->ntopics);
    mod->sum_alpha = blas_dasum(mod->alpha);
    outlog("initial alpha:%s", "");
    vct_outlog(mod->alpha);
    // set up the biggest posterior needed
    posterior* post =
        posterior_alloc(mod->type,
                        mod->ntopics,
                        max_nterms(dat));

    // allocate sufficient statistics
    suff_stats* suff_stats =
        suff_stats_alloc(mod->type,
                         mod->ntopics,
                         dat->nterms,
                         dat->ndocs);

    // create a vector of document labels
    double response[dat->ndocs];
    size_t order[dat->ndocs];
    for (d = 0; d < dat->ndocs; d++)
        response[d] = dat->docs[d]->label;

    // sort the document indices
    gsl_sort_index(order, response, 1, dat->ndocs);

    // compute N random documents from each group
    int groupsize = floor(dat->ndocs / mod->ntopics);
    int seeds[groupsize];
    if (num_init == -1)
        num_init = groupsize;

    for (k = 0; k < mod->ntopics; k++)
    {
        outlog("initializing topic %03d:", k);
        sample_integers(num_init, groupsize, seeds);
        for (d = 0; d < num_init; d++)
        {
            document* doc = get_doc(dat, order[k * groupsize + seeds[d]]);
            outlog("- doc %d (resp=% 5.3f)", seeds[d], doc->label);

            // set up the posterior
            gsl_matrix_set_zero(post->phi);
            gsl_vector_view v = gsl_matrix_column(post->phi, k);
            gsl_vector_set_all(&v.vector, 1.0);
            compute_column_sums(post->phi, post->sum_phi);
            if (mod->type == SLDA_MODEL)
            {
                compute_e_outer_sum_phi_n(post, doc);
            }
            // update sufficient statistics
            update_suff_stats(suff_stats, post, doc, mod);
        }
    }
    // update the model
    update_model(mod, suff_stats, 1.0, 10.0, 1, 1);
}


/* read a corpus from a directory */

corpus* read_corpus(char* dir, int incl_resp)
{
    outlog("reading corpus from %s", dir);
    int total_words = 0;
    sprintf(TMP_STRING, "%s/documents.dat", dir);
    corpus* c = corpus_alloc();
    FILE *fileptr = fopen(TMP_STRING, "r");
    int length, count, word, n;
    while ((fscanf(fileptr, "%10d", &length) != EOF))
    {
        c->docs = (document**) realloc(c->docs, sizeof(document*)*(c->ndocs+1));
        c->docs[c->ndocs] = document_alloc(length);
        document* doc = c->docs[c->ndocs];
        doc->label = 0;
        for (n = 0; n < length; n++)
        {
            fscanf(fileptr, "%10d:%10d", &word, &count);
            word = word - VOCABULARY_OFFSET;
            doc->words[n] = word;
            vset(doc->counts, n, (double) count);
            doc->total += count;
            total_words += count;
            if (word >= c->nterms) { c->nterms = word + 1; }
        }
        c->ndocs = c->ndocs + 1;
    }
    fclose(fileptr);
    outlog("read %d documents from %s", c->ndocs, TMP_STRING);
    outlog("read %d words containing %d terms", total_words, c->nterms);

    if (incl_resp)
    {
        sprintf(TMP_STRING, "%s/response.dat", dir);
        gsl_vector* annotations = gsl_vector_alloc(c->ndocs);
        vct_scanf(TMP_STRING, annotations);
        int i;
        for (i = 0; i < annotations->size; i++)
            get_doc(c, i)->label = vget(annotations, i);
    }
    c->directory = dir;
    return(c);
}

