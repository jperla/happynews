#ifndef CORPUS_H
#define CORPUS_H

#include "typedefs.h"
#include "utils.h"
#include <string.h>
#include <assert.h>
#include <gsl/gsl_sort_vector.h>
#include <gsl/gsl_permutation.h>

/**************************************************************************/
/* allocating and deleting a corpus                                       */
/*                                                                        */

// allocate an empty corpus
corpus * corpus_alloc();

// free an allocated corpus
void free_corpus(corpus* dat);

// allocate a document
document * document_alloc(int length);

// free a document
void free_document(document* doc);

// add a document to a corpus
void add_doc_to_corpus(document * doc, corpus * corp);

/* reading a corpus */

// read data from files
corpus * read_corpus(char* docs_file, char* resp_file,
                     int collapse_counts, int nterms);

// !!! NOT YET NEEDED

// print a summary of the data
void print_data_summary(corpus* dat);


/**************************************************************************/
/* misc. useful functions for corpora                                     */
/*                                                                        */

// !!! clean this up: many are not in corpus.c anymore

// create a document from a string
document * string_to_doc(char* vect_string, int collapse_counts);

// for all documents, return the max num of unique terms
int max_nterms(corpus* dat);

// apply a fold to a corpus
void apply_fold(corpus* dat, int fold, int* folds, short include);

// normalize a corpus by document length
void normalize_word_counts(corpus* dat);

// compute which documents to trim.
void trim_docs(corpus* dat, double trim_prop);

// compute the standard deviation of the response
void compute_response_mean_and_sd(corpus* dat,
                                  double* sd,
                                  double* mean);



// select certain documents
corpus* apply_selector(corpus* dat,
                       int* selector,
                       short include); // 0: exclude, 1: include

// compute the fan and lv score
void compute_fan_lv_score(corpus* dat,
                          gsl_vector* term_score,
                          gsl_vector* term_counts);

// compute min and max
double min_label(corpus* dat);
double max_label(corpus* dat);

// set the number of terms by iterating through the docs
void set_nterms(corpus* dat);

// write the selected array (of documents used)
void write_selected_array(corpus* dat, char* dir);

#endif
