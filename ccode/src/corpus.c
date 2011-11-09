#include "corpus.h"

extern settings* OPTS;

/* allocate a new empty corpus */

corpus * corpus_alloc()
{
  corpus* corp = malloc(sizeof(corpus));
  corp->nterms = 0;
  corp->ndocs = 0;
  corp->nterms = 0;
  corp->docs = malloc(sizeof(document*));
  return(corp);
}

/* free a corpus */

void free_corpus(corpus* dat)
{
  int d;
  for (d = 0; d < dat->ndocs; d++)
  {
    free_document(dat->docs[d]);
  }
  free(dat->selected);
  free(dat);
}

/* allocate a document of the specified number of unique terms */

document * document_alloc(int length)
{
  document* d;
  d = malloc(sizeof(document));
  d->length = length;
  d->total = 0;
  d->words = malloc(sizeof(int)*length);
  if (length > 0)
    d->counts = gsl_vector_calloc(length);
  else
    d->counts = gsl_vector_calloc(1);
  d->label = 0;
  return(d);
}

/* free a document */

void free_document(document* doc)
{
  free(doc->words);
  gsl_vector_free(doc->counts);
  free(doc);
}


/* return a new document from sparse vector string */

document * string_to_doc(char* vect_string, int collapse_counts)
{
  int length, n, j, word, count;
  char * token = strtok(vect_string, " ");
  sscanf(token, "%d", &length);
  document * doc = document_alloc(length);
  if (!collapse_counts)
    doc->length = 0;
  for (n = 0; n < length; n++)
  {
    token = strtok(NULL, " ");
    sscanf(token, "%10d:%10d", &word, &count);
    word = word - VOCABULARY_OFFSET;
    assert(word >= 0);
    if (TERM_OCC) count = 1;
    if (LOG_OCC) count = floor(log(count));
    if (SCALE_OCC) count = floor(count / SCALE_OCC_VAL);
    if (count == 0) { continue; }

    doc->total += count;
    if (collapse_counts)
    {
      vset(doc->counts, n, (double) count);
      doc->words[n] = word;
    }
    // if we not collapsing, add <count> copies of term
    else
    {
      for (j = 0; j < count; j++)
      {
        doc->words =
          (int*) realloc(doc->words,
                         sizeof(int) * (doc->length + 1));
        doc->words[doc->length] = word;
        doc->length += 1;
      }
    }
  }
  // if we are not collapsing the counts, set them all to 1.0
  if (collapse_counts == 0)
  {
    gsl_vector_free(doc->counts);
    doc->counts = gsl_vector_alloc(doc->total);
    gsl_vector_set_all(doc->counts, 1.0);
  }
  return(doc);
}


/* read a corpus from files. */

corpus * read_corpus(char* docs, char* resp,
                     int collapse_counts, int nterms)
{
  outlog("reading corpus from %s", docs);
  outlog("assuming %d terms", nterms);
  if (resp != NULL)
    outlog("reading response from %s", resp);

  if (collapse_counts == 0)
    outlog("not collapsing counts.");
  else
    outlog("collapsing counts.");

  if (TERM_OCC) outlog("term occ : term count is unchanged.");
  if (LOG_OCC) outlog("log occ : term count is set to its log.");
  if (SCALE_OCC) outlog("scale occ : term count is divided by %5.3f.",
                        SCALE_OCC_VAL);

  corpus* c = corpus_alloc();
  FILE *fileptr = fopen(docs, "r");
  int length, count, word, n, j, total_words=0;

  // !!! can we read this a line at a time? and then use string_to_doc?
  // !!! in general, this function should be divided up into several functions

  c->nterms = nterms;
  while ((fscanf(fileptr, "%10d", &length) != EOF))
  {
    c->docs = (document**) realloc(c->docs,
                                   sizeof(document*)*(c->ndocs+1));
    if (collapse_counts == 0)
    {
      c->docs[c->ndocs] = document_alloc(1);
      c->docs[c->ndocs]->length = 0;
    }
    else
    {
      c->docs[c->ndocs] = document_alloc(length);
    }
    document* doc = c->docs[c->ndocs];

    for (n = 0; n < length; n++)
    {
      fscanf(fileptr, "%10d:%10d", &word, &count);
      word = word - VOCABULARY_OFFSET;
      assert(word < nterms);
      if (TERM_OCC) count = 1;
      if (LOG_OCC) count = floor(log(count));
      if (SCALE_OCC) count = floor(count / SCALE_OCC_VAL);
      if (count == 0) { continue; }

      total_words += count;
      doc->total += count;

      if (collapse_counts == 0)
      {
        for (j = 0; j < count; j++)
        {
          doc->words = (int*) realloc(doc->words,
                                      sizeof(int) * (doc->length + 1));
          doc->words[doc->length] = word;
          doc->length += 1;
        }
      }
      else
      {
        vset(doc->counts, n, (double) count);
        doc->words[n] = word;
      }
    }

    if (collapse_counts == 0)
    {
      gsl_vector_free(doc->counts);
      if (doc->total > 0) {
        doc->counts = gsl_vector_alloc(doc->total);
        gsl_vector_set_all(doc->counts, 1.0);
      }
      else {
        doc->counts = gsl_vector_calloc(1);
      }
    }
    c->ndocs = c->ndocs + 1;
  }
  fclose(fileptr);
  outlog("read %d documents from %s", c->ndocs, TMP_STRING);
  outlog("# observed words = %8d", total_words);
  outlog("# terms          = %8d", c->nterms);

  // load the response if appropriate
  if (resp != NULL)
  {
    gsl_vector* annotations = gsl_vector_alloc(c->ndocs);
    vct_scanf(resp, annotations);
    int i;
    for (i = 0; i < annotations->size; i++)
      c->docs[i]->label = vget(annotations, i);
  }

  // select all the documents by default
  c->selected = malloc(sizeof(short) * (c->ndocs));
  for (n = 0; n < c->ndocs; n++) c->selected[n] = 1;
  c->nselected = c->ndocs;

  return(c);
}

/*
  compute the mean and standard deviation of the response.  takes into
  account whether each document has been selected.

  !!! this should be edited to no longer select documents
  !!! or, we can remove this function and compute the training sd from R

*/

void compute_response_mean_and_sd(corpus* dat, double* mean, double* sd)
{
  int idx = 0;
  double resp[dat->nselected];
  int d;
  for (d = 0; d < dat->ndocs; d++)
  {
    if (dat->selected[d]==0) continue;
    resp[idx++] = dat->docs[d]->label;
  }
  *sd = gsl_stats_sd(resp, 1, idx);
  *mean = gsl_stats_mean(resp, 1, idx);
  outlog("corpus response mean=%5.3f, sd=%5.3f", *mean, *sd);
}

/* return the maximum number of unique terms in a collection */

int max_nterms(corpus* dat)
{
  int d;
  int max_length = 0;
  for (d = 0; d < dat->ndocs; d++)
  {
    int length = dat->docs[d]->length;
    if (length > max_length)
      max_length = length;
  }
  return(max_length);
}

/*
  populate the selected array based on the absolute value of the
  response variable. e.g. trim_prop = 0.1 means trim the top 10%
*/

void trim_docs(corpus* dat, double trim_prop)
{
  gsl_vector* abs_resp = gsl_vector_calloc(dat->ndocs);
  gsl_permutation * perm = gsl_permutation_alloc(dat->ndocs);
  gsl_permutation* rank = gsl_permutation_alloc(dat->ndocs);
  int d;
  for (d = 0; d < dat->ndocs; d++)
  {
    vset(abs_resp, d, -fabs(dat->docs[d]->label));
  }
  gsl_sort_vector_index(perm, abs_resp);
  gsl_permutation_inverse(rank, perm);
  int trim = ceil(trim_prop * dat->ndocs);
  outlog("trimming %d documents", trim);
  int count = 0;
  for (d = 0; d < dat->ndocs; d++)
  {
    if (rank->data[d] < trim)
    {
      dat->selected[d] = 0;
      outlog("- [%04d] trimming doc %d; response = %5.3f", ++count, d, dat->docs[d]->label);
    }
  }
  gsl_vector_free(abs_resp);
  gsl_permutation_free(perm);
  gsl_permutation_free(rank);
}

/* write the selected array */

void write_selected_array(corpus* dat, char* dir)
{
  char filename[100];
  sprintf(filename, "%s/selected.dat", dir);
  FILE* file_ptr = fopen(filename, "w");
  int d;
  for (d = 0; d < dat->ndocs; d++)
    fprintf(file_ptr, "%d\n", dat->selected[d]);
  fclose(file_ptr);
}
