#include "main.h"
#include "typedefs.h"

settings* OPTS = NULL;

/* mode for estimating a model */

int main_fit()
{
  // !!! assert that the right options are set up

  // read the number of vocabulary terms
  int nterms = count_lines(OPTS->vocab_file);

  // read the corpus; exclude response if LDA
  corpus* c;
  if (OPTS->type == LDA_MODEL)
    c = read_corpus(OPTS->docs_file, NULL, TRUE, nterms);
  else
    c = read_corpus(OPTS->docs_file, OPTS->resp_file, FALSE, nterms);

  // fit the model
  model* m = fit(c, OPTS->model_dir);

  // run inference without the response (for posthoc regression)
  m->type = LDA_MODEL;
  OPTS->inf_dir = malloc(1000 * sizeof(char));
  sprintf(OPTS->inf_dir, "%s/training-inference/", OPTS->model_dir);
  write_inference(c, m, OPTS->inf_dir);

  return(0);
}


/* mode for performing inference */

int main_inf()
{
  // !!! assert that the right options are set up

  outlog("what how did this not start? 0");
  int nterms = count_lines(OPTS->vocab_file);
  outlog(" and what about this?? 1");
  model * m = read_model(OPTS->model_dir);
  outlog("up to the 2?");
  corpus* c;
  outlog("3");
  if (m->type == LDA_MODEL)
    c = read_corpus(OPTS->docs_file, NULL, TRUE, nterms);
  else
    c = read_corpus(OPTS->docs_file, OPTS->resp_file, FALSE, nterms);
  outlog("4");

  // enforce full variational inference
  OPTS->var_maxiter = -1;
  printf("5");
  m->type = LDA_MODEL;
  printf("6");
  write_inference(c, m, OPTS->inf_dir);
  printf("7");

  return(0);
}


/* main function */

int main(int argc, char* argv[])
{
  outlog("USAGE: slda --fit/inf --docs <> --resp <> --vocab <> --modeldir <> --type <> <slda/lda> <> --vb <smoothing or 0 for ML> --ntopics <> --infdir <>");
  set_default_settings();
  set_settings_from_args(argc, argv);
  print_settings();

  printf("here");
  if (OPTS->task == 0) {
    printf("main fit");
    main_fit();
  }
  if (OPTS->task == 1) {
    printf("main infer");
    main_inf();
  }
  printf("there");

  return(0);
}
