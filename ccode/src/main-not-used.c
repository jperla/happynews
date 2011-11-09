int main_xval()
{
  // !!! assert that the right OPTS are set up and print OPTS

  // read the corpus
  short incl_resp = 1;
  short collapse_counts = 0;
  if (OPTS->type == LDA_MODEL) {
    incl_resp = 0;
    collapse_counts = 1;
  }
  corpus* c;
  c = read_corpus(OPTS->docs_file, OPTS->resp_file,
                  collapse_counts);

  int* folds = load_int_array(OPTS->fold_file, c->ndocs);
  int nfolds = max_int_array(folds, c->ndocs);
  char model_dir[100];
  int fold;
  for (fold = 1; fold <= nfolds; fold++) {
    outlog("analyzing fold %03d / %03d", fold, nfolds);
    apply_fold(c, fold, folds, 0);
    sprintf(model_dir, "%s/fold%03d", OPTS->model_dir, fold);
    model* mod = fit(c, model_dir);
    mod->type = LDA_MODEL;   // !!! ignore response in inference (yuck)
    char inf_dir[100];
    sprintf(inf_dir, "%s/inference/", model_dir);
    write_inference(c, mod, inf_dir);
    free_model(mod);
  }
  free_corpus(c);
  return(0);
}

