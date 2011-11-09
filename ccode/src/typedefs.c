#include "typedefs.h"

extern settings* OPTS;

/* for handling command line options */

static struct option long_options[] =
{
  {"docs",     required_argument, 0, 'd'},
  {"response", required_argument, 0, 'r'},
  {"vocab",    required_argument, 0, 'v'},
  {"modeldir", required_argument, 0, 'm'},
  {"infdir"  , required_argument, 0, 'o'},
  {"type"    , required_argument, 0, 't'},
  {"ntopics" , required_argument, 0, 'k'},
  {"vb"      , required_argument, 0, 's'},
  {"fit"     , no_argument      , 0, 'f'},
  {"infer"   , no_argument      , 0, 'i'},
  {0, 0, 0, 0} // !!! why do we need this?
};


/* decode the type string */

int decode_type(char* str)
{
  if (strcmp(str, "lda") == 0) return(LDA_MODEL);
  if (strcmp(str, "slda") == 0) return(SLDA_MODEL);
  outlog("ERROR: invalid model type");
  assert(0);
}


/* set settings use long options */

void set_settings_from_args(int argc, char* argv[])
{
  int c = 0, idx;
  while (c != -1)
  {
    c = getopt_long(argc, argv, "fik:t:m:r:d:o:a:b:c:s:", long_options, &idx);
    switch (c)
    {
      outlog("option %s", long_options[idx].name);
      if (optarg) outlog("argument = %s", optarg);

    case 'd': OPTS->docs_file = optarg; break;
    case 'v': OPTS->vocab_file = optarg; break;
    case 'r': OPTS->resp_file = optarg; break;
    case 'm': OPTS->model_dir = optarg; break;
    case 'o': OPTS->inf_dir = optarg; break;
    case 't': OPTS->type = decode_type(optarg); break;
    case 'k': OPTS->ntopics = atoi(optarg); break;
    case 'f': OPTS->task = 0; break;
    case 'i': OPTS->task = 1; break;
    case 'x': OPTS->task = 2; break;
    case 's': OPTS->vb_smooth = atof(optarg); break;

    default: break;
    }
    if (OPTS->vb_smooth == 0)
      OPTS->use_var_bayes = FALSE;
    else
      OPTS->use_var_bayes = TRUE;
  }
}


/* set the default settings */

void set_default_settings()
{
  OPTS = malloc(sizeof(settings));

  // inference settings

  OPTS->var_conv = 1e-4;
  OPTS->var_maxiter = -1;  // -1 means "run until convergence"
  OPTS->var_miniter = 0;
  OPTS->phi_only = FALSE;
  OPTS->max_doc_change = -1;

  // fitting settings

  OPTS->em_conv = 1e-4;
  OPTS->permute_words = FALSE;
  OPTS->em_max_iter = 100;
  OPTS->em_min_iter = 10;

  OPTS->trim_prop = 0.00;
  OPTS->init_alpha_scale = 1.0;
  // OPTS->coef_init_type = COEF_INIT_ZERO;
  OPTS->coef_init_type = COEF_INIT_GRIDDED;
  OPTS->min_coef = -1.0;
  // OPTS->topic_init_type = TOPIC_INIT_FROM_UNIFORM;
  OPTS->topic_init_type = TOPIC_INIT_FROM_RANDOM;
  OPTS->init_num_docs = 1;
  OPTS->init_smooth = 10;
  OPTS->init_sigma2 = 0.0;
  OPTS->adapt_var_maxiter = FALSE; // do we do (ad-hoc) short circuiting?
  OPTS->fit_sigma2 = TRUE;
  OPTS->backtrack_sigma2 = FALSE;

  // settings with no defaults

  OPTS->type = -1;
  OPTS->ntopics = -1;
  OPTS->docs_file = NULL;
  OPTS->vocab_file = NULL;
  OPTS->resp_file = NULL;
  OPTS->model_dir = NULL;
  OPTS->inf_dir = NULL;
  OPTS->task = -1;


}


/* print the current settings */

void print_settings()
{
  assert(OPTS != NULL);

  outlog("");
  outlog("**************************************************");
  outlog("variational inference settings");
  outlog("");
  outlog("%20s = %g", "convergence", OPTS->var_conv);
  outlog("%20s = %d", "max iter", OPTS->var_maxiter);
  outlog("%20s = %d", "min iter", OPTS->var_miniter);
  outlog("%20s = %d", "phi only", OPTS->phi_only);
  outlog("%20s = %f", "max doc change", OPTS->max_doc_change);

  outlog("");
  outlog("**************************************************");
  outlog("modeling settings");
  outlog("");
  outlog("%20s = %d", "type", OPTS->type);
  outlog("%20s = %d", "num topics", OPTS->ntopics);
  outlog("%20s = %s", "docs", OPTS->docs_file);
  outlog("%20s = %s", "vocab", OPTS->vocab_file);
  outlog("%20s = %s", "resp", OPTS->resp_file==NULL?"null":OPTS->resp_file);
  outlog("%20s = %s", "model directory", OPTS->model_dir);
  outlog("%20s = %s", "inference directory", OPTS->inf_dir==NULL?"null":OPTS->inf_dir);
  outlog("%20s = %d", "task", OPTS->task);

  outlog("");
  outlog("**************************************************");
  outlog("EM settings");
  outlog("");
  outlog("%20s = %f", "convergence", OPTS->em_conv);
  outlog("%20s = %d", "permute words", OPTS->permute_words);
  outlog("%20s = %d", "EM max iter", OPTS->em_max_iter);
  outlog("%20s = %d", "EM min iter", OPTS->em_min_iter);
  outlog("%20s = %d", "use vb", OPTS->use_var_bayes);
  if (OPTS->use_var_bayes == 1)
    outlog("%20s = %f", "vb smooth", OPTS->vb_smooth);
  if (OPTS->trim_prop > 0)
    outlog("%20s = %f", "trim prop", OPTS->trim_prop);
  outlog("%20s = %d", "adapt var max iter", OPTS->adapt_var_maxiter);
  outlog("%20s = %f", "init alpha scale", OPTS->init_alpha_scale);
  outlog("%20s = %d", "coef init type", OPTS->coef_init_type);
  outlog("%20s = %g", "min coef", OPTS->min_coef);
  outlog("%20s = %d", "topic init type", OPTS->topic_init_type);
  outlog("%20s = %d", "init num docs", OPTS->init_num_docs);
  outlog("%20s = %f", "init smooth", OPTS->init_smooth);
  outlog("%20s = %f", "init sigma2", OPTS->init_sigma2);
  outlog("%20s = %d", "fit sigma2", OPTS->fit_sigma2);
  outlog("%20s = %d", "backtrack sigma2", OPTS->backtrack_sigma2);
}

// !!! above, decode the coded things like task and model type
// !!! only print relevant things, e.g., EM when we are doing fitting

/* short circuiting and EM

   max_doc_change is an upper bound on the percent increase in the
   elbo from the previous em iteration (for a document) to the current
   one.  when it is equal to -1, this "feature" is not used.  it is
   never used in the current code.

   var_max_iter is the alternative way to perform short circuiting.
   when adapt_var_max_iter is true then the number of variational
   iterations begins at 2 and goes up from there.  the specific
   schedule is {2,4,8,16,32,42,52,62,...}.  this happens when the ELBO
   decreases.

   !!! should we backtrack in these settings?  unclear.  this whole
   set up feels like something of a patched together hack.

   continuing with the hack: when the EM convergence criterion is met,
   we run "final EM iterations" where the maximum number of
   variational iterations is set to infinity (-1, in the code).  we
   then hardcode the EM convergence criterion to 1e-4.

   !!! oy vey.
*/
