typedef struct settings
{
    int type;                      // model type
    int ntopics;                   // number of topics
    int nrestarts;                 // number of times to restart em
    double em_conv;                // em convergence criterion
    int permute_words;             // permute words at e step
    int em_max_iter;               // max em iterations
    int em_min_iter;               // min em iterations
    int use_var_bayes;             // employ variational bayes
    double vb_smooth;              // variational bayes smoothing
    double trim_prop;              // proportion of response to trim

    short init_type;               // initialization type
    char* init_topics;             // initial topics
    int init_num_docs;             // number of docs to initialize with
    double init_alpha_scale;       // initial sum alpha_i
    double init_smooth;            // initial topics smoothing
    double init_coeff;             // initial coefficient

    double var_conv;               // variational convergence criterion
    int var_maxiter;               // max number of variational iter
    int var_miniter;               // min number of variational iter
    int phi_only;                  // use "phi only" inference
    int collapsed;                 // use collapsed variational inference

    int verbose;                   // verbose flag
    char* logfile;                 // log file
    // !!! outlog stuff here
} settings;
