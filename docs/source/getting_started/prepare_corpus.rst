Preprocessing
=============

TMNT provides a preprocessor that can be used to *vectorize* a text dataset. The pre-processor
wraps `sklearn.feature_extraction.text.CountVectorizer`, providing some convenient
extensions to support corpora stored in `.jsonl` format.  The pre-processor can
be used directly via the API or using the `bin/prepare_corpus.py` script.

Preprocessing via the Command-line
++++++++++++++++++++++++++++++++++

The `bin/prepare_corpus.py` script handles pre-processing from the command-line and
includes the following options::

    --tr_input TR_INPUT   Directory of training files (jsonl batches) or SINGLE
                        file
    --val_input VAL_INPUT
                        Directory of validation test files (jsonl batches) or
                        SINGLE file
    --tst_input TST_INPUT
                        Directory of held out test files (jsonl batches) or
                        SINGLE file
    --file_pat FILE_PAT   File pattern to select files as input data (default '*.jsonl')
    --tr_vec_file TR_VEC_FILE
                        Output file with training documents in sparse vector
                        format
    --val_vec_file VAL_VEC_FILE
                        Output file with test validation documents in sparse
                        vector format
    --tst_vec_file TST_VEC_FILE
                        Output file with heldout test documents in sparse
                        vector format
    --vocab_size VOCAB_SIZE
                        Size of the vocabulary to construct
    --vocab_file VOCAB_FILE
                        File for resulting vocabulary
    --full_vocab_histogram FULL_VOCAB_HISTOGRAM
                        Optional output of entire histogram
    --json_text_key JSON_TEXT_KEY
                        Key for json field containing document text (default
                        is 'text')
    --json_label_key JSON_LABEL_KEY
                        Key for json field containing label (default is None).
                        Only set if labels always available
    --label_map LABEL_MAP
                        JSON object to file with mapping between labels and
                        indices
    --json_out_dir JSON_OUT_DIR
                        Create a new JSON list file with vectors added as a
                        field in this target directory
    --min_doc_length MIN_DOC_LENGTH
                        Minimum document length (in tokens)
    --custom_stop_words CUSTOM_STOP_WORDS
                        Custom stop-word file (one word per line)
    --label_prefix_chars LABEL_PREFIX_CHARS
                        Use first N characters of label
    --str_encoding STR_ENCODING
                        String/file encoding to use
    --log_dir LOG_DIR     Logging directory


Preparing a dataset with meta-data
++++++++++++++++++++++++++++++++++

When processing json data where one of the fields corresponds to meta-data to treat as labels or co-variates,
the `--json_label_key` should be used to identify this field.  The values of this field are assumed to be categorical
values of the dependent (random) variable.  The `prepare_corpus.py` script will
automatically create an index that maps the string values of this field to integers. The result of this mapping
is output to the file argument of `--label_map`. This label map provides a means to associate the integer-valued
labels in the sparse-vector output back to their associated string representations.



 




