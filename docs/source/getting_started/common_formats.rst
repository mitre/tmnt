Common Formats
==============

Input Documents
+++++++++++++++

TMNT aims for flexibility to handle data and documents in any one of a number of data formats.
However, many of the examples and tutorials use a "JSON list" format consisting of one
or more files with each file containing a separate serialized JSON object on each line. Each json
object is assumed to have a single field that contains the text of each document.  Additional
meta-data, if available, is typically contained in other (flat) fields.  Of particular note,
a *label* field is required for supervised or semi-supervised models. Below
is an example::
  
  {"id": "3664", "text": "This is the text of a document about science.", "label": "science"}
  {"id": "3665", "text": "This is the text of a document about politics.", "label": "politics"}
  ...

Configuration Files
+++++++++++++++++++

Models and their associated hyperparameters are represented through a simple JSON file that contains
various model attributes and sub-attributes. Configuration files are described in more detail
here: :ref:`config-options-label`

Model Space Files
+++++++++++++++++

Model selection is a first-class feature of TMNT. The space of possible models to consider
during model selection is specified using a YAML file format with various conventions described
in detail here: :ref:`model-selection-label`
