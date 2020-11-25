"""
Abstracting the I/O workflow

Inputs
- data_entries_in: list of dicts {Di} that can hold attributes of any form {Dij}

Outputs
- data_entries_out: list of dicts {Di} with additional attributes {Dik} than data_entries_in

Useful in the loop, which could also be byproducts
- "functions" that provide these additional attributes Dij -> Dik
 - semantic functions
 - localized geometric functions
 - neural-net functions
 - [so on and so forth]

----

Hover's main functionality: Dij -> Dik in a visual interface
- Bokeh server hosting a 2D plot with widgets and callbacks
 - lasso select
 - pointwise select
 - text box for applying an attribute key
 - text box or multiple-choice for applying an attribute value
 - [for text data] keyword/regex select
 - [yada yada]

- Two ways to invoke Bokeh:
 - Streamlit for high-level familiarization
 - Jupyter for hands-on prototyping
----

"You only pay for what you use... but you must use something"
Each use case deserves its own folder and loadable module hosting Python-based configs.
Think about what should be in that __init__ file:
[*] means mandatory on its level, [+] means optional on its level, [^] means built-in support should be in place

- [*] Data I/O from/to disk
 - [* Template/Example] a "DataEntry" class that regulates each piece of dict
  - [*] "load/save" class methods should handle the load/preprocessing/postprocessing/dump process, creating backup snapshots
  - [*] a "render" instance method that customizes the hover tooltip in Bokeh
  - [*] a "vectorize" static method with caching that returns a high-dimensional vector

- [+] dimensionality reduction specification
 - [*] a constant/gin that is configurable, default-able, and interactable (i.e. sets of parameters worth trying, maybe with a slider)

- [+] semantic functions
 - [^] Snorkel-based inspection and fitting
 - [+] genetic-algorithm-like optimization

- [+] localized geometric functions
 - [*] turning lasso-select into a readable/exportable function -- read human-learn's code for reference

- [+] neural-net-in-the-loop
 - [*] a "model_architecture" function that returns a Torch.nn subclass whose input dimensions match the DataEntry.vectorize() output dimensions
 - [*] how to train this model?
 - [+] active learning
  - [*] how to select sample(s)?
   - [^] selected sample(s) should be linked to Bokeh for immediate annotation
"""
