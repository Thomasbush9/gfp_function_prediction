## Ideas to solve issue with current implementation Boltz:

The main reason is to make the overall process more modular so that if one step breaks, we don't have to redo everything. 

Step 1: data management 

- create output dir, create paths used for next steps, create logging dirs -> folder wise 
- msa calculation: compute the msa for each sequence and add the path to the sequence. 
- boltz: load the model and the cache, start processing the files. 
