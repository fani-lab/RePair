# Preprocessed

Creates pairing files for datasets using the raw folder. 

we create 2 different files.

 - `pairing1.qrels.pairing2.ctx.tsv` : this file stores qid, context if any and both the provided pairings for retrieval rankers to assert qids with their respective query from the prediction file.
 - `pairing1.pairing2.tsv` : this file is the one with raw pairings in a tab seperated format to be passed on to the transformer model for **finetuning** and **prediction**
