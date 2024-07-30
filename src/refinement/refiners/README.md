# Refiners

The objective of query refinement is to produce a set of potential candidate queries that can function as enhanced and improved versions. This involves systematically applying various unsupervised query refinement techniques to each query within the input dataset.

<table align="center" border=0>
<thead>
  <tr><td colspan="3" style="background-color: white;"><img src="../../../misc/classdiagram.png", width="1000", alt="ReQue: Class Diagram"></td></tr>     
  <tr><td colspan="3">
      <p align="center">Class Diagram for Query Refiners in <a href="./src/refinement/">src/refinement/</a>. [<a href="https://app.lucidchart.com/documents/view/64fedbb0-b385-4696-9adc-b89bc06e84ba/HWEp-vi-RSFO">zoom in!</a>].</p>
      <p align="center"> The expanders are initialized by the Expander Factory in <a href="./src/refinement/refiner_factory.py">src/refinement/refiner_factory.py</a></p></td></tr> 
 </thead>
</table>

This README provides comprehensive information about the refiners used in the RePair project, categorized into _global_ and _local_ unsupervised refinement methods. 
These methods are crucial for generating gold-standard datasets for training supervised or semi-supervised query refinement techniques. 
[Global](#Global) methods operate solely on the original query, refining it without external context. 
In contrast, [Local](#Local) refiners take into account terms from the top-k retrieved documents obtained through an initial information retrieval process, such as _bm25_ or _qld_. 
The local approaches allow for the addition of similar or related terms to the original query, thereby enhancing the relevance and accuracy of the refined queries.

Here is the list of refiners:
| **Expander** 	| **Category** 	| **Analyze type** 	|
|---	|:---:	|:---:	|
| [backtranslation](#backtranslation) 	| Machine_Translation 	| Global 	|
| [tagmee](#tagmee) 	| Wikipedia 	| Global 	|
| [stemmers](#stemmers) 	| Stemming_Analysis 	| Global 	|
| [semantic](#semantic-refiners) 	| Semantic_Analysis 	| Global 	|
| [sensedisambiguation](#sense-disambiguation) 	| Semantic_Analysis 	| Global 	|
| [embedding-based](#embedding-based-methods) 	| Semantic_Analysis 	| Global 	|
| [anchor](#anchor) 	| Anchor_Text 	| Global 	|
| [wiki](#wiki) 	| Wikipedia 	| Global 	|
| [relevance-feedback](#relevance-feedback) 	| Top_Documents 	| Local 	|
| [bertqe](#bertqe) 	| Top_Documents 	| Local 	|

# Global

## backtranslation
Back translation, also known as reverse translation or dual translation, involves translating content, whether it is a query or paragraph, from one language to another and retranslating it to the original language. This method provides several options for the owner to make a decision that makes the most sense based on the task at hand.
For additional details, please refer to this [document](./misc/Backtranslation.pdf).

## Example
| **q** 	| **map q** 	| **language** 	| **translated q** 	| **backtranslated q** 	| **map q'** 	|
|---	|:---:	|:---:	|:---:	|:---:	|:---:	|
| Italian nobel prize winners 	| 0.2282 	| farsi 	| برندهای جایزه نوبل ایتالیایی 	| Italian Nobel laureates 	| 0.5665 	|
| banana paper making 	| 0.1111 	| korean 	| 바나나 종이 제조 	| Manufacture of banana paper 	| 1 	|
| figs 	| 0.0419 	| tamil 	|  அத்திமரங்கள்  	| The fig trees 	| 0.0709 	|

## tagme
This method replaces the original query's terms with the title of their Wikipedia articles.

## stemmers
It utilizes various lexical, syntactic, and semantic aspects of query terms and their relationships to reduce them to their roots, employing methods such as Krovetz, Lovins, PaiceHusk, Porter, Sremoval, Trunc4, and Trunc5.

## semantic refiners
It utilizes an external linguistic knowledge base, including thesaurus, WordNet, and ConceptNet, to extract related terms for the original query's terms.

## sense-disambiguation
It resolves the ambiguity of polysemous terms in the original query by analyzing the surrounding terms and then incorporates synonyms of the query terms as related terms.

## embedding-based methods
It utilizes pre-trained term embeddings from GloVe and Word2Vec to identify the most similar terms to the query terms.

## anchor
It uses pre-trained term embeddings from GloVe and Word2Vec to identify the most similar terms to the query terms.

## wiki
It uses embeddings trained on Wikipedia's hierarchical categories to add the most similar concepts to each query term.

# Local

## relevance-feedback
In this method, important terms from the top-k retrieved documents are incorporated into the original query based on metrics such as tf-idf. Additionally, clustering techniques like TermCluster, DocCluster, and ConceptCluster are used, employing graph clustering methods like Louvain on a graph where nodes represent terms and edges denote pairwise co-occurrence counts. This approach ensures that each cluster contains terms that frequently co-occur. To refine the original query, related terms are selected from the clusters to which the initial query terms belong.
    
## bertqe
which employs bert's contextualized word embeddings of terms in the top-k retrieved documents.
