# RePair

The RePair project is an open-source initiative designed to generate datasets containing pairs of original and refined queries. This pipeline involves multiple steps: it first takes an initial query, refines it to better capture the user's intent, retrieves relevant documents based on the refined query, evaluates the results, and then generates the pairs of original and refined queries. This comprehensive process helps in creating high-quality datasets that can be used to train and evaluate query refinement models more effectively.

## Table of Contents
<table align="center" border=0>
<tr>
<td>
   
1. [Introduction](#1-Introduction)
2. [Repair 1.0](#2-Repair-1.0)
3. [Repair 2.0](#3-Repair-2.0)
4. [Repair 2.1](#4-Repair-2.1)
5. [Acknowledgement](#5-Acknowledgement)
</td>
<td><img src='misc/repair_flow.jpg' width="100%" /></td>
</table>

   
## 1. Introduction
Web users often struggle to express their information needs clearly in short, vague queries, making it challenging for search engines to retrieve relevant results. The precision of search queries is essential, as it directly affects the quality and relevance of the search outcomes. Query refinement, which aims to improve search relevance by adjusting original queries, plays a crucial role in addressing this challenge. Effective query refinement can bridge the gap between user intent and search engine interpretation, leading to more accurate and satisfactory search results. However, current evaluation methods for query refinement models may not accurately reflect real-world usage patterns, highlighting the need for more robust evaluation frameworks.

In this GitHub repository, we present different versions of the RePair project, accessible through various branches. The main branch contains the most updated version, ensuring users have access to the latest improvements and features of the RePair project.

## 2. RePair 1.0

## 3. RePair 2.0

## 4. RePair 2.1

## 4. Acknowledgement
We benefit from [``trec_eval``](https://github.com/usnistgov/trec_eval), [``pyserini``](https://github.com/castorini/pyserini), [``ir-dataset``](https://ir-datasets.com/), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  



