# OUTPUTS
The final structure of the output will look like the below:

```bash
├── output
│   └── dataset_name                                 [such as robust04]
│       └── refined_queries_files                    [such as refiner.bt_bing_persian]
│       └── rag
│       │   └── rag_predictions                      [such as pred.base.local.0]
│       │   └── rag_ranker_files                     [such as pred.base.local.0.bm25]
│       │   └── rag_metric_files                     [such as pred.base.local.0.bm25.map]
│       └── ranker.metric                            [such as bm25.map]
│           └── ranker_files                         [such as refiner.bt_bing_persian.bm25]
│           └── metric_files                         [such as refiner.bt_bing_persian.bm25.map]
│           └── rag               
│           │    └── fusion                          [such as bm25.map.agg.+bt.all.tsv]
│           │        └── multi    
│           │        │   └── multi_k_ranker_files    [such as rag.bt.k0.bm25]
│           │        │   └── multi_k_metric_files    [such as rag.bt.k0.bm25.map]
│           │        └── rag_fusion_files            [bm25.map.agg.all.tsv]
│           └── agg                                  [such as refiner.bt_bing_persian.bm25]
│                └── agg_files                       [such as bm25.map.agg.+bt.all.tsv]
─
```

Every Dataset is stored under a common configuration for every ranker and metric that is computed upon it.
