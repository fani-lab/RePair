# OUTPUTS
The final structure of the output will look like the below:

```bash
├── output
│   └── dataset_name              [such as robust04]
│   │   └── refined_queries_files [such as refiner.bt_bing_persian] 
│   │   └── ranker.metric         [such as bm25.map]
│   │   │   └── ranker_files      [such as refiner.bt_bing_persian.bm25]
│   │   │   └── metric_files      [such as refiner.bt_bing_persian.bm25.map]
│   │   │   └── agg               [such as refiner.bt_bing_persian.bm25]
│   │   │   │    └── agg_files    [such as bm25.map.agg.+bt.all.tsv]

```

The results are available in the [./output](./output) file.

You can also access all the results through this [link](https://uwin365.sharepoint.com/:f:/s/cshfrg-QueryRefinement/Elx37qFuAb5FoI4wapL3Bo4B6wmroVrKqC3W-8wpe8ACQw?e=C4P6IR).
