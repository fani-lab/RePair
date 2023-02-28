# OUTPUTS
All our outputs for `aol-ia` and `msmarco.passage` can be obtained from [here](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EiVkfCxTjydKlpr3_VX-oo4B6o468rvnAQUq0VMkuUJL1Q?e=gGQvh4)
## Output Structure 
A brief sample of our output structure.  
```
- dataset
	-> model.config.platform.pairing-1.pairing-2.?variant 
		->model
		->prediction.n-model_training_checkpoint
		->prediction.n-model_training_checkpoint.ranker
		->prediction.n-model_training_checkpoint.ranker.metric
		->ranker.metric.agg.all_.tsv
		->ranker.metric.agg.all.tsv
		->ranker.metric.agg.gold.tsv
		->ranker.metric.boxes
			->{gold,platinum,diamond}.tsv
			->{gold,platinum,diamond}.qrels.tsv
			->stamps
				->{gold,platinum,diamond}.change.ranker
				->{gold,platinum,diamond}.change.ranker.metric
				->{gold,platinum,diamond}.original.ranker
				->{gold,platinum,diamond}.original.ranker.metric
				
			
```

Every Dataset is stored under a common configuration for every ranker and metric that is computed upon it.

An example would be: [msmarco.passage](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EiVkfCxTjydKlpr3_VX-oo4B6o468rvnAQUq0VMkuUJL1Q?e=gGQvh4)

- [`msmarco.passage ->t5.base.gc.docs.query`](https://uwin365-my.sharepoint.com/:f:/g/personal/lakshmiy_uwindsor_ca/EiVkfCxTjydKlpr3_VX-oo4B6o468rvnAQUq0VMkuUJL1Q?e=gGQvh4): describes we are using the `t5 base` as our model and config, `gc-> google cloud` as our platform, `docs.query` are the pairings.
- `t5.base.gc.docs.query -> bm25.map.agg.all.tsv` : this file contains every `bm25` ranker with a metric `map` for all predictions.
- `t5.base.gc.docs.query -> bm25.map.agg.gold.tsv`: this file contains all the best refined queries calculated by using `bm25` as ranker with a metric `map`


**some known issues:**
[T5 finetuning and prediction issue](https://github.com/fani-lab/RePair/issues/6):For some documents, T5 could not predict a query. This would inturn give us an assertion error 
[retrieval models throwing nan error](https://github.com/fani-lab/RePair/issues/8)
