# if 'refiner' in settings['cmd']:
#     # search step
#     from evl import trecw
#     print(f'using t5 as a refiner for a sample collection of msmarco')
#     refiner_output = f'../output/t5-refinement'
#     query_originals = pd.read_csv(f'{prep_output}/queries.dev.small.tsv', sep='\t', names=['qid', 'query'],
#                                         dtype={'qid': str})
#     query_changes = [(f'{refiner_output}/aol.title.pred.msmarco-1004000', f'{refiner_output}/aol.title.pred.msmarco-1004000.{settings["ranker"]}'),
#                      (f'{refiner_output}/aol.title.url.pred.msmarco-1004000', f'{refiner_output}/aol.title.url.pred.msmarco-1004000.{settings["ranker"]}'),
#                      (f'{refiner_output}/msmarco.pred-1004000', f'{refiner_output}/msmarco.pred-1004000.{settings["ranker"]}'),
#                      (f'{refiner_output}/msmarco.paraphrase.pred-1004000',f'{refiner_output}/msmarco.paraphrase.pred-1004000.{settings["ranker"]}')]
#     # for (i, o) in query_changes: MsMarcoPsg.to_search(i, o, query_originals['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])
#     #
#     # # originals search
#     # MsMarcoPsg.to_search_df(pd.DataFrame(query_originals['query']),
#     #                  f'{refiner_output}/msmarco.dev.small.{settings["ranker"]}',
#     #                  query_originals['qid'].values.tolist(), settings['ranker'],
#     #                  topk=settings['topk'], batch=settings['batch'])
#
#     # eval step
#     search_results = [(f'{refiner_output}/aol.title.pred.msmarco-1004000.{settings["ranker"]}',
#                        f'{refiner_output}/aol.title.pred.msmarco-1004000.{settings["ranker"]}."recip_rank.10"'),
#                       (f'{refiner_output}/aol.title.url.pred.msmarco-1004000.{settings["ranker"]}',
#                        f'{refiner_output}/aol.title.url.pred.msmarco-1004000.{settings["ranker"]}."recip_rank.10"'),
#                       (f'{refiner_output}/msmarco.pred-1004000.{settings["ranker"]}',
#                        f'{refiner_output}/msmarco.pred-1004000.{settings["ranker"]}."recip_rank.10"'),
#                       (f'{refiner_output}/msmarco.paraphrase.pred-1004000.{settings["ranker"]}',
#                        f'{refiner_output}/msmarco.paraphrase.pred-1004000.{settings["ranker"]}."recip_rank.10"'),
#                       (f'{refiner_output}/msmarco.dev.small.{settings["ranker"]}',
#                        f'{refiner_output}/msmarco.dev.small.{settings["ranker"]}."recip_rank.10"')
#                       ]
#     with multiprocessing.Pool(settings['ncore']) as p:
#         p.starmap(
#             partial(trecw.evaluate, qrels=f'{datapath}/qrels.dev.small.tsv', metric="recip_rank.10",
#                     lib=settings['treclib']), search_results)
if 'ds_split' in settings["cmd"]: refiner.train_test_split(box_path)

if 'refiner' in settings['cmd']:
    # search step
    from evl import trecw

    print(f'using t5 as a refiner for a sample collection of aol')
    refiner_output = f'../output/t5-refinement'
    query_originals_title = pd.read_csv(f'{prep_output}/aol.dev.title.tsv', sep='\t', names=['qid', 'query'],
                                        dtype={'qid': str})
    query_originals_title_url = pd.read_csv(f'{prep_output}/aol.dev.title.url.tsv', sep='\t', names=['qid', 'query'],
                                            dtype={'qid': str})
    query_changes_title = [
        (f'{refiner_output}/aol.title.pred-1004000', f'{refiner_output}/aol.title.pred-1004000.{settings["ranker"]}'),
        (f'{refiner_output}/msmarco.pred.aol.title-1004000',
         f'{refiner_output}/msmarco.pred.aol.title-1004000.{settings["ranker"]}')]
    query_changes_title_url = [(f'{refiner_output}/aol.title.url.pred-1004000',
                                f'{refiner_output}/aol.title.url.pred-1004000.{settings["ranker"]}'),
                               (f'{refiner_output}/msmarco.pred.aol.title.url-1004000',
                                f'{refiner_output}/msmarco.pred.aol.title.url-1004000.{settings["ranker"]}')]
    # for (i, o) in query_changes_title: Aol.to_search(i, o, query_originals_title['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])
    # for (i, o) in query_changes_title_url: Aol.to_search(i, o, query_originals_title_url['qid'].values.tolist(), settings['ranker'], topk=settings['topk'], batch=settings['batch'])
    # with multiprocessing.Pool(settings['ncore']) as p:
    #
    #     p.starmap(partial(Aol.to_search, qids=query_originals_title['qid'].values.tolist(), index_item='title', ranker=settings['ranker'], topk=settings['topk'], batch=settings['batch']), query_changes_title)
    #     p.starmap(partial(Aol.to_search, qids=query_originals_title_url['qid'].values.tolist(), index_item=None, ranker=settings['ranker'], topk=settings['topk'], batch=settings['batch']), query_changes_title_url)

    # originals search
    # Aol.to_search_df(pd.DataFrame(query_originals_title['query']), f'{refiner_output}/aol.dev.title.{settings["ranker"]}', query_originals_title['qid'].values.tolist(), index_item_str, settings['ranker'], topk=settings['topk'], batch=settings['batch'])
    # Aol.to_search_df(pd.DataFrame(query_originals_title_url['query']), f'{refiner_output}/aol.dev.title.url.{settings["ranker"]}', query_originals_title_url['qid'].values.tolist(), index_item_str, settings['ranker'], topk=settings['topk'], batch=settings['batch'])

    # eval step
    search_results = [(f'{refiner_output}/aol.title.pred-1004000.{settings["ranker"]}',
                       f'{refiner_output}/aol.title.pred-1004000.{settings["ranker"]}."recip_rank.10"'),
                      (f'{refiner_output}/msmarco.pred.aol.title-1004000.{settings["ranker"]}',
                       f'{refiner_output}/msmarco.pred.aol.title-1004000.{settings["ranker"]}."recip_rank.10"'),
                      (f'{refiner_output}/aol.title.url.pred-1004000.{settings["ranker"]}',
                       f'{refiner_output}/aol.title.url.pred-1004000.{settings["ranker"]}."recip_rank.10"'),
                      (f'{refiner_output}/msmarco.pred.aol.title.url-1004000.{settings["ranker"]}',
                       f'{refiner_output}/msmarco.pred.aol.title.url-1004000.{settings["ranker"]}."recip_rank.10"'),
                      (f'{refiner_output}/aol.dev.title.{settings["ranker"]}',
                       f'{refiner_output}/aol.dev.title.{settings["ranker"]}."recip_rank.10"'),
                      (f'{refiner_output}/aol.dev.title.url.{settings["ranker"]}',
                       f'{refiner_output}/aol.dev.title.url.{settings["ranker"]}."recip_rank.10"')]
    with multiprocessing.Pool(settings['ncore']) as p: p.starmap(
        partial(trecw.evaluate, qrels=f'{datapath}/qrels.tsv_', metric="recip_rank.10",
                lib=settings['treclib']), search_results)

if 'ds_split' in settings["cmd"]:
    refiner.train_test_split(box_path)