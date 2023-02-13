class Dataset(object):
    searcher = None

    @staticmethod
    def to_txt(pid): pass

    @staticmethod
    def to_pair(input, output, index_item, cat=True): pass

    @staticmethod
    def to_search(in_query, out_docids, qids, index_item, ranker='bm25', topk=100, batch=None): pass



