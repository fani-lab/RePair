class Query:

    def __init__(self, qid, q, args=None):
        self.qid = qid
        self.q = q
        self.docs_rel = []

        # Features
        if args['id']: self.user_id = args['id']
        if args['time']: self.time = args['time']
        if args['location']: self.time = args['location']

    def add_document(self, docid, rel): self.docs.append((docid, rel))