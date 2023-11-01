class Query:

    def __init__(self, dataset_name, q, qid, time=None, user_id=None):
        self.dataset_name = dataset_name
        self.qid = qid
        self.q = q

        # Features
        self.user_id = user_id
        self.time = time
