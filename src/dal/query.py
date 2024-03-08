from tqdm import tqdm
import pandas as pd

class Query:
    """
    Query Class

    Represents a query with associated attributes and features.

    Attributes:
        qid (int): The query identifier.
        q (str): The query text.
        docs (list): A list of tuples containing document information.
            Each tuple includes docid and relevancy, and additional information
            related to documents can be added in between.
        q_ (list): A list of tuples containing semantic similarity refined query, score and refiner's name.
        user_id (str, optional): The user identifier associated with the query.
        time (str, optional): The time of the query.
        location (str, optional): The location associated with the query.

    Args:
        qid (str): The query identifier.
        q (str): The query text.
        args (dict, optional): Additional features and attributes associated with the query,
            including 'id' for user identifier, 'time' for time information, and 'location'
            for location information.

    Example Usage:
        # Creating a Query object
        query = Query(qid='Q123', q='Sample query text', args={'id': 'U456', 'time': '2023-10-31'})

    """
    def __init__(self, domain, qid, q, args=None):
        self.domain = domain
        self.qid = qid
        self.q = q
        self.docs = dict()
        self.q_ = dict()

        # Features
        if args:
            if args['id']: self.user_id = args['id']
            if args['time']: self.time = args['time']
            if args['location']: self.time = args['location']
