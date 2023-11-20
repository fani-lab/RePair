from tqdm import tqdm
import pandas as pd

class Query:
    """
    Query Class

    Represents a query with associated attributes and features.

    Attributes:
        qid (int): The query identifier.
        q (str): The query text.
        docs (dict): A dictionary of tuples containing document information.
            Each tuple includes docid and relevancy, and additional information
            related to documents can be added in between.
        q_ (dict): A dictionary containing the refiner's name as the key and the tuple of (refined query, semantic similarity) as the value. Notice that type of the refined query must be Query.
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
    def __init__(self, domain, qid, q, docs, args=None):
        self.domain = domain
        self.qid = qid
        self.q = q
        self.docs = docs
        self.q_ = dict()
        self.qret= dict()
        self.original = False
        self.lang = 'English'

