from refinement.refiners.abstractqrefiner import AbstractQRefiner
from refinement.refiner_param import bing


class Bing(AbstractQRefiner):
    def __init__(self):
        AbstractQRefiner.__init__(self)


    '''
    Generates the queries using OpenAI's ChatGPT
    '''
    def get_refined_query(self, query, args=None):
        pass

    def get_relevant_doc(self, query):
        web_data = self.client.web.search(query=query)
        '''
        Web pages
        If the search response contains web pages, the first result's name and url
        are printed.
        '''
        if hasattr(web_data.web_pages, 'value'):
            print("\r\nWebpage Results#{}".format(len(web_data.web_pages.value)))
            first_web_page = web_data.web_pages.value[0]
            print("First web page name: {} ".format(first_web_page.name))
            print("First web page URL: {} ".format(first_web_page.url))
            return first_web_page.name
        else:
            print("Didn't find any web pages...")
            return None

    '''
    Returns the name of the model ('backtranslation) with name of the target language
    Example: 'backtranslation_fra_latn'
    '''
    def get_model_name(self):
        return 'chatgpt'


if __name__ == "__main__":
    qe = Bing()
    print(qe.get_model_name())
    print(qe.get_refined_query('This is my pc'))
