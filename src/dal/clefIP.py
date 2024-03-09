import json
import os 
import pathlib
import xml.etree.ElementTree as ET

from dal.ds import Dataset

from pyserini.search.lucene import LuceneSearcher
from shutil import copyfile

class ClefIP(Dataset):
    def __init__(self, settings, homedir, ncore): 
      try: super(ClefIP, self).__init__(settings=settings)
      except: self._build_index(homedir, Dataset.settings, ncore)

    @classmethod
    def _build_index(cls, homedir, settings, ncore):
        indexdir = settings['index']
        index_item = settings['index_item']
        ##qrels_cols = settings['qrels_cols']
        ##queries_cols = settings['queries']
        source_path = settings['source_path']

        print("Creating index using local files for ClefIP...")
        os.environ['IR_DATASETS_HOME'] = '/'.join(homedir.split('/')[:-1])
        if not os.path.isdir(os.environ['IR_DATASETS_HOME']): os.makedirs(os.environ['IR_DATASETS_HOME'])
        index_item_str = '.'.join(index_item)
        if not os.path.isdir(f'{indexdir}/{cls.user_pairing}{index_item_str}'): os.makedirs(f'{indexdir}/{cls.user_pairing}{index_item_str}')
        
        from cmn import lucenex
        print(f"Setting up ClefIP corpus at {homedir}...")

        ##qrels = cls.load_qrels()
        ##queries = cls.load_queries()

        print("Creating jsonl collections for indexing...")

        cls.create_jsonl(source_path, index_item, f'{homedir}/{index_item_str}')

        if len(os.listdir(f'{indexdir}/{cls.user_pairing}{index_item_str}')) == 0:
            lucenex.lucenex(f'{homedir}/{cls.user_pairing}{index_item_str}', f'{indexdir}/{cls.user_pairing}{index_item_str}/', ncore)
        
        if os.path.isfile(f'{homedir}/qrels'): copyfile(f'{homedir}/qrels', f'{homedir}/qrels.test.tsv')
        if os.path.isfile(f'{homedir}/queries.tsv'): copyfile(f'{homedir}/queries.tsv', f'{homedir}/queries.test.tsv')
        cls.searcher = LuceneSearcher(f'{indexdir}/{index_item_str}')
        return

    @classmethod
    def create_jsonl(cls, source_path, index_item, output):
        if not os.path.isdir(output): os.makedirs(output)
        if not os.path.isfile(f'{output}/docs.jsonl'):
            print(f'Converting ClefIP docs into jsonl collection for {index_item}')
            output_jsonl_file = open(f'{output}/docs.jsonl', 'w', encoding='utf-8', newline='\n')
            ## first, iterate through all paths in the directory python
            for filename in os.listdir(source_path):
                file_path = pathlib.Path(os.path.join(source_path, filename))

                ## some files downloaded have duplicate root nodes. 
                ## We need the file to have only one root node, hence the wrapper.  Then, we only keep the first instance of the patent-document node, as the others are duplicates.    
                data = b'<wrapper>' + file_path.read_bytes() + b'</wrapper>'

                tree = ET.fromstring(data)
                root = tree.find("patent-document")
                did = root.get("ucid")
                date = root.get("date")

                if not root.get("lang") == "EN":
                   continue

                # abstract (list of docs), date

                title = None
                listOfAbstracts = []

                for child in root.find("bibliographic-data").find("technical-data").findall("invention-title"):
                    if child.get("lang") == "EN":
                        title = child.text
                
                for child in root.findall("abstract"):
                    if not child.get("lang") == "EN":
                        continue
                    listOfParagraphs = []
                    for grandchild in child.findall("p"):
                        listOfParagraphs.append(grandchild.text)
                    listOfAbstracts.append(listOfParagraphs)

                doc = {'title': title, 'date': date, 'listOfAbstracts': listOfAbstracts}

                filtered_contents = ' '.join(doc[key] for key in index_item)

                json_string = json.dumps({"id": did, "contents": filtered_contents}, indent=4)
                output_jsonl_file.write(json_string + '\n')