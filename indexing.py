import os


def run_command(command):
    try:
        result = os.popen(command).read()
        print("Command Output:")
        print(result)
    except Exception as e:
        print("Error Occurred:")
        print(e)


if __name__ == '__main__':
    # Example usage
    command_to_run = "python -m pyserini.encode " \
                     "input --corpus ./dbpedia_corpus.jsonl " \
                     "--fields text " \
                     "--delimiter \"\\n\" " \
                     "--shard-id 0 " \
                     "--shard-num 1 " \
                     "output --embeddings data/corpus/dbpedia/dbpedia_encoder.title " \
                     "--to-faiss " \
                     "encoder --encoder castorini/tct_colbert-v2-hnp-msmarco " \
                     "--fields text " \
                     "--batch 32"

    run_command(command_to_run)
