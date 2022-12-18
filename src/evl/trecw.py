import os

def evaluate(in_docids, out_metrics, qrels, metric, lib='pytrec'):#or 'pytrec'
    if lib == 'pytrec':
        raise NotImplementedError
    else:
        cli_cmd = f'{lib} -q -m {metric} {qrels} {in_docids} > {out_metrics}'
        print(cli_cmd)
        stream = os.popen(cli_cmd)
        print(stream.read())


