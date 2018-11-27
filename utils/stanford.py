from __future__ import absolute_import

import json
import zipfile
import logging
from argparse import ArgumentParser
from io import BytesIO

import requests

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_data(in_dst_filename):
    dataset_url = 'http://nlp.stanford.edu/projects/kvret/kvret_dataset_public.zip'
    req = requests.get(dataset_url)

    result = []
    with zipfile.ZipFile(BytesIO(req.content)) as zip_in:
        for filename in zip_in.namelist():
            if 'public.json' in filename:
                with zip_in.open(filename) as json_in:
                    json_content = json.load(json_in)
                    for dialogue in json_content:
                        if not len(dialogue['dialogue']):
                            continue
                        result.append(dialogue['dialogue'][0]['data']['utterance'])
    print('Got {} utterances'.format(len(result)))
    with open(in_dst_filename, 'w', encoding='utf-8') as result_out:
        for utterance in result:
            print(utterance.lower(), file=result_out)


def configure_argument_parser():
    result_parser = ArgumentParser('Download Stanford Key-Value Network Dialogue dataset (all the user utterances)')
    result_parser.add_argument('result_file')
    return result_parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    args = parser.parse_args()
    download_data(args.result_file)
