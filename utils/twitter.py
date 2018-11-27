from __future__ import absolute_import

import gzip
import logging
import os

import requests

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_data(in_dst_file):
    dataset_urls = ['https://github.com/marsan-ma/chat_corpus/blob/master/twitter_en_big.txt.gz.partaa?raw=true',
                    'https://github.com/marsan-ma/chat_corpus/blob/master/twitter_en_big.txt.gz.partab?raw=true']

    TMP_FILENAME = '.tmp_twitter.gz'
    with open(TMP_FILENAME, 'wb') as f_out:
        for dataset_url in dataset_urls:
            logger.info('Downloading file at the url: {}'.format(dataset_url))
            req = requests.get(dataset_url, stream=True)
            assert req.status_code == 200, 'Error during downloading {}'.format(dataset_url)
            # req.raw.decode_content = True
            f_out.write(req.raw.read())
    with gzip.open(TMP_FILENAME, 'rb') as f_in, open(in_dst_file, 'wb') as f_out:
        file_content = f_in.read()
        f_out.write(file_content)
    os.remove(TMP_FILENAME)
