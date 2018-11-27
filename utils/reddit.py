from __future__ import absolute_import

import bz2
import logging
from io import BytesIO

import requests

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_data(in_dst_file):
    dataset_url = 'https://files.pushshift.io/reddit/comments/RC_2015-05.bz2'
    response = requests.get(dataset_url)

    with open(in_dst_file, 'wb') as new_file, bz2.BZ2File(BytesIO(response.content), 'rb') as file:
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(data)