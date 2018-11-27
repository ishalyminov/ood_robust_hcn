from __future__ import absolute_import

import json
import logging
import requests

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_data(in_dst_file):
    dataset_urls = ['https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response/human_response_1.json',
                    'https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response/human_response_2.json',
                    'https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response/human_response_3.json',
                    'https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response/human_response_4.json',
                    'https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response/human_response_5.json',
                    'https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response_entity/response0.json',
                    'https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response_entity/response1.json',
                    'https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response_entity/response2.json',
                    'https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response_entity/response3.json',
                    'https://raw.githubusercontent.com/echoyuzhou/ticktock_text_api/master/human_response_entity/response4.json']

    result_json = []
    for dataset_url in dataset_urls:
        data = requests.get(dataset_url).json()
        result_json += data
    with open(in_dst_file, 'w') as json_out:
        json.dump(result_json, json_out)
