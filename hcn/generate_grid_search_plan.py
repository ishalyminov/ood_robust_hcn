import argparse
import json

from itertools import product
from os import path, makedirs

from shutil import rmtree


def generate_plan(in_base_config, in_parameter_grid):
    result = []
    config_updates = [dict(zip(in_parameter_grid, x))
                      for x in product(*in_parameter_grid.values())]
    for config_update in config_updates:
        config = dict(in_base_config)
        config.update(config_update)
        result.append(config)
    return result


def main(in_base_config_file, in_parameter_grid_file, in_result_folder):
    with open(in_base_config_file) as base_config_in:
        base_config = json.load(base_config_in)
    with open(in_parameter_grid_file) as parameter_grid_in:
        parameter_grid = json.load(parameter_grid_in)

    configs = generate_plan(base_config, parameter_grid)
    if path.exists(in_result_folder):
        rmtree(in_result_folder)
    makedirs(in_result_folder)

    for index, config in enumerate(configs):
        with open(path.join(in_result_folder, 'config_{}.json'.format(index)), 'w') as config_out:
            json.dump(config, config_out)


def build_argument_parser():
    result = argparse.ArgumentParser()
    result.add_argument('base_config')
    result.add_argument('parameter_grid')
    result.add_argument('result_folder')
    return result


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()
    main(args.base_config, args.parameter_grid, args.result_folder)
