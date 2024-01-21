import json
import argparse
import csv
from random import sample
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_rate', type=float, default=0.8, choices=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        help='sampling rate')
    args = parser.parse_args()
    return args


def sampledata(file_name, rate):
    file_name = file_name + '.json'
    with open(file_name) as f:
        input = json.load(f)
        data_list = {}
        for i in input:
            bias_type = i['bias_type']
            if data_list.get(bias_type) is None:
                data_list[bias_type] = []
            data_list[bias_type].append(i)
        full_sample_list = []
        for d in data_list:
            d_list = data_list[d]
            d_list_len = len(d_list)
            sample_num = d_list_len * rate
            list_by_type = sample(d_list, math.ceil(sample_num))
            full_sample_list.extend(list_by_type)
        return full_sample_list


def main(args):
    file_paths = ['../data/paralled_cp', '../data/paralled_ss']
    for file_path in file_paths:
        data = sampledata(file_path, args.sample_rate)
        with open(file_path + '_' + str(args.sample_rate) + '.json', 'w') as fw:
            json.dump(data, fw, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main(args)
