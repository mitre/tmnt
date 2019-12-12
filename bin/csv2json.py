# coding: utf-8

import os, sys
import argparse

from tmnt.utils.csv2json import process_csv

parser = argparse.ArgumentParser('Convert CSV to json list file')
parser.add_argument('--csv_file', type=str, help='CSV input file')
parser.add_argument('--json_out_file', type=str, help='Output JSON list file')
parser.add_argument('--delimiter', type=str, default=',', help='Single character delimiter for csv file (default = ,)')

args = parser.parse_args()

if __name__ == '__main__':
    process_csv(args.csv_file, args.json_out_file, args.delimiter)
