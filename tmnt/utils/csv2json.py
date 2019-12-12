# coding: utf-8


import csv
import io
import json


def columns_to_json(header, row):
    assert(len(header) == len(row))
    d = {}
    for a,v in zip(header,row):
        d[a] = v
    return json.dumps(d)
        

def process_csv(csv_file, json_out_file, delimiter=','):
    with io.open(json_out_file, 'w') as out:
        with io.open(csv_file, 'r', newline='') as fp:
            reader = csv.reader(fp, delimiter=delimiter)
            i = 0
            header = []
            for row in reader:
                if i == 0:
                    header = row
                else:
                    js = columns_to_json(header, row)
                    out.write(js)
                    out.write('\n')
                i += 1
    
