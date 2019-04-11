# coding: utf-8

import xml
import argparse
import xml.etree.ElementTree as ET
import os
import io

def get_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate a Variational AutoEncoder topic model')
    parser.add_argument('--in_dir', type=str, help='Input directory of xml files')
    parser.add_argument('--out_dir', type=str, help='Output directory of plain text')
    return parser


def extract_abstracts_to_files(directory, out_dir):
    xml_files = [f for f in os.listdir(directory) if f.endswith(".xml")]
    for f in xml_files:
        f_ind = 0
        tr = ET.parse(directory + f)
        english_only = tr.findall(".//Article/[Language='eng']")
        for abstract in english_only:
            txt_node = abstract.find(".//AbstractText")
            if txt_node:
                txt = ''.join(txt_node.itertext())
                if txt and len(txt) > 350:
                    out_name = out_dir + f + '_' + str(f_ind) + '.txt'
                    with io.open(out_name, 'w') as op:
                        op.write(txt)
                        op.write('\n')
                    f_ind += 1

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    extract_abstracts_to_files(args.in_dir, args.out_dir)
