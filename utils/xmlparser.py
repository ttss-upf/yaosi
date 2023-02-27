import xml.etree.ElementTree as ET
import os
import sys
import argparse
import tqdm
from utils import config

parser = argparse.ArgumentParser()
parser.add_argument("-file", required=True)
opt = parser.parse_args()
    
def getFiles():
    _src = []  
    _target = []
    for filepath, dirnames, filenames in os.walk(config.data_root_folder + "raw_data/" + opt.file):
        for filename in filenames:
            tree = ET.parse(os.path.join(filepath,filename))
            benchmark = tree.getroot()
            for entries in benchmark[0]:
                triples = []
                for entry in entries:
                    if entry.tag == "modifiedtripleset":
                        for triple in entry:
                            triples.append(triple.text)
                    elif entry.tag == "lex":
                        _src.append(config.seperator.join(triples) + "\n")
                        _target.append(entry.text + "\n")
    src_file = open(config.data_root_folder + 'processed_data/' + opt.file + '/' + opt.file + "_src.txt", 'w+')
    tar_file = open(config.data_root_folder + 'processed_data/' + opt.file + '/' + opt.file + "_tar.txt", 'w+')
    src_file.writelines(_src)
    tar_file.writelines(_target)
    src_file.close()
    tar_file.close()


def main():
    getFiles()
    
if __name__ == "__main__":
    main()
