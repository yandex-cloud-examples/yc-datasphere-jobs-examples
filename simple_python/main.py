import argparse
from datetime import datetime
from typing import List

from lib import utils


parser = argparse.ArgumentParser(prog='test')
parser.add_argument('-d', '--data', required=True, help='Input file')
parser.add_argument('-r', '--result', required=True, help='Output file')


def run(numbers: List[int]) -> float:
    return utils.get_duration(numbers)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.data) as f:
        numbers = [int(x) for x in f.read().strip().split('\n')]
    result = run(numbers)
    with open(args.result, 'w') as f:
        f.write(str(result))
    with open('metrics.txt', 'w') as f:
        f.write(str(datetime.now()))
