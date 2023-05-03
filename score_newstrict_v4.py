#!/usr/bin/env python3

"""
Copyright 2019 Brian Thompson

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import argparse
import sys
from collections import defaultdict

import numpy as np

from dp_utils import read_alignments

"""
Faster implementation of lax and strict precision and recall, based on
   https://www.aclweb.org/anthology/W11-4624/.

"""


def _precision(goldalign, testalign):
    """
    Computes tpstrict, fpstrict, tplax, fplax for gold/test alignments
    """
    tpstrict = 0  # true positive strict counter
    tplax = 0     # true positive lax counter
    fpstrict = 0  # false positive strict counter
    fplax = 0     # false positive lax counter
    fnlax = 0     ##### new false negative lax counter #####
    fnstrict = 0  ##### new false negative strict counter #####
    
    # convert to sets, remove alignments empty on both sides
    testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    src_id_to_test_tgt_ids = defaultdict(set)
    for test_src, test_tgt in testalign:
        for test_src_id in test_src:
            for test_tgt_id in test_tgt:
                src_id_to_test_tgt_ids[test_src_id].add(test_tgt_id)
    
    tgt_id_to_test_src_ids = defaultdict(set)
    for test_src, test_tgt in testalign:
        for test_tgt_id in test_tgt:
            for test_src_id in test_src:
                tgt_id_to_test_src_ids[test_tgt_id].add(test_src_id)

    supersets = []
    for (gold_src, gold_target) in goldalign:
        if (gold_src, gold_target) == ((), ()):
            continue

        #### Changed strict scoring: increment by num of tp #####
        if (gold_src, gold_target) in testalign:
            tpstrict += 1
        
        else:
            # for every id in goldalign, get corresponding ids in test
            new_test_tgt_ids = set()
            for src_gold_id in gold_src:
                for tgt_id in src_id_to_test_tgt_ids[src_gold_id]:
                    new_test_tgt_ids.add(tgt_id)

            # repeat for every tgt id in gold
            new_test_src_ids = set()
            for tgt_gold_id in gold_target:
                for src_id in tgt_id_to_test_src_ids[tgt_gold_id]:
                    new_test_src_ids.add(src_id)
            
            reconstructed_test = (new_test_src_ids,new_test_tgt_ids)
            # convert goldsrc and goldtgt to sets to match reconstructed_test
            if reconstructed_test == (set(gold_src), set(gold_target)):
                tpstrict += 1
            supersets.append(reconstructed_test)
    
    # print(f'tpstrict is {tpstrict}, fpstrict is {fpstrict}, tplax is {tplax}, fplax is {fplax}, fnlax is {fnlax}, fnstrict is {fnstrict}')
    # return np.array([tpstrict, fpstrict, tplax, fplax, fnlax, fnstrict], dtype=np.int32)
    return tpstrict, supersets


def score_multiple(gold_list, test_list, value_for_div_by_0=0.0):
    # accumulate counts for all gold/test files
    pcounts = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    # rcounts = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    for goldalign, testalign in zip(gold_list, test_list):
        pcounts += _precision(goldalign=goldalign, testalign=testalign)
        ##### I calculate recall using fn values from pcounts #####
        # # recall is precision with no insertion/deletion and swap args
        # test_no_del = [(x, y) for x, y in testalign if len(x) and len(y)]
        # gold_no_del = [(x, y) for x, y in goldalign if len(x) and len(y)]
        # rcounts += _precision(goldalign=test_no_del, testalign=gold_no_del)

    # Compute results
    ##### pcounts: tpstrict, fpstrict, tplax, fplax, fnlax, fnstrict
    #####   precision = tp / (tp + fp)
    #####   recall = tp / (tp + fn)

    ##### precision strict and lax: two if blocks below unchanged #####
    if pcounts[0] + pcounts[1] == 0:
        pstrict = value_for_div_by_0
    else:
        pstrict = pcounts[0] / float(pcounts[0] + pcounts[1])

    if pcounts[2] + pcounts[3] == 0:
        plax = value_for_div_by_0
    else:
        plax = pcounts[2] / float(pcounts[2] + pcounts[3])

    ##### recall strict and lax: I use new fnlax and fnstrict from pcounts instead of rcounts #####
    if pcounts[0] + pcounts[5] == 0:
        rstrict = value_for_div_by_0
    else:
        rstrict = pcounts[0] / float(pcounts[0] + pcounts[5])

    if pcounts[2] + pcounts[4] == 0:
        rlax = value_for_div_by_0
    else:
        rlax = pcounts[2] / float(pcounts[2] + pcounts[4])

    ##### F1 code unchanged #####
    if (pstrict + rstrict) == 0:
        fstrict = value_for_div_by_0
    else:
        fstrict = 2 * (pstrict * rstrict) / (pstrict + rstrict)

    if (plax + rlax) == 0:
        flax = value_for_div_by_0
    else:
        flax = 2 * (plax * rlax) / (plax + rlax)

    result = dict(recall_strict=rstrict,
                  recall_lax=rlax,
                  precision_strict=pstrict,
                  precision_lax=plax,
                  f1_strict=fstrict,
                  f1_lax=flax)

    return result


def log_final_scores(res):
    print(' ---------------------------------', file=sys.stderr)
    print('|             |  Strict |    Lax  |', file=sys.stderr)
    print('| Precision   |   {precision_strict:.3f} |   {precision_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| Recall      |   {recall_strict:.3f} |   {recall_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| F1          |   {f1_strict:.3f} |   {f1_lax:.3f} |'.format(**res), file=sys.stderr)
    print(' ---------------------------------', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        'Compute strict/lax precision and recall for one or more pairs of gold/test alignments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--test', type=str, nargs='+', required=True,
                        help='one or more test alignment files')

    parser.add_argument('-g', '--gold', type=str, nargs='+', required=True,
                        help='one or more gold alignment files')

    args = parser.parse_args()

    if len(args.test) != len(args.gold):
        raise Exception('number of gold/test files must be the same')

    gold_list = [read_alignments(x) for x in args.gold]
    test_list = [read_alignments(x) for x in args.test]

    res = score_multiple(gold_list=gold_list, test_list=test_list)
    log_final_scores(res)

if __name__ == '__main__':
    main()