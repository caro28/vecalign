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

    # convert to sets, remove alignments empty on both sides
    testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    # mappings from source test sentence idxs to
    #    target gold sentence idxs for which the source test sentence 
    #    was found in corresponding source gold alignment
    src_id_to_gold_tgt_ids = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for gold_src_id in gold_src:
            for gold_tgt_id in gold_tgt:
                src_id_to_gold_tgt_ids[gold_src_id].add(gold_tgt_id)
    
    ##### new dict: maps tgt id to true source id
    tgt_id_to_gold_src_ids = defaultdict(set)
    for gold_src, gold_tgt in goldalign:
        for gold_tgt_id in gold_tgt:
            for gold_src_id in gold_src:
                tgt_id_to_gold_src_ids[gold_tgt_id].add(gold_src_id)

    # need dict: tgt_id_2_pred_src_id
    tgt_id_2_pred_src_id = defaultdict(set)
    for test_src, test_tgt in testalign:
        for test_tgt_id in test_tgt:
            for test_src_id in test_src:
                tgt_id_2_pred_src_id[test_tgt_id].add(test_src_id)
    
    # need dict: src_id_2_pred_tgt_id
    src_id_2_pred_tgt_id = defaultdict(set)
    for test_src, test_tgt in testalign:
        for test_src_id in test_src:
            for test_tgt_id in test_tgt:
                src_id_2_pred_tgt_id[test_src_id].add(test_tgt_id)

    for (test_src, test_target) in testalign:
        if (test_src, test_target) == ((), ()):
            continue
        if (test_src, test_target) in goldalign:
            # strict match
            tpstrict += 1
            print(f"$$$ got a tpstrict with test {(test_src, test_target)}")
            # tplax += 1
        else:
            # get gold tgt ids for pred_src ids
            reconstructed_gold_tgt_ids = set()
            for src_test_id in test_src:
                for tgt_id in src_id_to_gold_tgt_ids[src_test_id]:
                    reconstructed_gold_tgt_ids.add(tgt_id)
            
            # get gold src ids for pred_tgt ids
            reconstructed_gold_src_ids = set()
            for tgt_test_id in test_target:
                for src_id in tgt_id_to_gold_src_ids[tgt_test_id]:
                    reconstructed_gold_src_ids.add(src_id)

            # check for extra elements --> automatically not tp
            test_tgt_extra = set(test_target) - reconstructed_gold_tgt_ids
            test_src_extra = set(test_src) - reconstructed_gold_src_ids
            if (len(test_tgt_extra) != 0 or len(test_src_extra) != 0):
                fpstrict += 1
                print(f"**** got a fpstrict with test {(test_src, test_target)}")
            else:
                # check for missing elements and add corresponding ids to reconstructed sets
                tgt_missing_from_test = reconstructed_gold_tgt_ids - set(test_target)
                src_missing_from_test = reconstructed_gold_src_ids - set(test_src)
                # build reconstructed pred for tgt and src (pulling from other pred instances)
                reconstructed_test_src_ids = set()
                for tgt_id in tgt_missing_from_test:
                    for src_id in tgt_id_2_pred_src_id[tgt_id]:
                        reconstructed_test_src_ids.add(src_id)
                reconstructed_test_tgt_ids = set()
                for src_id in src_missing_from_test:
                    for tgt_id in src_id_2_pred_tgt_id[src_id]:
                        reconstructed_test_tgt_ids.add(tgt_id)
                # add test_src and test_tgt to reconstructed pred sets
                for tgt_id in test_target:
                    reconstructed_test_tgt_ids.add(tgt_id)
                for src_id in test_src:
                    reconstructed_test_src_ids.add(src_id)

                # if reconstructed test and reconstructed gold are matches, then pred is correct
                if (reconstructed_test_src_ids == reconstructed_gold_src_ids and reconstructed_test_tgt_ids == reconstructed_gold_tgt_ids):
                    tpstrict += 1
                    print(f"===== got a tpstrict with test {(test_src, test_target)}")
                else:
                    fpstrict += 1
                    print(f"+++++ got a fpstrict with test {(test_src, test_target)}")

    return np.array([tpstrict, fpstrict, tplax, fplax], dtype=np.int32)


def score_multiple(gold_list, test_list, value_for_div_by_0=0.0):
    # accumulate counts for all gold/test files
    pcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    rcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    for goldalign, testalign in zip(gold_list, test_list):
        pcounts += _precision(goldalign=goldalign, testalign=testalign)
        # recall is precision with no insertion/deletion and swap args
        test_no_del = [(x, y) for x, y in testalign if len(x) and len(y)]
        gold_no_del = [(x, y) for x, y in goldalign if len(x) and len(y)]
        rcounts += _precision(goldalign=test_no_del, testalign=gold_no_del)

    # Compute results
    # pcounts: tpstrict,fnstrict,tplax,fnlax
    # rcounts: tpstrict,fpstrict,tplax,fplax

    if pcounts[0] + pcounts[1] == 0:
        pstrict = value_for_div_by_0
    else:
        pstrict = pcounts[0] / float(pcounts[0] + pcounts[1])

    if pcounts[2] + pcounts[3] == 0:
        plax = value_for_div_by_0
    else:
        plax = pcounts[2] / float(pcounts[2] + pcounts[3])

    if rcounts[0] + rcounts[1] == 0:
        rstrict = value_for_div_by_0
    else:
        rstrict = rcounts[0] / float(rcounts[0] + rcounts[1])

    if rcounts[2] + rcounts[3] == 0:
        rlax = value_for_div_by_0
    else:
        rlax = rcounts[2] / float(rcounts[2] + rcounts[3])

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
