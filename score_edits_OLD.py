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
    
    fn_one_2_one = [] ##### new list to gather 1-1 true alignments that were missed in pred #####
    num_1_1_aligns = 0 ##### new counter of 1-1 alignments #####

    # convert to sets, remove alignments empty on both sides
    testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    ##### I swapped args: {test_src_id : [test_tgt_ids]} #####
    #####   testalign = predictions
    src_id_to_test_tgt_ids = defaultdict(set)
    for test_src, test_tgt in testalign:
        for test_src_id in test_src:
            for test_tgt_id in test_tgt:
                src_id_to_test_tgt_ids[test_src_id].add(test_tgt_id)
    
    ##### New dict Oct17: {test_tgt_id : [test_src_ids]} #####
    tgt_id_to_test_src_ids = defaultdict(set)
    for test_src, test_tgt in testalign:
        for test_tgt_id in test_tgt:
            for test_src_id in test_src:
                tgt_id_to_test_src_ids[test_tgt_id].add(test_src_id)

    for (gold_src, gold_target) in goldalign:
        ##### these are test cases below, to run on just 1 alignment example #####
        #####   to use: comment out line above and edit indentation #####
        # gold_src = (0, 1)
        # gold_target = (0, 1, 2, 3, 4, 5, 6, 7)
        # for gold_src, gold_target:

        ##### TODO: From original code - ok to not check for this? #####
        # if (test_src, test_target) == ((), ()):
        #     continue
        
        ##### get num of 1-1 alignments
        if (len(gold_src) == 1 and len(gold_target) == 1):
            num_1_1_aligns += 1

        #### Changed strict scoring: increment by num of tp #####
        if (gold_src, gold_target) in testalign:
            ##### There's a strict match between true and pred: pred tgt ids are tp for strict and lax
            tpstrict += len(gold_target)
            tplax += len(gold_target)
            ##### NEW Oct17: also add src ids to tp?
            tpstrict += len(gold_src)
            tplax += len(gold_src)
        else:
            ##### 1. Reconstitute true's src grouping and count intersections of new true and pred
            ##### target_ids changed: now includes pred tgt ids (for all src_gold_id per reconstituted grouping)
            target_ids = set()
            for src_gold_id in gold_src:
                for tgt_id in src_id_to_test_tgt_ids[src_gold_id]:
                    target_ids.add(tgt_id)

            ##### New list Oct17: for pred, gets src ids for each gold tgt id
            source_ids = set()
            for tgt_gold_id in gold_target:
                for src_id in tgt_id_to_test_src_ids[tgt_gold_id]:
                    source_ids.add(src_id)
            
            ##### New addition to target_ids Oct17: tgt ids from gold_target aligned to different src_id
            remainder_src = set(gold_src) ^ source_ids
            if len(remainder_src) != 0:
                for element in remainder_src:
                    for tgt_id in src_id_to_test_tgt_ids[element]:
                        target_ids.add(tgt_id)
                    
            ##### this is a test case below #####
            # for tgt_id in src_id_to_test_tgt_ids[0]:
            #     target_ids.add(tgt_id)

            ##### There is an intersection between true tgt ids and pred tgt ids:
            #####   1.a. Get tplax, fnlax, fplax
            if set(gold_target).intersection(target_ids) != set():
                ##### number of pred tgt ids in intersection are true positives for new lax scoring
                tplax += len(set(gold_target).intersection(target_ids))

                ##### NEW Oct17: number of src tgt ids in intersection are tp's for new lax scoring
                tplax += len(set(gold_src).intersection(source_ids))
                
                ##### NB target_ids are pred, gold_target are true (per verse-level alignment)
                remainder_tgt = set(gold_target) ^ target_ids
                for element in remainder_tgt:
                    if element in gold_target:
                        ##### tgt is in true but not pred, therefore false negative
                        fnlax += 1
                    else:
                        ##### tgt is in pred but not in true, therefore false positive
                        fplax += 1
                
                ##### NEW Oct17: account for errors on src side
                #####   remainder_src: these ids in pred but not in true, therefore false positives
                fplax += len(remainder_src)
                #####   id in true src but not in pred src (source_ids), therefore false negative
                for element in gold_src:
                    if element in source_ids:
                        continue
                    else:
                        fnlax += 1

                ##### 1.b We still need to increment fpstrict: every pred tgt id is a false positive for strict scoring
                #####   NEW Oct17: same for every pred src id
                fpstrict += len(target_ids)
                fpstrict += len(source_ids)

            ##### There's no intersection between true and pred: then every pred tgt id is a false positive
            #####   NB print statement is to check if this condition is met by test sets (not the case in en1-en2)
            else:
                fpstrict += len(target_ids)
                fplax += len(target_ids)
                ##### NEW Oct17: also for source_ids?
                fpstrict += len(source_ids)
                fplax += len(source_ids)
                print('++++++++++++ pred is not in gold and there was no intersection between reconstituted sets ++++++++++++')

            ##### TODO: this code block below may be redundant. And may need to add fnstrict to code block above, where 'no strict match and no intersection' condition actually happens?
            ##### 2. There was no strict match, therefore every true tgt id is a false negative ####
            fnstrict += len(gold_target)
            ##### NEW Oct17: also gold_src? and also applies to fnlax?
            fnstrict += len(gold_src)
            fnlax += len(gold_target)
            fnlax += len(gold_src)

            ##### get list of 1-1 alignments that were missed #####
            if (len(gold_src) == 1 and len(gold_target) == 1):
                fn_one_2_one.append((gold_src, gold_target))

    print(f'tpstrict is {tpstrict}, fpstrict is {fpstrict}, tplax is {tplax}, fplax is {fplax}, fnlax is {fnlax}, fnstrict is {fnstrict}')
    print(f'There are {len(fn_one_2_one)} false negatives from {num_1_1_aligns} 1-1 alignments. The fn are:\n {fn_one_2_one}')
    return np.array([tpstrict, fpstrict, tplax, fplax, fnlax, fnstrict], dtype=np.int32)


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
