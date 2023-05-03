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



def build_supersets(supersets_tuple, 
src_id_2_other_tgt_ids_dict, 
tgt_id_2_other_src_ids_dict,
src_id_2_self_tgt_ids_dict,
tgt_id_2_self_src_ids_dict):
    # initialize supersets
    superset_src = supersets_tuple[0]
    superset_tgt = supersets_tuple[1]

    idx_alignments = set()
    add_to_superset_src = set()
    add_to_superset_tgt = set()

    # for each src id, find tgt in other and then add corresponding src in self
    for src_id in superset_src:
        # add alignment idx of self for first iteration
        if src_id in src_id_2_self_tgt_ids_dict.keys():
            idx_alignments.add(src_id_2_self_tgt_ids_dict[src_id]["alignment_idx"])
            if src_id in src_id_2_other_tgt_ids_dict.keys():
                # get tgt_ids in other to investigate
                tgt_ids_in_other = src_id_2_other_tgt_ids_dict[src_id]["aligned_ids"]
                for tgt_id in tgt_ids_in_other:
                    if tgt_id in tgt_id_2_self_src_ids_dict.keys():
                        # save src ids in self that align to tgt_id in other
                        for src_id in tgt_id_2_self_src_ids_dict[tgt_id]["aligned_ids"]:
                            add_to_superset_src.add(src_id)
                            # add idx of alignment that this new src id comes from
                            idx_alignments.add(src_id_2_self_tgt_ids_dict[src_id]["alignment_idx"])
                            # add tgt_id if adding src_id: i.e. add full new alignment in self to superset
                            superset_tgt.add(tgt_id)
                            idx_alignments.add(tgt_id_2_self_src_ids_dict[tgt_id]["alignment_idx"])
    for src_id in add_to_superset_src:
        superset_src.add(src_id)

    # for each tgt id, find src in other and then add corresponding tgt in self
    for tgt_id in superset_tgt:
        if tgt_id in tgt_id_2_self_src_ids_dict.keys():
            # add alignment idx of self for first iteration
            idx_alignments.add(tgt_id_2_self_src_ids_dict[tgt_id]["alignment_idx"])
            # get src ids in other to investigate
            if tgt_id in tgt_id_2_other_src_ids_dict.keys():
                src_ids_to_investigate = tgt_id_2_other_src_ids_dict[tgt_id]["aligned_ids"]
                for src_id in src_ids_to_investigate:
                    # save tgt ids in self that align to src_id in other
                    if src_id in src_id_2_self_tgt_ids_dict.keys():
                        for tgt_id in src_id_2_self_tgt_ids_dict[src_id]["aligned_ids"]:
                            add_to_superset_tgt.add(tgt_id)
                            # add idx of alignment that this new tgt_id in self comes from
                            idx_alignments.add(tgt_id_2_self_src_ids_dict[tgt_id]["alignment_idx"])
                            # add corresponding src id in self to add full new alignment to superset
                            superset_src.add(src_id)
                            idx_alignments.add(src_id_2_self_tgt_ids_dict[src_id]["alignment_idx"])
    for tgt_id in add_to_superset_tgt:
        superset_tgt.add(tgt_id)

    current_superset = (superset_src,superset_tgt)

    return current_superset, idx_alignments


def get_corresponding_sentences(supersets_tuple, 
src_id_2_gold_tgt_ids_dict, 
tgt_id_2_gold_src_ids_dict,
src_id_2_test_tgt_id_dict,
tgt_id_2_test_src_id_dict
):
'''
Not used? Delete?
'''
    # initialize supersets
    superset_src = supersets_tuple[0]
    superset_tgt = supersets_tuple[1]

    # initialize sets to save idx of gold and test aligns
    idx_gold_alignments = set()
    idx_test_alignments = set()

    # for each src id, get tgt in gold and test
    for src_id in superset_src:
        # TODO: does adding idx at the outset like in row below mess up results? (To deal with null alignments)
        idx_gold_alignments.add(src_id_2_gold_tgt_ids_dict[src_id]["alignment_idx"])
        if src_id not in src_id_2_gold_tgt_ids_dict.keys():
            # then superset remains unchanged
            superset_tgt = superset_tgt
        else:
            for tgt_id in src_id_2_gold_tgt_ids_dict[src_id]["aligned_ids"]:
                superset_tgt.add(tgt_id)
                # add idx of gold alignment the tgt_id came from
                idx_gold_alignments.add(src_id_2_gold_tgt_ids_dict[src_id]["alignment_idx"])
        if src_id not in src_id_2_test_tgt_id_dict.keys():
            superset_tgt = superset_tgt
        else:
            for tgt_id in src_id_2_test_tgt_id_dict[src_id]["aligned_ids"]:
                superset_tgt.add(tgt_id)
                # add idx of test alignment the tgt_id came from
                idx_test_alignments.add(src_id_2_test_tgt_id_dict[src_id]["alignment_idx"])
        
    # for each tgt id, get src in gold and test
    for tgt_id in superset_tgt:
        if tgt_id not in tgt_id_2_gold_src_ids_dict.keys():
            superset_src = superset_src
        else:
            for src_id in tgt_id_2_gold_src_ids_dict[tgt_id]["aligned_ids"]:
                superset_src.add(src_id)
                # add idx of gold alignment the src_id came from
                idx_gold_alignments.add(tgt_id_2_gold_src_ids_dict[tgt_id]["alignment_idx"])
        if tgt_id not in tgt_id_2_test_src_id_dict.keys():
            superset_src = superset_src
        else:
            for src_id in tgt_id_2_test_src_id_dict[tgt_id]["aligned_ids"]:
                superset_src.add(src_id)
                # add idx of test alignment the src_id came from
                idx_test_alignments.add(tgt_id_2_test_src_id_dict[tgt_id]["alignment_idx"])
    
    current_superset = (superset_src,superset_tgt)
    return current_superset, idx_gold_alignments, idx_test_alignments



def _precision(goldalign, testalign):
    # convert to sets, remove alignments empty on both sides
    # testalign = set([(tuple(x), tuple(y)) for x, y in testalign if len(x) or len(y)])
    # goldalign = set([(tuple(x), tuple(y)) for x, y in goldalign if len(x) or len(y)])

    src_id_to_gold_tgt_ids = {}
    for idx, (gold_src, gold_tgt) in enumerate(goldalign):
        # print((gold_src, gold_tgt))
        for gold_src_id in gold_src:
            # print(gold_src_id)
            if gold_src_id in src_id_to_gold_tgt_ids.keys():
                for gold_tgt_id in gold_tgt:
                    src_id_to_gold_tgt_ids[gold_src_id]["aligned_ids"].add(gold_tgt_id)
            else:
                # build dict with data per src id
                inner_dict = {}
                inner_dict["aligned_ids"] = set()
                for gold_tgt_id in gold_tgt:
                    inner_dict["aligned_ids"].add(gold_tgt_id)
                inner_dict["alignment_idx"] = idx
                # add to outer dict
                src_id_to_gold_tgt_ids[gold_src_id] = inner_dict

    # print(f"src id to gold tgt dict is {src_id_to_gold_tgt_ids}")

    tgt_id_to_gold_src_ids = {}
    for idx, (gold_src, gold_tgt) in enumerate(goldalign):
        for gold_tgt_id in gold_tgt:
            if gold_tgt_id in tgt_id_to_gold_src_ids.keys():
                for gold_src_id in gold_src:
                    tgt_id_to_gold_src_ids[gold_tgt_id]["aligned_ids"].add(gold_src_id)
            else:
                inner_dict = {}
                inner_dict["aligned_ids"] = set()
                for gold_src_id in gold_src:
                    inner_dict["aligned_ids"].add(gold_src_id)
                inner_dict["alignment_idx"] = idx
                # add to outer dict
                tgt_id_to_gold_src_ids[gold_tgt_id] = inner_dict

    # print(f"tgt to gold src: {tgt_id_to_gold_src_ids}")

    tgt_id_to_test_src_id = {}
    for idx, (test_src, test_tgt) in enumerate(testalign):
        for test_tgt_id in test_tgt:
            if test_tgt_id in tgt_id_to_test_src_id.keys():
                for test_src_id in test_src:
                    tgt_id_to_test_src_id[test_tgt_id]["aligned_ids"].add(test_src_id)
            else:
                inner_dict = {}
                inner_dict["aligned_ids"] = set()
                for test_src_id in test_src:
                    inner_dict["aligned_ids"].add(test_src_id)
                inner_dict["alignment_idx"] = idx
                # add to outer dict
                tgt_id_to_test_src_id[test_tgt_id] = inner_dict
    
    # print(f"tgt to test src: {tgt_id_2_test_src_id}")

    src_id_to_test_tgt_id = {}
    for idx, (test_src, test_tgt) in enumerate(testalign):
        for test_src_id in test_src:
            if test_src_id in src_id_to_test_tgt_id.keys():
                for test_tgt_id in test_tgt:
                    src_id_to_test_tgt_id[test_src_id]["aligned_ids"].add(test_tgt_id)
            else:
                inner_dict = {}
                inner_dict["aligned_ids"] = set()
                for test_tgt_id in test_tgt:
                    inner_dict["aligned_ids"].add(test_tgt_id)
                inner_dict["alignment_idx"] = idx
                # add to outer dict
                src_id_to_test_tgt_id[test_src_id] = inner_dict

    # print(f"src to test tgt: {src_id_2_test_tgt_id}")

    # 1. Iterate through testalign, build and save supersets from test
    supersets_from_test = []
    superset_to_testalign_idx_from_test = {}

    for (test_src, test_target) in testalign:
        # print(f"test is {(test_src, test_target)}")
        if (test_src, test_target) == ((), ()):
            continue
        # if empty on one side only, change to "null" (for superset reconstruction)
        if test_src == ():
            test_src = ["null"]
        if test_target == ():
            test_target = ["null"]
        idx_from_test = set()
        previous_superset = (set(test_src), set(test_target))
        current_superset = set()
        while current_superset != previous_superset:
            current_superset, idx_test_alignments_ = build_supersets(
                previous_superset,
                src_id_to_gold_tgt_ids,
                tgt_id_to_gold_src_ids,
                src_id_to_test_tgt_id,
                tgt_id_to_test_src_id
            )
            # save idx alignments visited
            for idx in idx_test_alignments_:
                idx_from_test.add(idx)
            # print(f"inside loop current superset is {current_superset}")
            previous_superset = current_superset

        # print(f"loop is done, current superset is {current_superset}")
        # add superset 
        supersets_from_test.append(current_superset)
        superset_to_testalign_idx_from_test[str(current_superset)] = idx_from_test
    # print(f"supersets from test are {supersets_from_test}")
    
    # 2. Iterate through goldalign, build and save supersets from gold
    supersets_from_gold = []
    superset_to_goldalign_idx_from_gold = {}

    for (gold_src, gold_tgt) in goldalign:
        # print(f"gold is {(gold_src, gold_tgt)}")
        # if empty on one side only, change to "null" (for superset reconstruction)
        if gold_src == ():
            gold_src = ["null"]
        if gold_tgt == ():
            gold_tgt = ["null"]
        idx_from_gold = set()
        previous_superset = (set(gold_src), set(gold_tgt))
        current_superset = set()
        while current_superset != previous_superset:
            current_superset, idx_gold_alignments_ = build_supersets(
                previous_superset,
                src_id_to_test_tgt_id,
                tgt_id_to_test_src_id,
                src_id_to_gold_tgt_ids,
                tgt_id_to_gold_src_ids
            )
            # save idx of gold alignments visited
            for idx in idx_gold_alignments_:
                idx_from_gold.add(idx)
            # print(f"inside loop current superset is {current_superset}")
            previous_superset = current_superset

        # print(f"loop is done, current superset is {current_superset}")
        # add superset 
        supersets_from_gold.append(current_superset)
        superset_to_goldalign_idx_from_gold[str(current_superset)] = idx_from_gold
        
    # print(f"supersets from gold are {supersets_from_gold}")

    # 3. Number of tp's = number of matches between test-reconstructed and gold-reconstructed supersets
    matching_supersets = []
    not_matches = []
    for superset in supersets_from_test:
        if superset in matching_supersets:
            continue
        if superset in supersets_from_gold:
            matching_supersets.append(superset)
        else:
            not_matches.append(superset)
    
    print(f"len of not_matches is {len(not_matches)}")
    print(not_matches)
    print(f"matches are {matching_supersets}")
    print(f"len of matches is {len(matching_supersets)}")
    # print(superset_to_goldalign_idx_from_gold)
    # print(superset_to_goldalign_idx_from_test)

    # get number of gold alignments included in each matching superset
    num_correct = 0
    for match in matching_supersets:
        num_goldaligns = len(superset_to_goldalign_idx_from_gold[str(match)])
        num_correct += num_goldaligns
    
    print(superset_to_goldalign_idx_from_gold)
    print(superset_to_goldalign_idx_from_gold.values())
    print(f"number correct is {num_correct}")

    return num_correct / len(goldalign)
    

def score_multiple(gold_list, test_list, value_for_div_by_0=0.0):
    # accumulate counts for all gold/test files
    pcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    rcounts = np.array([0, 0, 0, 0], dtype=np.int32)
    # for goldalign, testalign in zip(gold_list, test_list):
    #     pcounts += _precision(goldalign=goldalign, testalign=testalign)
    #     # recall is precision with no insertion/deletion and swap args
    #     test_no_del = [(x, y) for x, y in testalign if len(x) and len(y)]
    #     gold_no_del = [(x, y) for x, y in goldalign if len(x) and len(y)]
    #     rcounts += _precision(goldalign=test_no_del, testalign=gold_no_del)

    # Compute results
    # pcounts: tpstrict,fnstrict,tplax,fnlax
    # rcounts: tpstrict,fpstrict,tplax,fplax

    for goldalign, testalign in zip(gold_list, test_list):
        accuracy_score = _precision(goldalign, testalign)

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
                  f1_lax=flax,
                  accuracy=accuracy_score)

    return result


def log_final_scores(res):
    print(' ---------------------------------', file=sys.stderr)
    print('|             |  Strict |    Lax  |', file=sys.stderr)
    print('| Precision   |   {precision_strict:.3f} |   {precision_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| Recall      |   {recall_strict:.3f} |   {recall_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| F1          |   {f1_strict:.3f} |   {f1_lax:.3f} |'.format(**res), file=sys.stderr)
    print('| Accuracy    |   {accuracy:.3f} |   "n/a" |'.format(**res), file=sys.stderr)
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
