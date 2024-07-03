import sys
import os
from pathlib import Path


def cmd_mln(mln, main_path, workspace_path, preprocessing=True, mln_threads=8,
            mln_iters=100, mln_lr=0.001, mln_threshold_of_rule=0.8):
    # shell(cmd_mln(
    #             mln = Path(__file__).absolute().parent / "pLogicNet" / "mln" / "mln",
    #             main_path = dataspace.absolute(),
    #             workspace_path = workspace_iter.absolute(),
    #             preprocessing = False,
    #             mln_threads = args.mln_threads,
    #             mln_iters = args.mln_iters,
    #             mln_lr = args.mln_lr,
    #             mln_threshold_of_rule = args.mln_threshold_of_rule
    #         ))
    if preprocessing == True:
        return '{} -observed {}/train.txt -out-hidden {}/hidden.txt -save {}/mln_saved.txt -thresh-rule {} -iterations 0 -threads {}'.format(
            mln, main_path, main_path, main_path, mln_threshold_of_rule, mln_threads
        )
    else:
        return '{} -load {}/mln_saved.txt -probability {}/annotation.txt -out-prediction {}/pred_mln.txt -out-rule {}/rule.txt -thresh-triplet 1 -iterations {} -lr {} -threads {}'.format(
            mln, main_path, workspace_path, workspace_path, workspace_path, mln_iters, mln_lr, mln_threads
        )


# This function computes the probability of a triplet being true based on the MLN outputs.
def mln_triplet_prob(h, r, t, hrt2p):
    # KGE algorithms tend to predict triplets like (e, r, e), which are less likely in practice.
    # Therefore, we give a penalty to such triplets, which yields some improvement.
    if h == t:
        if hrt2p.get((h, r, t), 0) < 0.5:
            return -100
        return hrt2p[(h, r, t)]
    else:
        if (h, r, t) in hrt2p:
            return hrt2p[(h, r, t)]
        return 0.5

# This function reads the outputs from MLN and KGE to do evaluation.
# Here, the parameter weight controls the relative weights of both models.
def evaluate(mln_pred_file, kge_pred_file, output_file, weight):
    hit1 = 0
    hit3 = 0
    hit10 = 0
    mr = 0
    mrr = 0
    cn = 0

    hrt2p = dict()
    with open(mln_pred_file, 'r') as fi:
        for line in fi:
            h, r, t, p = line.strip().split('\t')[0:4]
            hrt2p[(h, r, t)] = float(p)

    with open(kge_pred_file, 'r') as fi:
        while True:
            truth = fi.readline()
            preds = fi.readline()

            if (not truth) or (not preds):
                break

            truth = truth.strip().split()
            preds = preds.strip().split()

            h, r, t, mode, original_ranking = truth[0:5]
            original_ranking = int(original_ranking)

            if mode == 'h':
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    preds[k][1] += mln_triplet_prob(e, r, t, hrt2p) * weight

                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == h:
                        ranking = k + 1
                        break
                if ranking == -1:
                    ranking = original_ranking

            if mode == 't':
                preds = [[pred.split(':')[0], float(pred.split(':')[1])] for pred in preds]

                for k in range(len(preds)):
                    e = preds[k][0]
                    preds[k][1] += mln_triplet_prob(h, r, e, hrt2p) * weight

                preds = sorted(preds, key=lambda x:x[1], reverse=True)
                ranking = -1
                for k in range(len(preds)):
                    e = preds[k][0]
                    if e == t:
                        ranking = k + 1
                        break
                if ranking == -1:
                    ranking = original_ranking

            if ranking <= 1:
                hit1 += 1
            if ranking <=3:
                hit3 += 1
            if ranking <= 10:
                hit10 += 1
            mr += ranking
            mrr += 1.0 / ranking
            cn += 1

    mr /= cn
    mrr /= cn
    hit1 /= cn
    hit3 /= cn
    hit10 /= cn

    print('MR: ', mr)
    print('MRR: ', mrr)
    print('Hit@1: ', hit1)
    print('Hit@3: ', hit3)
    print('Hit@10: ', hit10)

    with open(output_file, 'w') as fo:
        fo.write('MR: {}\n'.format(mr))
        fo.write('MRR: {}\n'.format(mrr))
        fo.write('Hit@1: {}\n'.format(hit1))
        fo.write('Hit@3: {}\n'.format(hit3))
        fo.write('Hit@10: {}\n'.format(hit10))

def augment_triplet(pred_file, trip_file, out_file, threshold):
    with open(pred_file, 'r') as fi:
        data = []
        for line in fi:
            l = line.strip().split()
            data += [(l[0], l[1], l[2], float(l[3]))]

    with open(trip_file, 'r') as fi:
        trip = set()
        for line in fi:
            l = line.strip().split()
            trip.add((l[0], l[1], l[2]))

        for tp in data:
            if tp[3] < threshold:
                continue
            trip.add((tp[0], tp[1], tp[2]))

    with open(out_file, 'w') as fo:
        for h, r, t in trip:
            fo.write('{}\t{}\t{}\n'.format(h, r, t))
