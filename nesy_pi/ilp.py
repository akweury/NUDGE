# Created by shaji on 21-Apr-23
import datetime
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, recall_score
from tqdm import tqdm
import torch

from src import config
from nesy_pi import eval_clause_infer
from nesy_pi import semantic as se
from nesy_pi.aitk import ai_interface
from nesy_pi.aitk.utils.fol import bk, logic
from nesy_pi.aitk.utils import nsfr_utils, visual_utils, lang_utils, logic_utils, log_utils
from nesy_pi.aitk.utils.fol.refinement import RefinementGenerator

from nesy_pi.ilp_utils import remove_duplicate_clauses, remove_conflict_clauses, update_refs
from nesy_pi.pi_utils import gen_clu_pi_clauses, gen_exp_pi_clauses, generate_new_predicate


def clause_eval(args, lang, FC, clauses, step, eval_data=None):
    # clause evaluation
    NSFR = ai_interface.get_nsfr(args, lang, FC, clauses)
    # evaluate new clauses
    target_preds = [clauses[0].head.pred.name]
    img_scores, p_pos = get_clause_score(NSFR, args, target_preds, eval_data)
    clause_scores = get_suff_score(img_scores[:, :, args.index_pos], img_scores[:, :, args.index_neg])
    return img_scores, clause_scores, p_pos


def clause_robust_eval(args, lang, FC, clauses, step, eval_data=None):
    # clause evaluation
    NSFR = ai_interface.get_nsfr(args, lang, FC, clauses)
    # evaluate new clauses
    target_preds = [clauses[0].head.pred.name]
    img_scores = get_clause_score(NSFR, args, target_preds, eval_data, args.train_group_pos, args.train_group_neg)
    clause_scores = get_clause_3score(img_scores[:, :, args.index_pos], img_scores[:, :, args.index_neg], args, step)
    return img_scores, clause_scores


# def clause_prune(args, clauses, score_all, scores):
#     # classify clauses
#     clause_with_scores = prune_low_score_clauses(clauses, score_all, scores, args)
#     # print best clauses that have been found...
#     clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)
#
#     # new_max, higher = logic_utils.get_best_clauses(refs_extended, scores, step, args, max_clause)
#     # max_clause, found_sn = check_result(args, clause_with_scores, higher, max_clause, new_max)
#
#     if args.pi_top > 0:
#         clauses, clause_with_scores = prune_clauses(clause_with_scores, args)
#     else:
#         clauses = logic_utils.top_select(clause_with_scores, args)
#
#     return clauses, clause_with_scores


def inv_consts(args, p_pos, clauses, lang):
    const_data = []
    new_consts = []
    for c_with_score in clauses:
        # invent const
        if c_with_score[1][1] > args.suff_min:
            rule_consts = []
            c = c_with_score[0]
            for atom in c.body:
                if "phi" == atom.pred.name or "rho" == atom.pred.name:
                    rule_consts.append([atom.terms[0].name, atom.terms[-2].name, atom.terms[-2].dtype])
            for rule_const in rule_consts:
                rule_consts_atom_id = []
                for a_i, atom in enumerate(lang.atoms):
                    if len(atom.terms) > 1 and atom.terms[0].name == rule_const[0] and atom.terms[-2].name == \
                            rule_const[1]:
                        rule_consts_atom_id.append(a_i)
                const_values = p_pos[:, rule_consts_atom_id].unique()
                const_values = const_values[const_values != 0]
                const_data.append([rule_const[-1], const_values])

    # create new const object

    if len(const_data) > 0:
        for new_const in const_data:
            new_const = lang.inv_new_const(new_const[0], new_const[1])
            new_consts.append(new_const)

        lang.generate_atoms()
        return True, list(set(new_consts))

    return False, new_consts


def ilp_search(args, lang, init_clauses, FC):
    """
    given one or multiple neural predicates, searching for high scoring clauses, which includes following steps
    1. extend given initial clauses
    2. evaluate each clause
    3. prune clauses

    """
    extend_step = 0
    clause_with_scores = []
    clauses = init_clauses
    while extend_step <= args.max_step and not args.is_done:
        # clause extension
        clauses = clause_extend(args, lang, clauses)
        if args.is_done:
            break

        # clause evaluation
        img_scores, clause_scores, p_pos = clause_eval(args, lang, FC, clauses, extend_step)
        # classify clauses
        clause_with_scores = sort_clauses_by_score(clauses, img_scores, clause_scores, args)
        # print best clauses that have been found...
        # clause_with_scores = logic_utils.sorted_clauses(clause_with_scores, args)

        # new_max, higher = logic_utils.get_best_clauses(refs_extended, scores, step, args, max_clause)
        # max_clause, found_sn = check_result(args, clause_with_scores, higher, max_clause, new_max)

        # prune clauses
        if args.pi_top > 0:
            clauses, clause_with_scores = prune_clauses(clause_with_scores, args)
        else:
            clauses = logic_utils.top_select(clause_with_scores, args)

        has_new_consts, const_data = inv_consts(args, p_pos, clause_with_scores, lang)
        if has_new_consts:
            VM = ai_interface.get_vm(args, lang)
            FC = ai_interface.get_fc(args, lang, VM, args.rule_obj_num)

        # save data
        all_c = [c[0] for c in lang.all_clauses]
        new_c = [c for c in clause_with_scores if c[0] not in all_c and c[1][1] > 0.9]
        lang.all_clauses += new_c

        done, clause_with_scores = check_result(args, clause_with_scores, lang.all_clauses)
        clauses = [c[0] for c in clause_with_scores]
        extend_step += 1

        if done:
            args.is_done = True
            break

    if len(clauses) > 0:
        lang.clause_with_scores = clause_with_scores
        # args.last_refs = clauses
    return clauses, FC


def explain_scenes(args, lang, clauses):
    """ explaination should improve the sufficient percentage """
    new_explain_pred_with_scores = explain_invention(args, lang, clauses)
    pi_exp_clauses = gen_exp_pi_clauses(args, lang, new_explain_pred_with_scores)
    lang.pi_clauses += pi_exp_clauses


def ilp_pi(args, lang, clauses, e):
    # predicate invention by clustering
    new_clu_pred_with_scores = cluster_invention(args, lang, clauses, e)
    # convert to strings
    new_clauses_str_list, kp_str_list = generate_new_clauses_str_list(args, new_clu_pred_with_scores)
    pi_clu_clauses, pi_kp_clauses = gen_clu_pi_clauses(args, lang, new_clu_pred_with_scores, new_clauses_str_list,
                                                       kp_str_list)
    lang.pi_kp_clauses = extract_kp_pi(lang, pi_kp_clauses, args)
    lang.pi_clauses += pi_clu_clauses

    if len(lang.invented_preds) > 0:
        # add new predicates
        args.no_new_preds = False
        lang.generate_atoms()

    # log
    if args.show_process:
        log_utils.add_lines(f"======  Total PI Number: {len(lang.invented_preds)}  ======", args.log_file)
        for p in lang.invented_preds:
            log_utils.add_lines(f"{p}", args.log_file)

        log_utils.add_lines(f"========== Total {len(lang.pi_clauses)} PI Clauses ======== ", args.log_file)
        for c in lang.pi_clauses:
            log_utils.add_lines(f"{c}", args.log_file)


def ilp_test(args, lang):
    log_utils.add_lines(f"================== ILP TEST ==================", args.log_file)
    if args.show_process:
        log_utils.print_result(args, lang)

    reset_args(args)
    clauses, e = reset_lang(lang, args, args.neural_preds, full_bk=True)

    VM = ai_interface.get_vm(args, lang)
    FC = ai_interface.get_fc(args, lang, VM, e)
    # ILP
    # searching for a proper clause to describe the patterns.

    args.iteration = args.max_step
    # returned clauses have to cover all the positive images and no negative images
    ilp_search(args, lang, clauses, FC)

    sorted_clauses_with_scores = sorted(lang.all_clauses, key=lambda x: x[1][2], reverse=True)
    sorted_clauses_with_scores = [c for c in sorted_clauses_with_scores if c[1][1] > args.sc_th][:args.top_k]

    success, clauses = log_utils.print_test_result(args, lang, sorted_clauses_with_scores)
    return success, sorted_clauses_with_scores


def ilp_predict(NSFR, args, th=None, split='train'):
    pos_pred = torch.tensor(args.val_group_pos)
    neg_pred = torch.tensor(args.val_group_neg)

    predicted_list = []
    target_list = []
    count = 0

    pm_pred = torch.cat((pos_pred, neg_pred), dim=0)
    train_label = torch.zeros(len(pm_pred))
    train_label[:len(pos_pred)] = 1.0

    target_set = train_label.to(torch.int64)

    for i, sample in tqdm(enumerate(pm_pred, start=0)):
        # to cuda
        sample = sample.unsqueeze(0)
        # infer and predict the target probability
        V_T = NSFR(sample).unsqueeze(0)
        predicted = nsfr_utils.get_prob(V_T, NSFR, args).squeeze(1).squeeze(1)
        predicted_list.append(predicted.detach())
        target_list.append(target_set[i])
        count += V_T.size(0)  # batch size

    predicted_all = torch.cat(predicted_list, dim=0).detach().cpu().numpy()
    target_set = torch.tensor(target_list).to(torch.int64).detach().cpu().numpy()

    if th == None:
        fpr, tpr, thresholds = roc_curve(target_set, predicted_all, pos_label=1)
        accuracy_scores = []
        print('ths', thresholds)
        for thresh in thresholds:
            accuracy_scores.append(accuracy_score(
                target_set, [m > thresh for m in predicted_all]))

        accuracies = np.array(accuracy_scores)
        max_accuracy = accuracies.max()
        max_accuracy_threshold = thresholds[accuracies.argmax()]
        rec_score = recall_score(
            target_set, [m > thresh for m in predicted_all], average=None)

        print('target_set: ', target_set, target_set.shape)
        print('predicted: ', predicted_all, predicted.shape)
        print('accuracy: ', max_accuracy)
        print('threshold: ', max_accuracy_threshold)
        print('recall: ', rec_score)

        return max_accuracy, rec_score, max_accuracy_threshold
    else:
        accuracy = accuracy_score(target_set, [m > th for m in predicted_all])
        rec_score = recall_score(
            target_set, [m > th for m in predicted_all], average=None)
        return accuracy, rec_score, th


def ilp_eval(success, args, lang, clauses, g_data):
    scores_dict = {}
    if not success:
        log_utils.add_lines(f"ILP failed.", args.log_file)

    if len(clauses) == 0:
        return
    # target_predicate = [clauses[0].head.pred.name]
    # calculate scores
    VM = ai_interface.get_vm(args, lang)
    FC = ai_interface.get_fc(args, lang, VM, args.rule_obj_num)

    # evaluate all test images at once
    img_scores, clauses_scores = clause_eval(args, lang, FC, clauses, 0, eval_data="test")
    clause_with_scores = sort_clauses_by_score(clauses, img_scores, clauses_scores, args)
    success, clauses = log_utils.print_test_result(args, lang, clause_with_scores)

    for d_i, data_type in enumerate(["true", "false"]):
        if data_type == "true":
            img_sign = config.score_example_index["pos"]
        else:
            img_sign = config.score_example_index["neg"]
        scores_dict[data_type] = {}
        scores_dict[data_type]["clause"] = []
        scores_dict[data_type]["score"] = []
        for img_i in range(len(args.test_data[args.label][d_i])):
            # scores_sorted, scores_indices = torch.sort(img_scores[:, i, img_sign], descending=True)
            score_best = img_scores[0, img_i, img_sign]
            clause_best = clauses[0]
            scores_dict[data_type]["score"].append(score_best)
            scores_dict[data_type]["clause"].append(clause_best)

            # if data_type == "false" and score_best > 0.9:
            #     print("(FP)")
            # elif data_type == "true" and score_best < 0.1:
            #     print("(FN)")
    # visual_utils.visualization(args, lang, scores_dict)

    log_utils.add_lines("===================== top clause score ================================", args.log_file)
    positive_res = torch.tensor(scores_dict['true']['score'])
    negative_res = torch.tensor(scores_dict['false']['score'])
    tp_count = len(positive_res[positive_res > 0.95])
    p_total = len(positive_res)
    fp_count = len(negative_res[negative_res > 0.95])
    f_total = len(negative_res)
    log_utils.add_lines(f"Recall: {tp_count / p_total} ({tp_count}/{p_total})", args.log_file)
    log_utils.add_lines(f"Precision: {tp_count / ((fp_count + tp_count) + 1e-20)} ({tp_count}/{fp_count + tp_count})",
                        args.log_file)
    log_utils.add_lines("=======================================================================", args.log_file)

    return scores_dict


def ilp_robust_eval(args, lang):
    scores_dict = {}
    clauses = lang.all_clauses
    # target_predicate = [clauses[0].head.pred.name]
    # calculate scores
    VM = ai_interface.get_vm(args, lang)
    FC = ai_interface.get_fc(args, lang, VM, args.group_e)

    # evaluate all test images at once
    img_scores, clauses_scores = clause_robust_eval(args, lang, FC, clauses, 0, eval_data="train")
    clause_with_scores = sort_clauses_by_score(clauses, img_scores, clauses_scores, args)
    success, clauses = log_utils.print_test_result(args, lang, clause_with_scores)

    for data_type in ["true", "false"]:
        if data_type == "true":
            img_sign = config.score_example_index["pos"]
        else:
            img_sign = config.score_example_index["neg"]
        scores_dict[data_type] = {}
        scores_dict[data_type]["clause"] = []
        scores_dict[data_type]["score"] = []
        for img_i in range(len(args.train_group_pos)):
            # scores_sorted, scores_indices = torch.sort(img_scores[:, i, img_sign], descending=True)

            score_best = img_scores[0, img_i, img_sign]
            clause_best = clauses[0]
            scores_dict[data_type]["score"].append(score_best)
            scores_dict[data_type]["clause"].append(clause_best)

    visual_utils.visualization_robust(args, lang, scores_dict)
    return scores_dict


def keep_best_preds(args, lang):
    p_inv_best = sorted(lang.invented_preds_with_scores, key=lambda x: x[1][2], reverse=True)
    p_inv_best = p_inv_best[:args.pi_top]
    p_inv_best = logic_utils.extract_clauses_from_bs_clauses(p_inv_best, "best inv clause", args)

    for new_p in p_inv_best:
        if new_p not in lang.all_invented_preds:
            lang.all_invented_preds.append(new_p)
    for new_c in lang.pi_clauses:
        if new_c not in lang.all_pi_clauses and new_c.head.pred in p_inv_best:
            lang.all_pi_clauses.append(new_c)


def reset_args(args):
    args.is_done = False
    args.iteration = 0
    args.max_clause = [0.0, None]
    args.no_new_preds = False
    args.no_new_preds = True


def reset_lang(lang, args, neural_pred, full_bk):
    e = args.rule_obj_num

    lang.all_clauses = []
    lang.invented_preds_with_scores = []
    init_clause = lang.load_init_clauses(args.label_name, e)
    # update predicates
    lang.update_bk(neural_pred, full_bk)
    # update language
    lang.mode_declarations = lang_utils.get_mode_declarations(e, lang)

    return init_clause, e


def get_clause_3score(score_pos, score_neg, args, c_length=0):
    scores = torch.zeros(size=(3, score_pos.shape[0])).to(args.device)

    # negative scores are inversely proportional to sufficiency scores
    score_negative_inv = 1 - score_neg

    # calculate sufficient, necessary, sufficient and necessary scores
    ness_index = config.score_type_index["ness"]
    suff_index = config.score_type_index["suff"]
    sn_index = config.score_type_index["sn"]
    scores[ness_index, :] = score_pos.sum(dim=1) / score_pos.shape[1]
    scores[suff_index, :] = score_negative_inv.sum(dim=1) / score_negative_inv.shape[1]
    scores[sn_index, :] = scores[0, :] * scores[1, :] * args.weight_tp + c_length * (args.weight_length / args.max_step)
    return scores


def get_suff_score(score_pos, score_neg):
    # negative scores are inversely proportional to sufficiency scores
    data_size = score_pos.shape[1]
    pos_score = score_pos.sum(dim=1) / (0.99 * data_size)
    neg_score = (1 - score_neg).sum(dim=1) / (0.99 * data_size)
    scores = torch.cat((pos_score.unsqueeze(dim=1), neg_score.unsqueeze(dim=1)), dim=1)

    return scores


def get_clause_score(NSFR, args, pred_names, eval_data, pos_group_pred=None, neg_group_pred=None, batch_size=None):
    """ input: clause, output: score """

    if pos_group_pred is None:
        if eval_data == "play":
            pos_group_pred = args.test_data
        elif eval_data == "test":
            pos_group_pred = args.test_data[args.label][0]
        else:
            pos_group_pred = args.train_data[args.label][0]

    if neg_group_pred is None:
        if eval_data == "play":
            neg_group_pred = args.test_data
        elif eval_data == "test":
            neg_group_pred = args.test_data[args.label][1]
        else:
            neg_group_pred = args.train_data[args.label][1]
    if batch_size is None:
        batch_size = args.batch_size

    train_size = len(pos_group_pred)
    bz = args.batch_size
    V_T_pos = torch.zeros(len(NSFR.clauses), train_size, len(NSFR.atoms)).to(args.device)
    V_T_neg = torch.zeros(len(NSFR.clauses), train_size, len(NSFR.atoms)).to(args.device)
    P_pos = torch.zeros(train_size, len(NSFR.atoms)).to(args.device)
    P_neg = torch.zeros(train_size, len(NSFR.atoms)).to(args.device)
    img_scores = torch.zeros(size=(V_T_pos.shape[0], V_T_pos.shape[1], 2)).to(args.device)
    for i in range(int(train_size / batch_size)):
        date_now = datetime.datetime.today().date()
        time_now = datetime.datetime.now().strftime("%H_%M_%S")
        # print(f"({date_now} {time_now}) eval batch {i + 1}/{int(train_size / args.batch_size_train)}")
        l_i = i * bz
        r_i = (i + 1) * bz
        g_tensors_pos = pos_group_pred[l_i:r_i]
        g_tensors_neg = neg_group_pred[l_i:r_i]
        # V_T_pos.dim = clause num * img num * atoms num
        V_T_pos[:, l_i:r_i, :], P_pos[l_i:r_i, :] = NSFR.clause_eval_quick(g_tensors_pos, P_pos[l_i:r_i, :])
        if not eval_data == "play":
            V_T_neg[:, l_i:r_i, :], P_neg[l_i:r_i, :] = NSFR.clause_eval_quick(g_tensors_neg, P_pos[l_i:r_i, :])
    # each score needs an explanation
    score_positive = NSFR.get_target_prediciton(V_T_pos, pred_names, args.device)
    score_positive[score_positive == 1] = 0.99
    if score_positive.size(2) > 1:
        score_positive = score_positive.max(dim=2, keepdim=True)[0]
    index_pos = config.score_example_index["pos"]
    img_scores[:, :, index_pos] = score_positive[:, :, 0]

    if not eval_data == "play":
        score_negative = NSFR.get_target_prediciton(V_T_neg, pred_names, args.device)
        score_negative[score_negative == 1] = 0.99
        if score_negative.size(2) > 1:
            score_negative = score_negative.max(dim=2, keepdim=True)[0]

        index_neg = config.score_example_index["neg"]
        img_scores[:, :, index_neg] = score_negative[:, :, 0]

    return img_scores, P_pos


def sort_clauses_by_score(clauses, scores_all, scores, args):
    clause_with_scores = []
    for c_i, clause in enumerate(clauses):
        clause_with_scores.append((clause, scores[c_i], scores_all[c_i]))

    if len(clause_with_scores) > 0:
        c_sorted = sorted(clause_with_scores, key=lambda x: x[1][1], reverse=True)
        # for c in c_sorted:
        #     log_utils.add_lines(f"clause: {c[0]} {c[1]}", args.log_file)
        return c_sorted

    return clause_with_scores


def clause_extend(args, lang, clauses):
    refs = []
    B_ = []

    refinement_generator = RefinementGenerator(lang=lang)
    for c in clauses:
        refs_i = refinement_generator.refinement_clause(c)
        unused_args, used_args = log_utils.get_unused_args(c)
        refs_i_removed = remove_duplicate_clauses(refs_i, unused_args, used_args, args)
        # remove already appeared refs
        refs_i_removed = list(set(refs_i_removed).difference(set(B_)))
        B_.extend(refs_i_removed)
        refs.extend(refs_i_removed)

    # remove semantic conflict clauses
    refs_no_conflict = remove_conflict_clauses(refs, lang.pi_clauses, args)
    if len(refs_no_conflict) == 0:
        refs_no_conflict = clauses
        args.is_done = True

    if args.show_process:
        log_utils.add_lines(f"=============== extended {len(refs_no_conflict)} clauses =================",
                            args.log_file)
        for ref in refs_no_conflict:
            log_utils.add_lines(f"{ref}", args.log_file)
    return refs_no_conflict


def prune_low_ness_clauses(args, clause_with_scores):
    if args.show_process:
        log_utils.add_lines(f"- score pruning ... ({len(clause_with_scores)} clauses)", args.log_file)
    score_unique_c = []
    appeared_scores = []
    for c in clause_with_scores:
        if c[1][0] > args.ness_th:
            score_unique_c.append(c)
            appeared_scores.append(c[1][0])

    if args.show_process:
        log_utils.add_lines(f"- {len(score_unique_c)} clauses left.", args.log_file)
        for c in score_unique_c:
            positive_score = c[2][:, config.score_example_index["pos"]]
            negative_score = 1 - c[2][:, config.score_example_index["neg"]]
            failed_pos_index = ((positive_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
            failed_neg_index = ((negative_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
            log_utils.add_lines(f"(sco-uni): {c[0]} {torch.round(c[1], decimals=2).reshape(-1)} "
                                f"Failed Pos States: ({len(failed_pos_index)}/{c[2].shape[0]}) "
                                f"Failed Neg States: ({len(failed_neg_index)}/{c[2].shape[0]}) ", args.log_file)
    return score_unique_c


def prune_low_suff_clauses(args, clause_with_scores):
    if args.show_process:
        log_utils.add_lines(f"- score pruning ... ({len(clause_with_scores)} clauses)", args.log_file)
    score_unique_c = []
    appeared_scores = []
    for c in clause_with_scores:
        if c[1][1] > args.suff_min:
            score_unique_c.append(c)
            appeared_scores.append(c[1])
    score_unique_c = sorted(score_unique_c, key=lambda x: x[1][1], reverse=True)
    if args.show_process:
        log_utils.add_lines(f"- {len(score_unique_c)} clauses left.", args.log_file)
        for c in score_unique_c:
            positive_score = c[2][:, config.score_example_index["pos"]]
            negative_score = 1 - c[2][:, config.score_example_index["neg"]]
            failed_pos_index = ((positive_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
            failed_neg_index = ((negative_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
            log_utils.add_lines(f"(suff-uni): {c[0]} {torch.round(c[1], decimals=2).reshape(-1)} "
                                f"Failed Pos States: ({len(failed_pos_index)}/{c[2].shape[0]}) "
                                f"Failed Neg States: ({len(failed_neg_index)}/{c[2].shape[0]}) ", args.log_file)
    return score_unique_c


def prune_semantic_repeat_clauses(args, c_score_pruned):
    if args.show_process:
        log_utils.add_lines(f"- semantic pruning ... ({len(c_score_pruned)} clauses)", args.log_file)
    semantic_unique_c = []
    semantic_repeat_c = []
    appeared_semantics = []
    for c in c_score_pruned:
        c_semantic = logic_utils.get_semantic_from_c(c[0])
        if not eval_clause_infer.eval_semantic_similarity(c_semantic, appeared_semantics, args):
            semantic_unique_c.append(c)
            appeared_semantics.append(c_semantic)
        else:
            semantic_repeat_c.append(c)
    if args.show_process:
        log_utils.add_lines(f"- {len(c_score_pruned)} clauses left.", args.log_file)
        for c in c_score_pruned:
            positive_score = c[2][:, config.score_example_index["pos"]]
            negative_score = 1 - c[2][:, config.score_example_index["neg"]]
            failed_pos_index = ((positive_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
            failed_neg_index = ((negative_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
            log_utils.add_lines(f"(sem-uni): {c[0]} {torch.round(c[1], decimals=2).reshape(-1)} "
                                f"Failed Pos States: ({len(failed_pos_index)}/{c[2].shape[0]}) "
                                f"Failed Neg States: ({len(failed_neg_index)}/{c[2].shape[0]}) ", args.log_file)

    return semantic_unique_c


def prune_repeat_score_clauses(args, c_score_pruned):
    if args.show_process:
        log_utils.add_lines(f"- score unique pruning ... ({len(c_score_pruned)} clauses)", args.log_file)
    # for c in clause_with_scores:
    #     log_utils.add_lines(f"(clause before pruning) {c[0]} {c[1].reshape(3)}", args.log_file)
    score_unique_c = []
    score_repeat_c = []
    appeared_scores = []
    for c in c_score_pruned:
        if not eval_clause_infer.eval_score_similarity(c[1][1], appeared_scores, args.similar_th):
            score_unique_c.append(c)
            appeared_scores.append(c[1][1])
        else:
            score_repeat_c.append(c)
    c_score_pruned = score_unique_c

    if args.show_process:
        log_utils.add_lines(f"- {len(score_unique_c)} clauses left.", args.log_file)
        for c in score_unique_c:
            positive_score = c[2][:, config.score_example_index["pos"]]
            negative_score = 1 - c[2][:, config.score_example_index["neg"]]
            failed_pos_index = ((positive_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
            failed_neg_index = ((negative_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
            log_utils.add_lines(f"(score-uni): {c[0]} {torch.round(c[1], decimals=2).reshape(-1)} "
                                f"Failed Pos States: ({len(failed_pos_index)}/{c[2].shape[0]}) "
                                f"Failed Neg States: ({len(failed_neg_index)}/{c[2].shape[0]}) ", args.log_file)

    return c_score_pruned


def prune_clauses(clause_with_scores, args):
    refs = []
    # prune necessity zero clauses
    if args.score_unique:
        c_score_pruned = prune_low_ness_clauses(args, clause_with_scores)
        c_score_pruned = prune_low_suff_clauses(args, c_score_pruned)

        c_score_pruned = prune_repeat_score_clauses(args, c_score_pruned)
    else:
        c_score_pruned = clause_with_scores
    if args.semantic_unique:
        c_score_pruned = prune_semantic_repeat_clauses(args, c_score_pruned)
    refs += update_refs(c_score_pruned, args)
    return refs, c_score_pruned


def top_kp(args, clauses):
    suff_percents = torch.tensor([c[1][1] for c in clauses])
    ness_percents = torch.tensor([c[1][0] for c in clauses])
    log_utils.add_lines(f"- total ness percent {ness_percents.sum():.3f}", args.log_file)
    indices_ranked = suff_percents.argsort(descending=True)
    percent_count = 0
    top_clauses = []
    for i in indices_ranked:
        if percent_count < args.top_kp:
            percent_count += ness_percents[i]
            top_clauses.append(clauses[i])
    log_utils.add_lines(f"- {len(top_clauses)} clauses left.", args.log_file)

    return top_clauses


def check_result(args, clause_with_scores, all_clauses):
    done = False
    if len(clause_with_scores) == 0:
        return done, []

    all_c = [c[0] for c in all_clauses]
    new_c = [c for c in clause_with_scores if c[0] not in all_c]
    clause_with_scores = all_clauses + new_c

    # rank by sufficiency

    suff_percents = torch.tensor([c[1][1] for c in clause_with_scores])
    indices_ranked = suff_percents.argsort(descending=True)
    c_score_pruned = [clause_with_scores[i] for i in indices_ranked]

    # rank by necessity
    ness_percents = []
    suff_percents = []
    ness_used_data = torch.zeros_like(clause_with_scores[0][2][:, 0], dtype=torch.bool)
    suff_used_data = torch.zeros_like(clause_with_scores[0][2][:, 1], dtype=torch.bool)
    for c in c_score_pruned:
        c_ness_indices = c[2][:, config.score_example_index["pos"]] > 0.9
        c_suff_indices = c[2][:, config.score_example_index["neg"]] > 0.9
        ness_percent = (~ness_used_data * c_ness_indices).sum() / len(ness_used_data)
        suff_percent = (~suff_used_data * c_suff_indices).sum() / len(suff_used_data)
        ness_used_data[c_ness_indices] = True
        suff_used_data[c_suff_indices] = True
        ness_percents.append(ness_percent)
        suff_percents.append(suff_percent)
    ness_ranked, ness_indices = torch.tensor(ness_percents).sort(descending=True)
    suff_rankded = torch.tensor(suff_percents)[ness_indices]

    ness_maximize_c = []
    ness_percent_total = 0
    suff_percent_total = 0
    for c_i in range(min(args.top_k, len(ness_indices))):
        ness_maximize_c.append(c_score_pruned[ness_indices[c_i]])
        ness_percent_total += ness_ranked[c_i]
        suff_percent_total += suff_rankded[c_i]

    saved_ness_percents = 0
    saved_ness_percents_all = []
    saved_suff_percents = 0
    saved_suff_percents_all = []
    if len(all_clauses) > 0:
        saved_ness_used_data = torch.zeros_like(all_clauses[0][2][:, 0], dtype=torch.bool)
        saved_suff_used_data = torch.zeros_like(all_clauses[0][2][:, 1], dtype=torch.bool)
        for c_i, c in enumerate(all_clauses):
            c_ness_indices = c[2][:, config.score_example_index["pos"]] > 0.9
            c_suff_indices = c[2][:, config.score_example_index["neg"]] > 0.9
            ness_percent = (~saved_ness_used_data * c_ness_indices).sum() / len(saved_ness_used_data)
            suff_percent = (~saved_suff_used_data * c_suff_indices).sum() / len(saved_suff_used_data)
            saved_ness_used_data[c_ness_indices] = True
            saved_suff_used_data[c_suff_indices] = True
            saved_ness_percents += ness_percent
            saved_suff_percents += suff_percent

            saved_suff_percents_all.append(suff_percent)
            saved_ness_percents_all.append(ness_percent)

    if saved_ness_percents > args.nc_th:
        done = True
    # log
    print('\n\n')
    for c_i, c in enumerate(ness_maximize_c):
        positive_score = c[2][:, config.score_example_index["pos"]]
        negative_score = 1 - c[2][:, config.score_example_index["neg"]]
        failed_pos_index = ((positive_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
        failed_neg_index = ((negative_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
        log_utils.add_lines(f"{c_i + 1}/{len(ness_maximize_c)} ness:{torch.round(c[1][0], decimals=2):.2f},"
                            f"suff:{torch.round(c[1][1], decimals=2):.2f} "
                            f"+:({len(failed_pos_index)}/{c[2].shape[0]}) "
                            f"-:({len(failed_neg_index)}/{c[2].shape[0]}) {c[0]} ", args.log_file)
    log_utils.add_lines(f"-Total Ness: {ness_percent_total:.2f}\n"
                        f"Total Suff: {suff_percent_total:.2f}\n ", args.log_file)

    log_utils.add_lines(f"- Saved Total {len(all_clauses)} clauses.", args.log_file)
    for c_i, c in enumerate(all_clauses):
        positive_score = c[2][:, config.score_example_index["pos"]]
        negative_score = 1 - c[2][:, config.score_example_index["neg"]]
        failed_pos_index = ((positive_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
        failed_neg_index = ((negative_score < 0.9).nonzero(as_tuple=True)[0]).tolist()
        log_utils.add_lines(f"{c_i + 1}/{len(all_clauses)} "
                            f"contri ness:{saved_ness_percents_all[c_i]:.2f},"
                            f"contri suff:{saved_suff_percents_all[c_i]:.2f},"
                            f"ness:{torch.round(c[1][0], decimals=2):.2f},"
                            f"suff:{torch.round(c[1][1], decimals=2):.2f} "
                            f"+:({len(failed_pos_index)}/{c[2].shape[0]}) "
                            f"-:({len(failed_neg_index)}/{c[2].shape[0]}) {c[0]} ", args.log_file)
    log_utils.add_lines(f"Saved Total Ness: {saved_ness_percents:.2f}\n"
                        f"Saved Total Suff: {(1 - saved_suff_percents):.2f}", args.log_file)
    return done, ness_maximize_c


def explain_invention(args, lang, clauses):
    log_utils.add_lines("- (explain clause) -", args.log_file)

    index_pos = config.score_example_index["pos"]
    index_neg = config.score_example_index["neg"]
    explained_clause = []
    for clause, scores, score_all in lang.clause_with_scores:
        increased_score = scores - scores
        if scores[0] > args.sc_th:
            for atom in clause.body:
                if atom.pred.pi_type == config.pi_type['bk']:
                    unclear_pred = atom.pred
                    atom_terms = atom.terms
                    if unclear_pred.name in bk.pred_pred_mapping.keys():
                        new_pred = generate_explain_pred(args, lang, atom_terms, unclear_pred)
                        if new_pred is not None:
                            new_atom = logic.Atom(new_pred, atom_terms)
                            clause.body.append(new_atom)
            VM = ai_interface.get_vm(args, lang)
            FC = ai_interface.get_fc(args, lang, VM, args.n_obj)
            NSFR = ai_interface.get_nsfr(args, lang, FC)
            score_all_new = get_clause_score(NSFR, args, ["kp"])
            scores_new = get_clause_3score(score_all_new[:, :, index_pos], score_all_new[:, :, index_neg],
                                           args, 1)
            increased_score = scores_new - scores

        explained_clause.append([clause, scores])
        log_utils.add_lines(f"(clause) {clause} {scores}", args.log_file)
        log_utils.add_lines(f"(score increasing): {increased_score}", args.log_file)
    return explained_clause


def cluster_invention(args, lang, clauses, e):
    found_ns = False

    clu_lists = search_independent_clauses_parallel(args, lang, clauses, e)
    new_preds_with_scores = generate_new_predicate(args, lang, clu_lists, pi_type=config.pi_type["clu"])
    new_preds_with_scores = sorted(new_preds_with_scores, key=lambda x: x[1][0], reverse=True)
    new_preds_with_scores = new_preds_with_scores[:args.pi_top]
    lang.invented_preds_with_scores += new_preds_with_scores

    if args.show_process:
        log_utils.add_lines(f"new PI: {len(new_preds_with_scores)}", args.log_file)
        for new_c, new_c_score in new_preds_with_scores:
            log_utils.add_lines(f"{new_c} {new_c_score.reshape(3)}", args.log_file)

    args.is_done = found_ns
    return new_preds_with_scores


def generate_new_clauses_str_list(args, new_predicates):
    pi_str_lists = []
    kp_str_lists = []
    for [new_predicate, p_score] in new_predicates:
        single_pi_str_list = []
        # head_args = "(O1,O2)" if new_predicate.arity == 2 else "(X)"
        kp_clause = f"{args.label_name}(X):-"
        head_args = "("

        for arg in new_predicate.args:
            head_args += arg + ","
        head_args = head_args[:-1]
        head_args += ")"
        kp_clause += f"{new_predicate.name}{head_args}."
        kp_str_lists.append(kp_clause)

        head = new_predicate.name + head_args + ":-"
        for body in new_predicate.body:
            body_str = ""
            for atom_index in range(len(body)):
                atom_str = str(body[atom_index])
                end_str = "." if atom_index == len(body) - 1 else ","
                body_str += atom_str + end_str
            new_clause = head + body_str
            single_pi_str_list.append(new_clause)
        pi_str_lists.append([single_pi_str_list, p_score])

    return pi_str_lists, kp_str_lists


def extract_kp_pi(new_lang, all_pi_clauses, args):
    new_all_pi_clausese = []
    for pi_c in all_pi_clauses:
        pi_c_head_name = pi_c.head.pred.name
        new_all_pi_clausese.append(pi_c)
    return new_all_pi_clausese


def search_independent_clauses_parallel(args, lang, clauses, e):
    patterns = logic_utils.get_independent_clusters(args, lang, clauses)
    # trivial: contain multiple semantic identity bodies
    patterns = logic_utils.check_trivial_clusters(patterns)

    # TODO: parallel programming
    index_neg = config.score_example_index["neg"]
    index_pos = config.score_example_index["pos"]

    # evaluate each new patterns
    clu_all = []
    for cc_i, pattern in enumerate(patterns):
        score_neg = torch.zeros((pattern[0][2].shape[0], len(pattern))).to(args.device)
        score_pos = torch.zeros((pattern[0][2].shape[0], len(pattern))).to(args.device)
        # score_max = torch.zeros(size=(score_neg.shape[0], score_neg.shape[1], 2)).to(args.device)

        for f_i, [c_i, c, c_score] in enumerate(pattern):
            score_neg[:, f_i] = c_score[:, index_neg]
            score_pos[:, f_i] = c_score[:, index_pos]

        # in each cluster, choose score of highest scoring clause as valid score
        score_neg = score_neg.max(dim=1, keepdims=True)[0]
        score_pos = score_pos.max(dim=1, keepdims=True)[0]

        # score_max[:, :, index_pos] = score_pos[:, :, 0]
        # score_max[:, :, index_neg] = score_neg[:, :, 0]
        score_pos = score_pos.permute(1, 0)
        score_neg = score_neg.permute(1, 0)
        score_all = get_clause_3score(score_pos, score_neg, args, len(pattern[0][1].body) - e)
        clu_all.append([pattern, score_all])

    index_suff = config.score_type_index['suff']
    index_ness = config.score_type_index['ness']
    index_sn = config.score_type_index['sn']

    clu_suff = [clu for clu in clu_all if clu[1][index_suff] > args.inv_sc_th and clu[1][index_ness] > args.inv_nc_th]
    # clu_ness = [clu for clu in clu_all if clu[1][index_ness] > args.nc_th and clu[1][index_sn] > args.sn_min_th]
    # clu_sn = [clu for clu in clu_all if clu[1][index_sn] > args.sn_th]
    clu_classified = sorted(clu_suff, key=lambda x: x[1][0], reverse=True)
    clu_lists_sorted = sorted(clu_all, key=lambda x: x[1][index_ness], reverse=True)
    return clu_classified


def ilp_train(args, lang):
    # for neural_pred in args.neural_preds:
    reset_args(args)
    init_clauses, e = reset_lang(lang, args, args.neural_preds, full_bk=True)
    while args.iteration < args.max_step and not args.is_done:
        # update system
        VM = ai_interface.get_vm(args, lang)
        FC = ai_interface.get_fc(args, lang, VM, e)
        clauses, FC = ilp_search(args, lang, init_clauses, FC)  # searching for high sufficiency clauses
        if args.is_done:
            break
        if args.with_pi and len(lang.all_clauses) > 0:
            ilp_pi(args, lang, clauses, e)
            # update predicate list
            lang.preds = lang.append_new_predicate(lang.preds, lang.invented_preds)
            # update language
            lang.mode_declarations = lang_utils.get_mode_declarations(e, lang)
        args.iteration += 1
    # save the promising predicates
    keep_best_preds(args, lang)


def ilp_train_explain(args, lang, level):
    for neural_pred in args.neural_preds:
        reset_args(args)
        init_clause, e = reset_lang(lang, args, level, neural_pred, full_bk=False)
        while args.iteration < args.max_step and not args.is_done:
            # update system
            VM = ai_interface.get_vm(args, lang)
            FC = ai_interface.get_fc(args, lang, VM, e)
            clauses = ilp_search(args, lang, init_clause, FC, level)
            if args.with_explain:
                explain_scenes(args, lang, clauses)
            if args.with_pi:
                ilp_pi(args, lang, clauses, e)
            args.iteration += 1
        # save the promising predicates
        keep_best_preds(args, lang)
        if args.found_ns:
            break


def train_nsfr(args, rtpt, lang, clauses):
    VM = ai_interface.get_vm(args, lang)
    FC = ai_interface.get_fc(args, lang, VM, args.group_e)
    nsfr = ai_interface.get_nsfr(args, lang, FC, clauses, train=True)

    optimizer = torch.optim.RMSprop(nsfr.get_params(), lr=args.lr)
    bce = torch.nn.BCELoss()
    loss_list = []
    stopping_threshold = 1e-4
    test_acc_list = np.zeros(shape=(1, args.epochs))
    # prepare perception result
    train_pos = torch.tensor(args.train_group_pos)
    train_neg = torch.tensor(args.train_group_neg)
    test_pos = args.test_group_pos
    test_neg = args.test_group_neg
    val_pos = args.val_group_pos
    val_neg = args.val_group_neg
    train_pred = torch.cat((train_pos, train_neg), dim=0)
    train_label = torch.zeros(len(train_pred)).to(args.device)
    train_label[:len(train_pos)] = 1.0

    for epoch in range(args.epochs):

        # infer and predict the target probability
        loss_i = 0
        train_size = train_pred.shape[0]
        bz = args.batch_size_train
        for i in range(int(train_size / args.batch_size_train)):
            x_data = train_pred[i * bz:(i + 1) * bz]
            y_label = train_label[i * bz:(i + 1) * bz]
            V_T = nsfr(x_data).unsqueeze(0)

            predicted = nsfr_utils.get_prob(V_T, nsfr, args)
            predicted = predicted.squeeze(2)
            predicted = predicted.squeeze(0)
            loss = bce(predicted, y_label)
            loss_i += loss.item()
            loss.backward()
            optimizer.step()
        loss_i = loss_i / (i + 1)
        loss_list.append(loss_i)
        rtpt.step(subtitle=f"loss={loss_i:2.2f}")
        # writer.add_scalar("metric/train_loss", loss_i, global_step=epoch)
        log_utils.add_lines(f"(epoch {epoch}/{args.epochs - 1}) loss: {loss_i}", args.log_file)

        if epoch > 5 and loss_list[epoch - 1] - loss_list[epoch] < stopping_threshold:
            break

        if epoch % 20 == 0:
            nsfr.print_program()
            log_utils.add_lines("Predicting on validation data set...", args.log_file)

            acc_val, rec_val, th_val = se.run_ilp_predict(args, nsfr, th=0.33, split='val')
            log_utils.add_lines(f"acc_val:{acc_val} ", args.log_file)
            log_utils.add_lines("Predi$\alpha$ILPcting on training data set...", args.log_file)

            acc, rec, th = se.run_ilp_predict(args, nsfr, th=th_val, split='train')
            log_utils.add_lines(f"acc_train: {acc}", args.log_file)
            log_utils.add_lines(f"Predicting on test data set...", args.log_file)

            acc, rec, th = se.run_ilp_predict(args, nsfr, th=th_val, split='train')
            log_utils.add_lines(f"acc_test: {acc}", args.log_file)

    final_evaluation(nsfr, args)
    return nsfr


def final_evaluation(NSFR, args):
    # validation split
    log_utils.add_lines(f"Predicting on validation data set...", args.log_file)
    acc_val, rec_val, th_val = ilp_predict(NSFR, args, 0.33, split="val")
    # training split
    log_utils.add_lines(f"Predicting on training data set...", args.log_file)
    acc, rec, th = ilp_predict(NSFR, args, th_val, "train")
    # test split
    log_utils.add_lines(f"Predicting on test data set...", args.log_file)
    acc_test, rec_test, th_test = ilp_predict(NSFR, args, th_val, "test")

    log_utils.add_lines(f"training acc: {acc}, threshold: {th}, recall: {rec}", args.log_file)
    log_utils.add_lines(f"val acc: {acc_val}, threshold: {th_val}, recall: {rec_val}", args.log_file)
    log_utils.add_lines(f"test acc: {acc_test}, threshold: {th_test}, recall: {rec_test}", args.log_file)
