from collections import defaultdict

from utils.preprocessing import PAD_ID
from utils.training_utils import batch_generator

import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def evaluate(in_model, in_dataset, runs_number=1):
    accuracies, loss_dict = [], defaultdict(lambda: [])
    for run in range(runs_number):
        accuracy_i, loss_dict_i = evaluate_single_run(in_model, in_dataset)
        accuracies.append(accuracy_i)
        for key, value in loss_dict_i.items():
            loss_dict[key].append(value)
    return np.mean(accuracies), {key: np.mean(value) for key, value in loss_dict.items()}


def evaluate_single_run(in_model, in_dataset):
    loss_dict = defaultdict(lambda: [])
    predictions = []
    batch_gen = batch_generator(in_dataset, 64)
    for batch in batch_gen:
        batch_predictions, batch_loss_dict = in_model.forward(*batch)
        for key, value in batch_loss_dict.items():
            loss_dict[key].append(value)
        predictions += batch_predictions.flatten().tolist()
    y = in_dataset[-2]
    return (sum([gold_i == pred_i and gold_i != PAD_ID for gold_i, pred_i in zip(y.flatten(), predictions)]) / np.sum(list(map(lambda x: int(x != 0), y.flatten()))),
            {key: np.mean(value) for key, value in loss_dict.items()})


def evaluate_advanced(in_model, in_dataset, in_action_templates, in_fallback_utterance, post_ood_turns=None, ignore_ood_accuracy=False, runs_number=1):
    result_dict = defaultdict(lambda: [])

    for _ in range(runs_number):
        result_dict_i = evaluate_advanced_single(in_model, in_dataset, in_action_templates, in_fallback_utterance, post_ood_turns=post_ood_turns, ignore_ood_accuracy=ignore_ood_accuracy)
    for key, value in result_dict_i.items():
        result_dict[key].append(value)
    return {key: np.mean(value) if key == 'loss' else np.sum(value) for key, value in result_dict.items()}


def evaluate_advanced_single(in_model, in_dataset, in_action_templates, in_fallback_utterance, post_ood_turns=None, ignore_ood_accuracy=False):
    if in_fallback_utterance in in_action_templates:
        fallback_action = in_action_templates.index(in_fallback_utterance)
    else:
        fallback_action = len(in_action_templates) - 1

    losses, predictions = [], []
    batch_gen = batch_generator(in_dataset, 64)
    for batch in batch_gen:
        batch_predictions, batch_loss_dict = in_model.forward(*batch)
        losses.append(batch_loss_dict['loss'])
        predictions += batch_predictions.tolist()

    y = in_dataset[-2]
    y_true_dialog, y_pred_dialog = [], []
    y_true_all, y_pred_all = [], []
    y_true_binary, y_pred_binary = [], []
    for y_true_i, y_pred_i in zip(y, predictions):
        y_true_dialog_i, y_pred_dialog_i = [], []
        for y_true_i_j, y_pred_i_j in zip(y_true_i, y_pred_i):
            if y_true_i_j != PAD_ID:
                y_true_dialog_i.append(y_true_i_j)
                y_pred_dialog_i.append(y_pred_i_j)
        y_true_dialog.append(y_true_dialog_i)
        y_pred_dialog.append(y_pred_dialog_i)
        y_true_all += y_true_dialog_i
        y_pred_all += y_pred_dialog_i
    
    y_true_binary = [int(action == fallback_action) for action in y_true_all]
    y_pred_binary = [int(action == fallback_action) for action in y_pred_all]

    ood_f1 = f1_score(y_true_binary, y_pred_binary, average='binary')
    acc = accuracy_score(y_true_all, y_pred_all)

    total_turns = 0
    correct_turns = 0
    correct_continuous_turns = 0
    total_ood_turns = 0
    correct_ood_turns = 0
    for y_true, y_pred in zip(y_true_dialog, y_pred_dialog):
        error_occurred = False
        for y_i_true, y_i_pred in zip(y_true, y_pred):
            error_occurred = error_occurred or y_i_true != y_i_pred
            current_turn_counts = not (y_i_true == fallback_action and ignore_ood_accuracy)
            total_turns += int(current_turn_counts)
            correct_turns += int(y_i_true == y_i_pred and current_turn_counts)
            correct_continuous_turns += int(not error_occurred)
            if y_i_true == fallback_action:
                total_ood_turns += 1
                correct_ood_turns += int(y_i_true == y_i_pred)
    post_ood_correct, post_ood_total = 0, 0
    if post_ood_turns is not None:
        non_pad_counter = 0
        post_ood_true, post_ood_pred = [], []
        for y_true_i, y_pred_i in zip(np.array(y).flatten(), np.array(predictions).flatten()):
            if y_true_i == PAD_ID:
                continue
            if non_pad_counter in post_ood_turns:
                post_ood_true.append(y_true_i)
                post_ood_pred.append(y_pred_i)
            non_pad_counter += 1
        post_ood_correct = sum([int(true_i == pred_i) for true_i, pred_i in zip(post_ood_true, post_ood_pred)])
        post_ood_total = len(post_ood_turns)

    assert abs(acc - correct_turns / total_turns) < 1e-7

    return {'avg_loss': np.mean(losses),
            'correct_turns': correct_turns,
            'total_turns': total_turns,
            'total_ood_turns': total_ood_turns,
            'correct_ood_turns': correct_ood_turns,
            'total_post_ood_turns': post_ood_total,
            'correct_post_ood_turns': post_ood_correct,
            'correct_continuous_turns': correct_continuous_turns,
            'ood_f1': ood_f1}


def mark_post_ood_turns(in_dialogs):
    total_turn_counter, ind_turn_counter = 0, 0
    prev_action = None
    post_ood_turn_indices_clean, post_ood_turn_indices_noisy = set([]), set([])
    for dialog_i in in_dialogs['dialogs']:
        for turn in dialog_i['turns']:
            if 'input' not in turn:
                continue
            system_action = turn['label']
            if system_action != len(in_dialogs['actions']) - 1:
                if prev_action == len(in_dialogs['actions']) - 1:
                    post_ood_turn_indices_clean.add(ind_turn_counter)
                    post_ood_turn_indices_noisy.add(total_turn_counter)
                ind_turn_counter += 1
            total_turn_counter += 1
            prev_action = system_action
    return post_ood_turn_indices_clean, post_ood_turn_indices_noisy
