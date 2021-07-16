import argparse
import json
import os
import pickle
import random
import shutil

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torch.optim import Adam
from transformers import AutoTokenizer
from torch.nn import KLDivLoss, LogSoftmax

from config import Args
from model import BertClassifier, prepare_for_model
from suicideData import get_data_suicide, change_label_predict_list_stype

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="Pleae input one of A, B, C, Asum, Bsum, Csum")
cmd_args = parser.parse_args()
assert cmd_args.task in ["A", 'B', 'C', "Asum", "Bsum", "Csum"], 'Please input correct task type!'
# Asum, Bsum, Csum:
# summarized version of task A / B / C,
# using the popular K-means based summarization: https://github.com/dmmiller612/bert-extractive-summarizer

arg = Args(cmd_args)


def change_label_subprocess(logits, cuda_labels):
    # change label on the fly (e.g., using the predicted label as the pseudo-labels) -- not useful
    # also tried random labeling at each iteration, not working
    predicted_label = torch.argmax(logits.squeeze()).item()
    original_label = cuda_labels.item()
    choice = np.random.choice(["origin", "predict", "random"], 1, p=arg.PL_label_change_dist).tolist()[0]
    if choice == "origin":
        _label = original_label
    elif choice == "predict":
        _label = predicted_label
    else:
        _label = np.random.choice([0, 1, 2, 3]).tolist()
    return torch.LongTensor([_label, ]).cuda()


def process_batch(_batch, model, mode, tokenizer, data, contrastive_learning=False, label2user=None,
                  change_label=False):
    _input, _labels = _batch
    if len(_input) > arg.max_group_size:
        _input = random.sample(_input, arg.max_group_size)
    batch_inputs = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x)) for x in _input]
    max_len_in_batch = min(max([len(x) for x in batch_inputs]), arg.MAX_LEN)
    batch_labels = _labels
    batch_inputs, batch_labels = prepare_for_model(tokenizer, batch_inputs, batch_labels, max_len_in_batch)
    # once tried deterministic annealing to encourage exploration, not working
    logits = model(
        {"input_ids": batch_inputs["input_ids"].cuda(), "attention_mask": batch_inputs["attention_mask"].cuda()})
    batch_labels = batch_labels[0].unsqueeze(dim=0)
    if change_label:
        batch_labels = change_label_subprocess(logits, batch_labels)

    loss = model.criterion(logits, batch_labels)
    # once tried contrastive learning, not working....
    if not contrastive_learning or label2user is None or len(label2user) == 0 or arg.ctl_weight == 0:
        return logits, loss
    else:
        _label = _labels[0]
        candidate_set = set()
        for _candidate_group in arg.neg_sample_dict[_label]:
            candidate_set |= set(label2user[_candidate_group])
        neg_sample_ids = random.sample(candidate_set, min(len(candidate_set), arg.neg_sample_users_num))
        all_neg_logits_at_label = []
        for user_id in neg_sample_ids:
            neg_logits, neg_loss = process_batch(data[user_id], model, mode, tokenizer, data)
            all_neg_logits_at_label.append(neg_logits[0][_label].view(1))
            del neg_loss
            del neg_logits
            torch.cuda.empty_cache()
        all_contrastive_logits = torch.cat([logits[0][_label].view(1)] + all_neg_logits_at_label, dim=0).unsqueeze(0)
        all_labels = torch.LongTensor([0]).cuda()
        contrastive_loss = model.criterion(all_contrastive_logits, all_labels)
        del batch_labels
        del batch_inputs
        return logits, loss + arg.ctl_weight * contrastive_loss


def running(model: BertClassifier, tokenizer, optimizer, data, mode="train", store=False):
    assert mode in ["train", "test"]
    if mode == "test":
        model = model.eval()
    else:
        model = model.train()
    all_loss = 0
    predictions = []
    labels = []
    label2user = dict()
    kl_loss = KLDivLoss(reduction="batchmean", log_target=True)
    log_softmax = LogSoftmax(dim=1)
    outputs = []
    for i, _batch in enumerate(data):
        # here a "batch" means a "user"
        _input, _labels = _batch["user_input"], _batch["user_labels"]
        labels.append(_labels[0])
        logits, loss = process_batch([_input, _labels], model, mode, tokenizer, data,
                                     contrastive_learning=len(label2user) > 0,
                                     label2user=label2user,
                                     change_label=(i >= arg.PL_data_index and mode == "train"
                                                   and arg.allow_PL_label_change))
        all_loss += loss.item()
        predictions.append(torch.argmax(logits, dim=1).item())
        if store:
            outputs.append({"id": _batch["user_id"], "label": _labels[0], "pred": predictions[-1]})
        if mode == "train":
            if arg.use_aux_train_data and "aux" in _batch:
                aux_datas = _batch["aux"]
                # _aux_losses = []
                for _aux_data in aux_datas:
                    _logits, _aux_loss = process_batch([_aux_data, _labels], model, mode, tokenizer, data,
                                                       contrastive_learning=len(label2user) > 0,
                                                       label2user=label2user,
                                                       change_label=(i >= arg.PL_data_index and mode == "train"
                                                                     and arg.allow_PL_label_change))
                    loss += kl_loss(log_softmax(logits), log_softmax(_logits))
            loss.backward()
            if (i + 1) % arg.BATCH_SIZE == 0 or i == len(data) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if arg.use_DA:
                    model.DA("exp")
        del logits
        del loss
    avg_loss = all_loss / len(data)
    clf_report = classification_report(labels, predictions, target_names=arg.CLASS_NAMES_clf)
    _f1_score = f1_score(labels, predictions, average="macro")
    flagged_labels = change_label_predict_list_stype(labels)
    flagged_predicts = change_label_predict_list_stype(predictions)
    urgent_labels = change_label_predict_list_stype(labels, "urgent")
    urgent_predicts = change_label_predict_list_stype(predictions, "urgent")
    _urgent_f1_score = f1_score(urgent_labels, urgent_predicts, average="macro")
    _flagged_f1_score = f1_score(flagged_labels, flagged_predicts, average="macro")
    _confusion_matrix = confusion_matrix(labels, predictions)
    if store:
        if hasattr(arg, "predict_test_file") and arg.predict_test_file:
            pickle.dump(outputs, open(os.path.join(arg.model_dir_path, "dump_{}.pkl".format(
                arg.predict_test_file
            )), "wb"))
        else:
            pickle.dump(outputs, open(os.path.join(arg.model_dir_path, "dump_{}.pkl".format(
                "test"
            )), "wb"))
    return avg_loss, clf_report, _f1_score, _urgent_f1_score, _flagged_f1_score, _confusion_matrix


if __name__ == "__main__":
    os.makedirs(arg.model_dir_path, exist_ok=True)
    codename = os.path.basename(__file__)
    for _filename in [codename, "config.py", "model.py", "suicideData.py"]:
        shutil.copyfile(_filename, os.path.join(arg.model_dir_path, _filename))
    logger.add(os.path.join(arg.model_dir_path, arg.instance_name + "_log_{time}.txt"))
    logger.info("Config: {}".format(json.dumps(vars(arg), indent=4)))
    torch.manual_seed(arg.SEED)
    logger.info("Loading Suicide Data...")
    random.seed(arg.SEED)
    train_data, valid_data, test_data = get_data_suicide(arg, cmd_args, logger)
    logger.info("Suicide Data have been loaded.")
    classifier = BertClassifier(arg.num_classes, arg).cuda()
    tokenizer = AutoTokenizer.from_pretrained(arg.BASE_MODEL)
    if arg.MODEL_TYPE in {"GPT-2"}:
        tokenizer.pad_token = tokenizer.eos_token
    optimizer = Adam(classifier.parameters(), lr=2e-5)
    best_dev_valid = 0
    for i in range(arg.EPOCHS):
        avg_train_loss, clf_train_report, _f1_train, urgent_f1_train, flagged_f1_train, cm_train = running(classifier,
                                                                                                           tokenizer,
                                                                                                           optimizer,
                                                                                                           train_data,
                                                                                                           "train")
        avg_valid_loss, clf_valid_report, _f1_valid, urgent_f1_valid, flagged_f1_valid, cm_valid = running(classifier,
                                                                                                           tokenizer,
                                                                                                           optimizer,
                                                                                                           valid_data,
                                                                                                           "test")
        logger.info("Epoch: {}".format(i + 1))
        logger.info("Training Loss: {}, Valid Loss: {}".format(avg_train_loss, avg_valid_loss))
        logger.info(
            "Train:\n macro_f1: {}, urgent_f1:{}, flagged_f1:{};".format(_f1_train, urgent_f1_train, flagged_f1_train))
        logger.info(
            "Valid:\n macro_f1: {}, urgent_f1:{}, flagged_f1:{};".format(_f1_valid, urgent_f1_valid, flagged_f1_valid))
        logger.info("Training CLF report: \n{}".format(clf_train_report))
        logger.info("Valid CLF report: \n{}".format(clf_valid_report))
        logger.info("Valid CLF Confusion Matrix: \n{}".format(cm_valid))
        if _f1_valid > best_dev_valid:
            best_dev_valid = _f1_valid
            classifier.save_pretrained(arg.model_dir_path)
            logger.info("Best updated at Epoch: {}".format(i + 1))

    if not arg.inference_mode:
        classifier.from_pretrained(arg.model_dir_path)
    else:
        classifier.from_pretrained(arg.trained_model_dir_path)
    avg_test_loss, clf_test_report, _f1_test, urgent_f1_test, flagged_f1_test, cm_test = running(classifier, tokenizer,
                                                                                                 optimizer, test_data,
                                                                                                 "test",
                                                                                                 store=arg.dump_test_output)
    logger.info("Test:\n macro_f1: {}, urgent_f1:{}, flagged_f1:{};".format(_f1_test, urgent_f1_test, flagged_f1_test))
    logger.info("Test Loss: {}, Test Report: \n {}".format(avg_test_loss, clf_test_report))
    logger.info("Test CLF Confusion Matrix: \n{}".format(cm_test))
