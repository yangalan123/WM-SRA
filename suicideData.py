import os, pickle
import random
from nltk import sent_tokenize, word_tokenize
import numpy as np
import re
import gensim
from config import Args


PREPROCESSING_FLAG = True
REMOVE_STOPWORD_FLAG = False


def remove_stopwords(words):
    my_stop_words = set(["feel", "like", "think", "want", "know", "day", "week",
                         "year", "hour", "month", "minute", "start", "come", "look", "time",
                         "try", "continue", "read", "night", "thing", "something",  # "need", "meet",
                         "probably", "place", "fill", "any", "sinc", "possible",
                         "anything", "tell", "because", "way", "really", "very", "please",
                         "anyone", "before", "only", "every", "right", "felt", "amp",
                         "today", "little", "yesterday", "ve", "hey", "else", "why",
                         "gonna", "guna", "tomorrow", "anybody",
                         ])

    Pronouns = set(["i", 'he', 'she', 'they', 'we', 'you', 'him', 'her', 'them', 'our', 'mine', "me",
                    'your', 'my', 'his', 'their', "ours", 'us', 'it', 'its', 'hers', 'theirs'])

    new_words = []
    for word in words:
        if word in Pronouns:
            new_words.append(word)
        elif word in gensim.parsing.preprocessing.STOPWORDS or word in my_stop_words:
            continue
        else:
            new_words.append(word)
    return new_words


# borrowed from ICWSM code: https://github.com/t-davidson/hate-speech-and-offensive-language
def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """

    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    reddit_pattern = r'r/[0-9a-zA-Z]+'
    _person_pattern = r'_[a-z]+_'

    if text_string != text_string:
        return " "

    parsed_text = text_string.lower()

    # parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    # parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(reddit_pattern, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub(_person_pattern, '', parsed_text)

    parsed_text = re.sub(space_pattern, ' ', parsed_text)
    # remove numbers and some punctuations
    parsed_text = re.sub(r'[\[\](){}$%^&"„”:#]+', '', parsed_text)
    parsed_text = re.sub(r"'", '', parsed_text)
    parsed_text = re.sub(r'[0-9]+', '', parsed_text)

    # add space before and after punctuations
    parsed_text = re.sub(r'[\.]+', ' . ', parsed_text)
    parsed_text = re.sub(r'[,]+', ' , ', parsed_text)
    parsed_text = re.sub(r'[\?]+', ' ? ', parsed_text)
    parsed_text = re.sub(r'[!]+', ' ! ', parsed_text)

    words = parsed_text.split()

    new_words = []
    for word in words:
        if len(word) > 0:
            new_words.append(word)
    if REMOVE_STOPWORD_FLAG:
        new_words = remove_stopwords(new_words)

    parsed_text = " ".join(new_words)
    return parsed_text


def concat_sentences(sents, arg: Args):
    # sents: list of str
    # concat sentences together to:
    # 1) save the GPU memory; 2) perserve more coherence (inter-sentence dependency)
    new_buf = []
    words = [word_tokenize(x) for x in sents]
    word_nums = [len(x) for x in words]
    pointer = 0
    word_count = 0
    _buf = []
    while pointer < len(sents):
        if word_nums[pointer] <= arg.MAX_LEN:
            if word_count + word_nums[pointer] <= arg.MAX_LEN:
                _buf.append(sents[pointer])
                word_count += word_nums[pointer]
            else:
                new_buf.append("".join(_buf))
                _buf.clear()
                _buf.append(sents[pointer])
                word_count = word_nums[pointer]
        else:
            if len(_buf) > 0:
                # assert len(_buf) > 0
                new_buf.append("".join(_buf))
                _buf.clear()
                _buf.append(sents[pointer])
                word_count = 0
            else:
                assert word_count == 0
                # assert len(_buf) == 0
            new_buf.append(sents[pointer])


        pointer += 1
    return new_buf

def get_data_suicide(arg: Args, cmd_args, logger, valid=0.2, grouping=True):
    def preprocessing(_data, user_id):
        input_buf = []
        for x in _data[user_id]:
            input_buf.append(x["post_title"])
            if grouping:
                input_buf.extend(concat_sentences(sent_tokenize(x["post_body"]), arg))
            else:
                input_buf.append(x["post_body"])

        # user_input = [preprocess(x["post_title"] + x["post_body"]) for x in input_buf]
        user_input = [preprocess(x) for x in input_buf]
        return user_input

    assert valid>=0 and valid <= 1, "valid proportion have to be in [0, 1]"
    root_dir = arg.data_root_dir
    with open(os.path.join(root_dir, "train.pkl"), "rb") as f_in:
        origin_train_data = pickle.load(f_in)
    valid_filename = arg.valid_filename
    train_filename = arg.processed_train_filename
    try:
        with open(os.path.join(root_dir, valid_filename), "rb") as f_in:
            valid_data = pickle.load(f_in)
        with open(os.path.join(root_dir, train_filename), "rb") as f_in:
            train_data = pickle.load(f_in)
    except:
        logger.info("Try loading valid data, failed, try to create valid set on the fly using {} of train set".format(valid))
        userIDs = list(origin_train_data.keys())
        random.shuffle(userIDs)
        train_num = int(len(userIDs) * (1-valid))
        trainIDs = userIDs[: train_num]
        validIDs = userIDs[train_num: ]
        _train_data = {}
        for x in trainIDs:
            _train_data[x] = origin_train_data[x].copy()
        _valid_data = {}
        for x in validIDs:
            _valid_data[x] = origin_train_data[x].copy()
        pickle.dump(_train_data, open(os.path.join(root_dir, train_filename), "wb"))
        pickle.dump(_valid_data, open(os.path.join(root_dir, valid_filename), "wb"))
        train_data = _train_data
        valid_data = _valid_data

    if not arg.inference_mode:
        with open(os.path.join(root_dir, "test.pkl"), "rb") as f_in:
            test_data = pickle.load(f_in)
    else:
        with open(os.path.join(root_dir, arg.predict_test_file), "rb") as f_in:
            test_data = pickle.load(f_in)

    source_data = [train_data, valid_data, test_data]
    types = ["train", "valid", "test"]
    if arg.USE_PL:
        with open(os.path.join(arg.PL_dir, arg.PL_filename), "rb") as f_in:
            PL_data = pickle.load(f_in)
        source_data.append(PL_data)
        types.append("PL")

    if arg.use_aux_train_data:
        aux_data = dict()
        for _filename in arg.aux_train_filenames:
            with open(os.path.join(root_dir,_filename), "rb") as f_in:
                _aux_data = pickle.load(f_in)
                for user_id in _aux_data:
                    user_input = preprocessing(_aux_data, user_id)
                    if user_id not in aux_data:
                        aux_data[user_id] = []
                    aux_data[user_id].append(user_input)
    output_data = []
    for _data, _type in zip(source_data, types):
        _buf = []
        for user_id in _data:
            user_input = preprocessing(_data, user_id)
            user_labels = [arg.CLASS_NAMES[x["author_label"]] for x in _data[user_id]]
            new_instances = {"user_id": user_id, "user_input": user_input,
                             "user_labels": user_labels}
            if _type == "train" and arg.use_aux_train_data:
                new_instances["aux"] = aux_data[user_id]
            _buf.append(new_instances)
        output_data.append(_buf)
    if arg.USE_PL:
        assert len(output_data) >= 4, "no PL data? {}".format(len(output_data))
        PL_data = output_data[3]
        print("load {} PL data".format(len(PL_data)))
        arg.PL_data_index = len(output_data[0])
        output_data[0].extend(PL_data)

    # PL data has been merged into training dataset, no need to include once again
    return output_data[:3]



def change_label_predict_list_stype(xs:list, type="flagged"):
    assert type in ["flagged", "urgent"]
    res = []
    if type == "flagged":
        for x in xs:
            if x >= 1:
                res.append(1)
            else:
                res.append(0)
    else:
        for x in xs:
            if x >= 2:
                res.append(1)
            else:
                res.append(0)
    return np.array(res)

