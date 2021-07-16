import pickle
from tqdm import tqdm
import os
import csv
import sys
maxInt = sys.maxsize

# thanks to stackoverflow:
# https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)
tasks = ["A", "B", "C"]
types = ["train", "test"]
crowd_files = [os.path.join(x, "crowd_{}.csv".format(x)) for x in types]
user2label = dict()
for _crowd_file in crowd_files:
    with open(_crowd_file, "r", encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        for line in reader:
            try:
                user_id, raw_label = line["user_id"], line["raw_label"]
            except KeyError:
                user_id, raw_label = line["user_id"], line["label"]
            if len(raw_label) == 0 or raw_label not in ["a", "b", "c", "d"]:
                # control user, excluded
                # print(user_id, _crowd_file)
                continue
            user2label[user_id] = raw_label

shared_task_files = [os.path.join("train", "shared_task_posts.csv"),
                     os.path.join("test", "shared_task_posts_test.csv")]

for _task in tasks:

    task_files = [os.path.join(x, "task_{}_{}.posts.csv".format(_task, x)) for x in types]


    task_set = {"train": set(), "test": set()}
    for _type, _task_file in zip(types, task_files):
        with open(_task_file, "r", encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            for line in reader:
                post_id, user_id = line["post_id"], line["user_id"]
                task_set[_type].add((post_id, user_id))

    data_set = {"train": dict(), "test": dict()}
    for _type, _shared_task_file in zip(types, shared_task_files):
        with open(_shared_task_file, "r", encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            for line in tqdm(reader):
                post_id, user_id, timestamp, subreddit, post_title, post_body = \
                    line["post_id"], line["user_id"], line["timestamp"], line["subreddit"], \
                    line["post_title"], line["post_body"]
                key = (post_id, user_id)
                if key in task_set[_type]:
                    if user_id not in user2label:
                        # print(_task, _type, _shared_task_file, user_id)
                        continue
                    if user_id not in data_set[_type]:
                        data_set[_type][user_id] = []
                    data_set[_type][user_id].append({
                        "timestamp": timestamp,
                        "subreddit": subreddit,
                        "post_title": post_title,
                        "post_body": post_body,
                        "author_label": user2label[user_id]}
                    )

    os.makedirs("tasks_{}".format(_task), exist_ok=True)
    for _type in data_set:
        for user_id in data_set[_type]:
            data_set[_type][user_id].sort(key=lambda x: x["timestamp"])
        with open(os.path.join("tasks_{}".format(_task), "{}.pkl".format(_type)), "wb") as f_out:
            pickle.dump(data_set[_type], f_out)






