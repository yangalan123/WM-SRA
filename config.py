import os


class Args:
    def __init__(self, cmd_args):
        self.data_root_dir = "./umd_reddit_suicidewatch_dataset_v2/umd_reddit_suicidewatch_dataset_v2/crowd/tasks_{}".format(
            cmd_args.task)
        self.BASE_MODEL = "bert-base-uncased"
        #self.BASE_MODEL = "roberta-base"
        #self.BASE_MODEL = "xlnet-base-cased"
        #self.BASE_MODEL = "albert-base-v2"
        # MODEL_TYPE ONLY FOR LOGGING AND SAVING PURPOSES
        # self.MODEL_TYPE = "BERT_with_att"
        # self.MODEL_TYPE = "BERT_with_PL_change"
        self.MODEL_TYPE = "BERT"
        #self.MODEL_TYPE = "Roberta"
        #self.MODEL_TYPE = "XLNET"
        self.BATCH_SIZE = 32
        self.SEED = 1111
        self.EPOCHS = 20
        self.MAX_LEN = 128
        #self.MAX_LEN = 256
        #self.MAX_LEN = 512
        self.REPORT_GAP = 1000
        self.num_head = 16
        self.bert_batch_size = self.BATCH_SIZE
        self.task_type = cmd_args.task

        # self.UPDATE_ITERATION = 32
        # self.CLASS_NAMES_clf = ["a", "b & c", "d"]
        self.CLASS_NAMES_clf = ["a", "b", "c", "d"]
        self.CLASS_NAMES = {"a": 0, "b": 1, "c": 2, "d": 3}
        # self.CLASS_NAMES = ["a", "b & c", "d"]
        # self.CLASS_NAMES = {"a": 0, "b": 1, "c": 1, "d": 2}
        self.num_classes = max([x[1] for x in self.CLASS_NAMES.items()]) + 1
        self.neg_sample_users_num = 5
        self.ctl_weight = 0
        # self.GROUP_STEP_SIZE = 40
        if self.ctl_weight > 0:
            self.instance_name = "model_neguser{}_epoch{}_bsz{}_len{}".format(self.neg_sample_users_num,
                                                                              self.EPOCHS,
                                                                              self.BATCH_SIZE,
                                                                              # self.GROUP_STEP_SIZE,
                                                                              self.MAX_LEN)
        else:
            self.instance_name = "model_epoch{}_bsz{}_len{}".format( self.EPOCHS, self.BATCH_SIZE, # self.GROUP_STEP_SIZE,
                                                                              self.MAX_LEN)
        self.dump_test_output = True
        self.inference_mode = False
        if self.inference_mode:
            self.predict_test_file = "opiate_sw_common_full.pkl"
            self.EPOCHS = 0
            self.trained_model_dir_path = ""
        # whether using pseudo-labelling or not
        self.USE_PL = True
        self.PL_dir = "./pseudo-labelling"
        self.PL_filename = "depop_PL.pkl"

        self.get_model_dir_path("SuicideModel")
        self.model_dir_path = os.path.join(self.model_dir_path, self.instance_name.lstrip("model_"))
        self.model_path = os.path.join(self.model_dir_path, "{}.pt".format(self.instance_name))
        # once tried deterministic annealing and contrastive learning, not working....
        self.use_DA = False
        self.neg_sample_dict = {
            0: [1, 2, 3],
            # 1: [2, 3],
            1: [0, 2, 3],
            # 2: [3],
            2: [0, 1, 3],
            # 3: [2]
            3: [0, 1, 2]
        }
        # whether storing valid partition for future reproducibility
        self.store_valid = True
        self.try_load_valid = True
        # self.valid_filename = "processed_valid.pkl"
        self.valid_filename = "raw_valid.pkl"
        self.max_group_size = 100
        # if self.store_valid:
        # self.load_train = True
        # else:
        # self.load_train = False
        # self.processed_train_filename = "processed_train.pkl"
        self.processed_train_filename = "raw_train.pkl"
        self.PL_data_index = 999999
        # self.allow_PL_label_change = True
        self.allow_PL_label_change = False
        # origin : predict : random
        # try to change labeling on the fly, not working...
        if self.allow_PL_label_change:
            self.PL_label_change_dist = [0.7, 0.2, 0.1]
        # whether use summarization and begin-end
        self.use_aux_train_data = False
        #self.use_aux_train_data = True
        if self.use_aux_train_data:
            #self.aux_train_filenames = ["summ_raw_train.pkl", "beged_raw_train.pkl"]
            #self.aux_train_filenames = ["beged_raw_train.pkl"]
            #self.aux_train_filenames = ["summ_raw_train.pkl"]
            #self.aux_train_filenames = ["word_sample_raw_train.pkl", "sent_sample_raw_train.pkl"]
            self.aux_train_filenames = ["word_sample_raw_train.pkl"]
            #self.aux_train_filenames = ["sent_sample_raw_train.pkl"]

    def get_model_dir_path(self, root="SuicideModel") -> str:
        self.model_dir_path = os.path.join(root, "task_{}".format(self.task_type),
        self.MODEL_TYPE, "NO_CTL_preprocess_with_no_pretrain_{}".format(self.PL_filename[:-4]))
