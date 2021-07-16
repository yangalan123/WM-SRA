from torch import nn
from transformers import AutoModel
import torch
class BertClassifier(nn.Module):
    def __init__(self, num_label, arg):
        super(BertClassifier, self).__init__()
        self.base_model = AutoModel.from_pretrained(arg.BASE_MODEL).eval()
        self.hidden_size = self.base_model.config.hidden_size
        self.output = nn.Linear(self.hidden_size, num_label)
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(0.1)
        self.transfer = nn.Sigmoid()
        self.arg = arg

    def forward(self, inputs, **kwargs):
        hiddens = []
        for i in range(0, len(inputs["input_ids"]), self.arg.bert_batch_size):
            # given our GPU memory is so limited
            hidden = self.base_model(inputs["input_ids"][i: i + self.arg.bert_batch_size],
                                     )[0][:, 0, :]
            hiddens.append(hidden)
        hiddens = torch.cat(hiddens, dim=0)
        hiddens = hiddens.mean(dim=0, keepdim=True)
        logits = self.output(hiddens)
        return logits

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path + "/{}_model.pt".format(self.arg.instance_name))

    def from_pretrained(self, path):
        self.load_state_dict(torch.load(path + "/{}_model.pt".format(self.arg.instance_name)), strict=False)

def prepare_for_model(tokenizer, batch_inputs, batch_labels, max_len_in_batch):
    FLAG_TYPE_IDS = True
    for i in range(len(batch_inputs)):
        res = tokenizer.prepare_for_model(ids=batch_inputs[i], max_length=max_len_in_batch, pad_to_max_length=True, truncation=True)
        batch_inputs[i] = dict()
        batch_inputs[i]["input_ids"] = torch.LongTensor(res["input_ids"])
        batch_inputs[i]["attention_mask"] = torch.LongTensor(res["attention_mask"])
        if "token_type_ids" in res:
            batch_inputs[i]["token_type_ids"] = torch.LongTensor(res["token_type_ids"])
        else:
            FLAG_TYPE_IDS = False
    batch_labels = torch.LongTensor(batch_labels).cuda()
    new_batch_inputs = dict()
    new_batch_inputs["input_ids"] = torch.stack([x["input_ids"] for x in batch_inputs]).cuda()
    new_batch_inputs["attention_mask"] = torch.stack([x["attention_mask"] for x in batch_inputs]).cuda()
    if FLAG_TYPE_IDS:
        new_batch_inputs["token_type_ids"] = torch.stack([x["token_type_ids"] for x in batch_inputs]).cuda()
    return new_batch_inputs, batch_labels
