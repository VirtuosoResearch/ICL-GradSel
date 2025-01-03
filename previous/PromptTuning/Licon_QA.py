import torch
import json
from torch.utils.data import DataLoader, random_split, Dataset

class Licon_QA(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, "r") as file:
            self.dataset = json.load(file)
        self.length = len(self.dataset)
    def __getitem__(self, index):
        instuction = "Help me answer the questions below. You have a negative sample and a positive sample for reference:"
        item = self.dataset[index]
        neg = "Negative: \n\tinput:"+item["Neg"]["input"]+"\n\toutput:"+item["Neg"]["output"]
        pos = "Positive: \n\tinput:"+item["Pos"]["input"]+"\n\toutput:"+item["Pos"]["output"]
        que = "Question: \n\t"+item["Question"]
        ans = "Answer: \n\t" + item["Answer"]
        prompt = instuction+"\n" + neg+"\n" + pos+"\n" + que
        return {"Question": prompt, "Answer": ans}
    def __len__(self):
        return self.length

licon = Licon_QA("./lila_contrastive.json")

def get_dataloader(batch_size: int, train_len: int, test_len: int):
    train_data, test_data = random_split(licon, [train_len, test_len])

    train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train,test

if __name__=="__main__":
    train_len = int(licon.length*0.8)
    test_len = licon.length-train_len
    traindata, testdata = get_dataloader(4,train_len, test_len)
    for item in traindata:
        print(item)
        break