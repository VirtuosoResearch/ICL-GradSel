import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split

class Alcon_QA(Dataset):
    def __init__(self,file_path):
        super().__init__()
        with open(file_path, "r") as file:
            self.data_dict = json.load(file)
            self.length = len(self.data_dict)
    def __getitem__(self, index):
        item = self.data_dict[index]

        instruction = "Here is a negative example and a positive example, help me answer the question below"

        Neg = "Negative example: \n\t"+item["Neg"]["input"]+" \n\tAnswer: "+item["Neg"]["output"]

        Pos = "Positive example: \n\t"+item["Pos"]["input"]+" \n\tAnswer: "+item["Pos"]["output"]

        Question = instruction+"\n"+Neg+"\n"+Pos+"\nQuestion: "+item["Question"]
        Answer = item["Answer"]

        return {"Question":Question, "Answer":Answer}
    def __len__(self):
        return self.length

data_path = "./alpaca_contrastive.json"
alcon_data = Alcon_QA(data_path)

train_size = int(0.8*len(alcon_data))
test_size = len(alcon_data)-train_size

train_data, test_data = random_split(alcon_data, [train_size,test_size])

print(len(train_data),len(test_data))
