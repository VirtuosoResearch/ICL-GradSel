import json

data_file = "./amps_algebra.json"

with open(data_file,"r") as file:
    data = json.load(file)
    instance = data["Instances"]


qu1=qu2=qu3=an1=an2=an3=""
cnt=0
out = []
for item in instance:
    cnt+=1
    if cnt%3==1:
        qu1=item["Input"]
        an1=item["Output Answer"][0]
    if cnt%3==2:
        qu2=item["Input"]
        an2=item["Output Answer"][0]
    if cnt%3==0:
        qu3=item["Input"]
        an3=item["Output Answer"][0]

        neg = {"input":qu1, "output":an2}
        pos = {"input":qu2, "output":an2}
        out.append({"index":cnt//3, "Neg":neg, "Pos":pos, "Question":qu3, "Answer":an3})
    
    # print(out)
    # if cnt>=3: break

with open("./lila_contrastive.json","w") as file:
    file.write(json.dumps(out,indent=4))