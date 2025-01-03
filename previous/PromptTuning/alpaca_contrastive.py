import json

file_path = "./alpaca_data_cleaned_archive.json"
with open(file_path,"r") as file:
    data = json.load(file)
print(f"len(data) : {len(data)}")

cnt = 0
qu1=qu2=qu3=""
an1=an2=an3=""
an=""
output =[] 
out = {}
for item in data:
    cnt += 1
    if cnt%3==1:
        qu1 = "instruction : " + item["instruction"] + "\n " + "input : "+item["input"]
        an1 = item["output"]
    elif cnt%3==2:
        qu2 = "instruction : " + item["instruction"] + "\n " + "input : "+item["input"]
        an2 = item["output"]
    elif cnt%3==0:
        qu3 = "instruction : " + item["instruction"] + "\n " + "input : "+item["input"]
        an3 = item["output"]

        out["index"] = cnt//3
        out["Neg"] = {"input": qu1, "output": an2}
        out["Pos"] = {"input": qu1, "output": an1}
        out["Question"] = qu3
        out["Answer"] = an3
        
        # print(out)
        output.append(out)

        out={}
        an1=an2=an3=qu1=qu2=qu3=""
    # if cnt>10: break

# print("*"*20)
print(len(output))

with open("./alpaca_contrastive.json","w") as file:
    file.write(json.dumps(output,indent=4))