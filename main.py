import Global.llm
import Global.prompts
import Global.test
from langchain_core.output_parsers import StrOutputParser
import Global.until
from langchain_core.documents  import Document  

llm=Global.llm.deepseek_by_ds("deepseek-chat")

txt=Global.test.txt_斗破苍穹1()
print(txt)
print(len(txt))

Prompt=Global.prompts.txtsliptPrompt(txt,len(txt),"人物，地点，事件")

import json
data=json.loads(Global.until.extract_json_raw(llm.invoke(Prompt).content))
print(data)


print(data["length"])
array=[]
for object in data["data"]:
    summary=object["s"]
    begin=object["b"]
    end=object["e"]
    print(summary,begin,end)
    array.append(Document(page_content=summary,metadata={"rawdata": txt[begin:end]}))
print(array)  


# Prompt=Global.prompts.txtsliptPrompt2(txt)
# print(Prompt)
# import json
# data=json.loads(Global.until.extract_json_raw(llm.invoke(Prompt).content))
# print(data)
# array=[]
# for object in data["data"]:
#     summary=object["s"]
#     begin=object["b"]
#     end=object["e"]
#     print(summary,begin,end)
#     array.append(Document(page_content=summary,metadata={"rawdata": txt[begin:end]}))
# print(array) 