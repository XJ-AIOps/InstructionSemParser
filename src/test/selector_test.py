import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from langchain import HuggingFacePipeline
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.llms.chatglm3 import ChatGLM3
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import faiss
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils import grace_trans

import pandas as pd
import json
import re

# tokenizer = AutoTokenizer.from_pretrained("/home/lifeimo/.cache/modelscope/hub/qwen/", trust_remote_code=True, truncation=True)

# model = AutoModelForCausalLM.from_pretrained("/home/lifeimo/.cache/modelscope/hub/qwen/", device_map="auto", trust_remote_code=True).eval()

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_length=4096,
#     top_p=1,
#     repetition_penalty=1.15
# )
# llm = HuggingFacePipeline(pipeline=pipe)

def extract_replaced_content(A, B):
    pattern = re.escape(B).replace(r'\<\*\>', '<*>')
    
    match = re.match(pattern, A)
    
    if match:
        return match.groups()
    else:
        return None

endpoint_url = "http://219.245.186.43:5006/v1/chat/completions"

examples = []

ds_name = 'Android'
with open(f'./dataset/{ds_name}/4shot/1.json', 'r') as f:
    ex = json.load(f)
    # print(ex)
    for js in ex['dic']:
        t = {}
        t['input'] = js['text']
        t['output'] = js['label'].replace('{', '{{').replace('}', '}}')
        examples.append(t)

output_set = {
    d['output'] for d in examples
}

llm = ChatGLM3(
    endpoint_url=endpoint_url,
    max_tokens=80000,
    # prefix_messages=messages,
    top_p=0.9,
)

def similarity_selected_fewshot_prompt():
    model_name = "bge-large-zh"
    model_kwargs= {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
 
    example_prompt = PromptTemplate( 
        template = "input: {input}\noutput: {output}",
        input_variables=["input", "output"]
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        embedding_model,
        faiss.FAISS,
        k=2,
    )

    similar_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="You are a log parser that converts the following log into a template format, replacing variable parts with a wildcard <*> as output.\n \
For example:",
        suffix="Now, give the template of the input. Just give things after 'output:'\ninput: [[{log}]]\noutput:",
        input_variables=["log"],
    )

    return similar_prompt

# class CustomExampleSelector(SemanticSimilarityExampleSelector):
#     def select_examples(self, input_variables: Dict[str, Any]) -> List[Example]:
#         all_examples = super().select_examples(input_variables)
        # 排除最相似的一个示例，即选择索引为1到4的示

prompt = similarity_selected_fewshot_prompt()
# print(prompt)
llm_chain = LLMChain(prompt=prompt, llm=llm)

def llm_parser(input):
    # print('The prompt is:\n' + prompt.format(log=input))
    # output = llm_chain(inputs={"log": input})
    # output = llm_chain.invoke(input)
    # # add_template(input, output['text'])
    # template = output['text']
    # template = output['text']
    template = prompt.format(log=input)
    # ans = semantic_parser(input, template)
    # replaced_content = extract_replaced_content(input, template)
    # replaced_content = extract_replaced_content('input a', 'input <*>')
    # print(input)
    # print(template)
    # print(replaced_content)
    # return ans
    return template
    # return replaced_content

def semantic_parser(input, output1):
#     prompt2 = "For every wildcard <*>, give the meaning and the type of it with the format of <*meaning:type*>, \
# and combine them into a new template as output.\nMaybe you need some knowledge of specialized areas of computer science to finish this task.\n \
# For example:\ninput:\n[[setSystemUiVisibility vis=0 mask=1 oldVal=40000500 newVal=40000500 diff=0 fullscreenStackVis=0 dockedStackVis=0, fullscreenStackBounds=Rect(0, 0 - 0, 0), dockedStackBounds=Rect(0, 0 - 0, 0)]]\n \
# [[setSystemUiVisibility vis=<*> mask=<*> oldVal=<*> newVal=<*> diff=<*> fullscreenStackVis=<*> dockedStackVis=<*>, fullscreenStackBounds=Rect(<*>), dockedStackBounds=Rect(<*>)]]\n \
# output:\n[[setSystemUiVisibility vis=<*vis:value*> mask=<*mask:value*> oldVal=<*old:value*> newVal=<*new:value*> diff=<*diff:value*> fullscreenStackVis=<*fullscreenStack:value*> dockedStackVis=<*dockedStack:value*>, fullscreenStackBounds=Rect(<*fullscreenStackBounds:triple*>), dockedStackBounds=Rect(<*dockedStackBounds:triple*>)]]\n \
# Now, give the input. Just give things after 'output:'\ninput:\n{input1}\n{input2}\noutput:"
#     prompt2 = "For every <*> in input2, give the meaning and the type of it with the format of <*meaning:type*>, \
# and combine them into a new template as output.\ninput:\n{input1}\n{input2}\noutput:"
    prompt2 = "给定字符串input1，将input1中的一些子串替换为<*>得到input2，请给出每个被替换掉的部分的英语含义解释，每个<*>仅输出一个概括性的单词，不需要额外输出。\ninput:\n{input1}\n{input2}\noutput:"
    p2 = PromptTemplate(template=prompt2, input_variables=['input1', 'input2'])
    llm_chain2 = LLMChain(prompt=p2, llm=llm)
    input_dict = {
        'input1': input,
        'input2': output1
    }
    output = llm_chain2.invoke(input_dict)
    # print(p2.format(input1=input, input2=output1))
    # print(output['text'])
    return output['text']

def add_template(input, output):
    d = {
        "input": input,
        "output": output
    }
    examples.append(d)
    # print("New template added!")
    if output not in output_set:
        output_set.add(output)
        

if __name__ == "__main__":
    
    
    logs = pd.read_csv(f"./dataset/{ds_name}/{ds_name}_2k.log_structured_corrected.csv")
    rawlist = logs.Content.tolist()

    pbar = tqdm(rawlist)
    with open("result_20240701.txt", "w") as f:
        for log in pbar:
            result = llm_parser(log)
            f.write(result+'\n')
            tqdm.write(result)      #进度条专用输出
    f.close()