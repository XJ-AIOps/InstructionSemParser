# from datasets import load_dataset
# import pandas as pd


# def load_train_data(r_dir=".", dataset="Apache", shot=4):
#     dataset = load_dataset('json', data_files=f'{r_dir}/{dataset}/{4}shot/1.json')
#     examples = [(x['text'], x['label']) for x in dataset['train']]
#     return examples


# def load_test_data(r_dir=".", dataset="Apache"):
#     logs = pd.read_csv(f"{r_dir}/{dataset}/{dataset}_2k.log_structured_corrected.csv")
#     return logs.Content.tolist()

import csv

def translate(input_file_path):
    # 输入和输出文件路径
    # input_file_path = 'input.txt'
    output_file_path = 'output.csv'

    # 打开txt文件并读取内容
    with open(input_file_path, 'r', encoding='utf-8') as txt_file:
        lines = txt_file.readlines()

    # 打开csv文件准备写入
    with open(output_file_path, 'w', encoding='utf-8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # 写入csv文件的第一行
        csv_writer.writerow(['LineId', 'EventTemplate'])
        
        # 写入剩下的行
        for i, line in enumerate(lines):
            csv_writer.writerow([i, line.strip()])

    print(f"Successfully converted {input_file_path} to {output_file_path}")

def translate2():
    import re

    def replace_braces(text):
        # 定义正则表达式模式，匹配 {} 中的内容
        pattern = r'\{(.*?)\}'
        
        # 替换 {} 为 <**>，并保留 {} 中的内容
        replaced_text = re.sub(pattern, r'<*\1*>', text)
        
        return replaced_text

    # 示例文本
    text = "setSystemUiVisibility vis=<*vis*> mask={mask} oldVal={oldVal} newVal={newVal} diff={diff} fullscreenStackVis={fullscreenStackVis} dockedStackVis={dockedStackVis}, fullscreenStackBounds=Rect({fullscreenStackBounds}), dockedStackBounds=Rect({dockedStackBounds})"

    # 调用函数并打印结果
    new_text = replace_braces(text)
    print(new_text)
# def get_log_messages(r_dir, dataset, shot=0):
#     train, test = [], []
#     if shot > 0:
#         demos = load_train_data(f"{r_dir}/dataset", dataset, shot)
#         for demo in demos:
#             train.append((demo[0].strip(), demo[1].strip()))
#     test_logs = load_test_data(f"{r_dir}/dataset", dataset)
#     for i, log in enumerate(test_logs):
#         test.append(log.strip())

#     return train, test

def grace_trans(str):
    return str.replace('{', '{{').replace('}', '}}')

if __name__ == '__main__':
    translate2()
