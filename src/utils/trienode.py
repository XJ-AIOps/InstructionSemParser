import pickle
import pandas as pd

import sys  # 导入sys模块
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000


# 测试代码

class TrieNode:
    def __init__(self):
        self.children = {}  # 子节点，这里存储的是单词
        self.is_end_of_word = False  # 标记该节点是否为某个单词的结尾

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        if word not in node.children:  # 如果单词不在子节点中，则添加新节点
            node.children[word] = TrieNode()
        node = node.children[word]
        node.is_end_of_word = True  # 标记为单词的结尾

    def find(self, word):
        node = self.root
        if word in node.children:
            return node.children[word].is_end_of_word
        return False

    def print_all_words(self, node=None, current_word=""):
        if node is None:
            node = self.root
        if node.is_end_of_word and current_word:
            print(current_word)
        for word, child in node.children.items():
            self.print_all_words(child, current_word + word)
    
    def print_all_paths(self, node=None, path=None):
        if node is None:
            node = self.root
            path = []

        if node.is_end_of_word and path:
            print(' -> '.join(path))  # 输出路径

        for word, child in node.children.items():
            self.print_all_paths(child, path + [word])


logs = pd.read_csv(f'./dataset/Android/Android_2k.log_structured_corrected.csv')
rawlist = logs.EventTemplate.tolist()
# 假设 rawlist 已经定义并且包含了字符串
trie = Trie()
for log in rawlist:
    words = log.split()  # 分割成单词列表
    for word in words:
        trie.insert(word)

# 输出字典树中的所有单词
trie.print_all_paths()
