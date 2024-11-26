import os
import numpy as np
import re
from openai import OpenAI
import json

import torch
import matplotlib.pyplot as plt
import ast

from pydantic import BaseModel
from typing import List

class TokenAlignment(BaseModel):
    comment_token: List[str]
    code_token: List[str]

class AlignmentOutput(BaseModel):
    alignments: List[TokenAlignment]

def initialize_centroids(X, k):
    # 随机选择k个初始聚类中心
    indices = torch.randperm(X.size(0))[:k]
    return X[indices]

def compute_distances(X, centroids):
    # 计算每个点到每个聚类中心的欧氏距离
    distances = torch.cdist(X, centroids, p=2)  # 使用L2范数
    return distances

def kmeans(X, k, num_iters=100):
    # 初始化聚类中心
    centroids = initialize_centroids(X, k)
    
    for _ in range(num_iters):
        # 计算距离并为每个点分配最近的聚类中心
        distances = compute_distances(X, centroids)
        labels = distances.argmin(dim=1)
        
        # 重新计算聚类中心
        new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(k)])
        
        # 检查是否有空聚类
        nan_mask = torch.isnan(new_centroids)
        new_centroids[nan_mask] = centroids[nan_mask]  # 如果出现空聚类，保持旧的中心

        # 更新聚类中心
        centroids = new_centroids
    
    # 返回最终的聚类中心和每个点的标签
    return centroids, labels

# 查找顺序匹配的索引 (允许非连续但必须顺序)
def find_ordered_token_indices(tokens, full_token_list):
    """找到匹配的 token 在 full_token_list 中的索引，要求匹配的 token 顺序出现"""
    token_indices = []
    current_index = -1  # 记录当前匹配的位置，初始为 -1
    
    for token in tokens:
        for idx in range(current_index + 1, len(full_token_list)):
            # print(idx, current_index, full_token_list[idx], token)
            if token == full_token_list[idx]:
                token_indices.append(idx)
                current_index = idx  # 更新当前匹配的索引位置
                break
    if len(token_indices) == 0:
        return []
    
    return token_indices

# 将索引序列转换为区间格式
def convert_to_intervals(indices):
    """将索引列表转换为区间格式"""
    if not indices:
        return []
    
    intervals = []
    start = indices[0]
    end = indices[0]
    
    for i in range(1, len(indices)):
        if indices[i] == end + 1:
            end = indices[i]
        else:
            intervals.append(start)
            intervals.append(end)
            start = indices[i]
            end = indices[i]
    
    intervals.append(start)
    intervals.append(end)
    return intervals



# set openai environ and key
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["OPENAI_API_KEY"] = "sk-kXKBrhvV20vfBpAdh7cdV8QRIgeS0hXceIuopLc5yEyeERKX"
os.environ["OPENAI_BASE_URL"] = "https://api.key77qiqi.cn/v1"

# load all training data
# 文件路径
file_path = '/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/dataset/python/train.jsonl'

# 存储所有数据的列表
train_data = []

# 读取 JSONL 文件中的所有数据
with open(file_path, 'r') as f:
    for line in f:
        train_data.append(json.loads(line.strip()))

# load all training data tokens
# 文件路径
file_path = '/home/yiming/cophi/training_dynamic/gcb_tokens_temp/Model/Epoch_1/tokenized_code_tokens_train.json'

# 读取 JSON 文件
with open(file_path, 'r') as f:
    code_tokens_strs = json.load(f)

# 文件路径
nl_file_path = '/home/yiming/cophi/training_dynamic/gcb_tokens_temp/Model/Epoch_1/tokenized_comment_tokens_train.json'

# 读取 JSON 文件
with open(nl_file_path, 'r') as f:
    nl_tokens_strs = json.load(f)

# 现在 code_tokens_strs 变量中包含了从 JSON 文件读取的数据
print("len(code_tokens_strs)", len(code_tokens_strs))  # 可以查看加载的数据
print("len(nl_tokens_strs)", len(nl_tokens_strs))  # 可以查看加载的数据

# load training data embeddings (pretrained model)
# Load the embeddings from the stored numpy file
code_token_output_path = "/home/yiming/cophi/training_dynamic/gcb_tokens_temp/train_code_cls_token_pt.npy"
all_embeddings = np.load(code_token_output_path)

print("all_embeddings.shape", all_embeddings.shape) 

# load selected unlabeled indices
random_indices = np.load('/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/random_indices.npy')

# load human labeled info
input_path = "/home/yiming/cophi/training_dynamic/NL-code-search-Adv/model/codebert/token_alignment/label_human_teacher.jsonl"
idx_list = []
match_list = []

with open(input_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip().rstrip(',')  # 去除行末的逗号
        json_obj = json.loads(line)
        idx_list.append(json_obj['idx'])
        match_list.append(json_obj['match'])

print("len(idx_list)", len(idx_list)) 

# load already auto labeled info
input_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/label_human_auto.jsonl"
auto_idx_list = []

with open(input_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip().rstrip(',')  # 去除行末的逗号
        json_obj = json.loads(line)
        auto_idx_list.append(json_obj['idx'])

# 使用集合操作来加速从 random_indices 中移除 idx_list 中的索引
# unlabeled_indices = list(set(random_indices) - set(auto_idx_list))
unlabeled_indices = list(set(range(33000, len(all_embeddings))) - set(auto_idx_list))
print("len(unlabeled_indices)", len(unlabeled_indices)) 

# 提取聚类中心的嵌入
cluster_centers = all_embeddings[idx_list]

# 提取未标注样本的嵌入
unlabeled_embeddings = all_embeddings[unlabeled_indices]
# 随机打乱未标注索引的顺序
import random
random.shuffle(unlabeled_indices)

# # 计算所有未标注样本到所有聚类中心的距离
# distances = np.linalg.norm(all_embeddings[:, np.newaxis] - cluster_centers, axis=2)

# # 找到每个样本到最近的聚类中心的距离和对应聚类中心的索引
# min_distances = distances.min(axis=1)
# closest_centers = distances.argmin(axis=1)

# 初始化用于存储每个未标注样本的最小距离和最近的聚类中心索引
min_distances = []
closest_centers = []

# 逐个计算未标注样本到所有聚类中心的距离，保留最小值
for unlabel_i in unlabeled_indices:
    # 计算当前未标注样本到所有聚类中心的距离
    distances_to_centers = np.linalg.norm(all_embeddings[unlabel_i] - cluster_centers, axis=1)
    # 找到最小距离和对应的聚类中心索引
    min_distance = distances_to_centers.min()
    closest_center = distances_to_centers.argmin()
    
    # 保存最小距离和对应聚类中心
    min_distances.append(min_distance)
    closest_centers.append(closest_center)
    if unlabel_i % 10000 == 0:
        print(unlabel_i)

# 定义距离阈值
distance_threshold = 2.7  # 这个值可以根据具体需求进行调整

# 筛选出距离小于阈值的样本索引
auto_label_indices = [unlabeled_indices[i] for i in range(len(unlabeled_indices)) if min_distances[i] < distance_threshold]
closest_teachers = [closest_centers[i] for i in range(len(unlabeled_indices)) if min_distances[i] < distance_threshold]
cannotlabeled_indices = list(set(unlabeled_indices) - set(auto_label_indices))

print("len(auto_label_indices)", len(auto_label_indices)) 
print("len(cannotlabeled_indices)", len(cannotlabeled_indices)) 

system_prompt = "You are an expert at aligning tokens between comments and code. You can accurately identify the similarities and differences between tokens, and you are highly skilled at matching tokens based on their semantics and functionality. You are given input data consisting of comment tokens and code tokens, and your task is to align them by identifying concepts in the comments and matching them to corresponding code tokens. Use the example cases below and output your results in the specified format."

# auto labelling 
# for i in range(len(auto_label_indices)):
for auto_label_ind in range(10):

    # construct teacher output
    teacher_ind = closest_teachers[auto_label_ind]
    cur_match_list = match_list[teacher_ind]
    cur_idx = idx_list[teacher_ind]
    teach_code_tokens = code_tokens_strs[cur_idx][1:]
    teach_comment_tokens = nl_tokens_strs[cur_idx][1:]

    teacher_output = ""
    match_idx = 0
    # 遍历 match_list
    for match_item in cur_match_list:
        match_idx += 1
        comment_match = match_item[0]
        code_match = match_item[1]
        matched_comment_tokens = []
        for i in range(0, len(comment_match), 2):
            comment_start, comment_end = comment_match[i], comment_match[i+1]
            matched_comment_tokens.extend(teach_comment_tokens[comment_start:comment_end+1])  # 提取代码 tokens
        
        # 处理代码 tokens 的区间，成对提取 [start, end]
        matched_code_tokens = []
        for i in range(0, len(code_match), 2):
            code_start, code_end = code_match[i], code_match[i+1]
            matched_code_tokens.extend(teach_code_tokens[code_start:code_end+1])  # 提取代码 tokens
        
        teacher_output += f"{match_idx}. {matched_comment_tokens}, {matched_code_tokens}\n"

    # construct teacher_prompt
    teacher_prompt = f"""
    Below is an example that demonstrates how to align comment tokens and code tokens:
    **Teacher Example:**
    Comment Tokens (teach_comment_tokens):
    {teach_comment_tokens}
    Code Tokens (teach_code_tokens):
    {teach_code_tokens}
    **Matching Output:**
    {teacher_output}
    """

    # construct student input
    student_idx = auto_label_indices[auto_label_ind]
    student_code_tokens = code_tokens_strs[student_idx][1:]
    student_comment_tokens = nl_tokens_strs[student_idx][1:]

    # construct student_prompt
    student_prompt = f"""
    Now, it’s your turn to align the tokens. You will be given two lists of tokens: one list contains comment tokens, and the other contains code tokens. Your task is to follow the pattern from the Teacher Example above, where each concept in the comment tokens is matched with the corresponding code tokens.

    Here are the tokens you need to process:

    Comment Tokens (student_comment_tokens):
    {student_comment_tokens}

    Code Tokens (student_code_tokens):
    {student_code_tokens}

    **Important Notes**:
    1. Not all tokens from the comments or the code must participate in the alignment. Some tokens in both the comment and the code may not have a corresponding match.
    2. Tokens from the code cannot be aligned with multiple concepts from the comments. Each code token can only align with one concept from the comment tokens.
    3. For each concept in the comment tokens, you must try to find as many semantically related tokens from the code as possible, while ensuring that code tokens are not reused across multiple concepts.
    4. Please ensure that the tokens are aligned and listed in the order in which they appear in the input.

    Follow the provided example and find the concepts in the `student_comment_tokens` that align with tokens in the `student_code_tokens`. Output the aligned tokens in the following format:
    {{
        "alignments": [
            {{"comment_token": ["token1"], "code_token": ["tokenA"]}},
            {{"comment_token": ["token2"], "code_token": ["tokenB"]}},
            {{"comment_token": ["token3"], "code_token": ["tokenC"]}}
        ]
    }}
    """

    promt_str = system_prompt + teacher_prompt + student_prompt

    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))

    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06", 
        messages=[{"role": "user", "content": promt_str}], 
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "alignment_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "alignments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "comment_token": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "code_token": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["comment_token", "code_token"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["alignments"],
                    "additionalProperties": False
                }
            }
        }, 
        max_tokens=500)
    
    # 检查内容，确保是以花括号开始和结束的 JSON 数据
    response_content = response.choices[0].message.content
    json_content = re.search(r'\{.*\}', response_content, re.DOTALL)
    if json_content:
        json_str = json_content.group(0)
        try:
            # 尝试加载为 JSON
            alignment_output = json.loads(json_str)

        except json.JSONDecodeError as e:
            print("JSON 解码错误:", e)
            continue
    else:
        print("未找到有效的 JSON 内容")
        continue

    # alignment_output = alignment_output = json.loads(response.choices[0].message.content)
    
    # 保存最终的结果
    final_results = []
    print(teacher_ind, student_idx)

    # 遍历 alignments 中的每个对齐项
    for alignment in alignment_output["alignments"]:
        comment_tokens = alignment["comment_token"]
        code_tokens = alignment["code_token"]
        
        # 输出每个对齐项的 comment_token 和 code_token
        print("Comment Tokens:", comment_tokens)
        print("Code Tokens:", code_tokens)
        print("-" * 40)
        
        # 在 student_comment_tokens 和 student_code_tokens 中查找这些 tokens 的索引
        comment_indices = find_ordered_token_indices(comment_tokens, student_comment_tokens)
        code_indices = find_ordered_token_indices(code_tokens, student_code_tokens)
        
        # 将找到的索引转换为区间格式
        comment_intervals = convert_to_intervals(comment_indices)
        code_intervals = convert_to_intervals(code_indices)

        print(comment_intervals, code_intervals)
        
        if len(comment_intervals) > 0 and len(code_intervals) > 0:
            final_results.append([comment_intervals, code_intervals])

    # 如果 final_results 为空，则跳过此样本并记录
    if not final_results:
        print("not final_results", auto_label_ind)
        cannotlabeled_indices.append(student_idx)
        continue

    print("final_results", final_results)

    # 创建新的数据项
    new_entry = {
        "idx": int(student_idx),
        "match": final_results
    }

    # 指定文件路径
    file_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/label_human_auto.jsonl"

    # 追加新数据到 JSONL 文件
    with open(file_path, 'a') as file:
        # 将新条目转换为 JSON 字符串并写入文件（自动换行以符合 JSONL 格式）
        file.write(json.dumps(new_entry) + '\n')

    print("新数据已成功添加到文件末尾。")

# print("len(cannotlabeled_indices)", len(cannotlabeled_indices)) 
# # 定义存储 `cannotlabeled_indices` 的文件路径
# cannotlabeled_file_path = "/home/yiming/cophi/projects/fork/CodeBERT/GraphCodeBERT/codesearch/cannotlabeled_indices.json"
# cannotlabeled_indices = [int(idx) for idx in cannotlabeled_indices]
# # 将无法标注的索引保存为 JSON 文件
# with open(cannotlabeled_file_path, 'w') as file:
#     json.dump(cannotlabeled_indices, file)

# print(f"无法标注的索引已保存到 {cannotlabeled_file_path}")
