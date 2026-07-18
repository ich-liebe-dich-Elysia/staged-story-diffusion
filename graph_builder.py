"""
Phase 3：构建用户故事关系图。

输入：已确定角色和动作的故事列表
输出：三张关系图（近义 / 合作 / 依赖），用 networkx 表示

近义关系：角色相同 + 动作语义相似度 > 阈值
合作关系：两故事动作中有共享业务对象
依赖关系：一个动作的完成是另一个动作的前提
"""

from __future__ import annotations

import jieba.posseg as pseg
import networkx as nx

import config
from dllm_interface import DLLMModel
from validator import _semantic_similarity, _llm_judge

from typing import List
import re
# ------------------------------------------------------------------ #
# 辅助：业务对象提取 + 归一化
# ------------------------------------------------------------------ #

def _extract_business_objects(action: str) -> set[str]:
    """
    从动作文本中提取业务对象，归一化到标准词条。
    先用 jieba 提取名词，再查词表归一化。
    """
    nouns = {w for w, f in pseg.cut(action) if f in ("n", "nz", "nn", "nr")}
    normalized = set()
    for noun in nouns:
        for canonical, aliases in config.BUSINESS_OBJECTS.items():
            if noun in aliases or noun == canonical:
                normalized.add(canonical)
                break
        else:
            # 未命中词表的名词也保留，让规则有机会匹配
            normalized.add(noun)
    return normalized


def _shared_business_objects(action_a: str, action_b: str) -> list[str]:
    """返回两个动作共享的业务对象列表（归一化后的交集）。"""
    objs_a = _extract_business_objects(action_a)
    objs_b = _extract_business_objects(action_b)
    return sorted(objs_a & objs_b)


# ------------------------------------------------------------------ #
# 辅助：依赖关系判定
# ------------------------------------------------------------------ #

def _check_dependency_rules(action_a: str, action_b: str) -> bool:
    """
    用规则词表判定 action_a 是否依赖 action_b。
    遍历 DEPENDENCY_RULES：若 action_a 包含某 key，且 action_b 包含对应的某个 value → 依赖成立。
    """
    for key, prerequisites in config.DEPENDENCY_RULES.items():
        if key in action_a:
            if any(pre in action_b for pre in prerequisites):
                return True
    return False


def _check_dependency_llm(action_a: str, action_b: str) -> bool:
    """用 LLM 判定 action_a 是否依赖 action_b（兜底）。"""
    prompt = (
      f"在软件系统中，要完成动作\"{action_a}\"，是否必须先完成\"{action_b}\"？\n"
      f"只输出 是 或 否。"
    )
    return _llm_judge(prompt) == 1


# ------------------------------------------------------------------ #
# 主函数：构建三张关系图
# ------------------------------------------------------------------ #

#该函数为修改后函数
def build_graphs(
    stories: list[dict],
    dllm: DLLMModel,
    synonymy_threshold: float = config.SIM_SYNONYMY_THRESHOLD,
    use_llm_for_dependency: bool = True,
) -> tuple[nx.Graph, nx.Graph, nx.DiGraph]:
    """
    构建用户故事关系图。

    参数：
        stories : list of dict，每项包含
                  {"id": int, "role": str, "action": str, "token_ids": list[int], "regions": dict}
        pairs_array : 二维数组，每个元素为 [source_id, target_id] 表示依赖关系
                      source_id 依赖 target_id (target_id 先于 source_id)
        synonymy_threshold  : 近义关系的动作相似度阈值
        use_llm_for_dependency : 是否在规则未命中时用 LLM 判定依赖

    返回：
        synonymy_graph  : nx.Graph  — 无向，边属性 {"sim": float}
        cooperation_graph : nx.Graph — 无向，边属性 {"shared": list[str]}
        dependency_graph  : nx.DiGraph — 有向 i→j 表示 i 依赖 j（j 先于 i）
    """
    synonymy_graph   = nx.Graph()
    cooperation_graph = nx.Graph()
    dependency_graph  = nx.DiGraph()

    n = len(stories)
    for s in stories:
        synonymy_graph.add_node(s["id"])
        cooperation_graph.add_node(s["id"])
        dependency_graph.add_node(s["id"])

    #将stories中的role和action分离出来，拼接并标明序号，方便作为prompt使用
    simple_US_str=format_stories_to_string_simple(stories)
    #使用对话形式引导大模型一次输出所有依赖关系
    pairs_str=two_stage_conversation(dllm,simple_US_str)
    #将大模型输出内容转化为二维数组pairs_array
    pairs_array=parse_pairs_to_2d_array(pairs_str)

    # ---- 使用二维数组构建依赖关系 ----
    for pair in pairs_array:
        if len(pair) >= 2:
            source_id = pair[0]
            target_id = pair[1]
            # 验证节点是否存在
            if source_id in [s["id"] for s in stories] and target_id in [s["id"] for s in stories]:
                dependency_graph.add_edge(source_id, target_id)

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = stories[i], stories[j]

            # ---- 近义关系 ----
            if si["role"] == sj["role"]:
                sim = _semantic_similarity(si["action"], sj["action"])
                if sim >= synonymy_threshold:
                    synonymy_graph.add_edge(si["id"], sj["id"], sim=sim)

            # ---- 合作关系 ----
            shared = _shared_business_objects(si["action"], sj["action"])
            if shared:
                cooperation_graph.add_edge(si["id"], sj["id"], shared=shared)

    return synonymy_graph, cooperation_graph, dependency_graph

def get_dependency_paths(dep_graph: nx.DiGraph) -> list[list[int]]:
    """
    从依赖图中提取所有简单路径（拓扑排序后的有向路径）。
    若存在环则忽略环中的回边（实际需求中依赖关系不应有环）。
    返回按拓扑顺序排列的路径列表，每条路径是 story id 的有序列表。
    """
    # 检测并去除环（正常需求中不应有循环依赖，此处做防御处理）
    dag = dep_graph.copy()
    while not nx.is_directed_acyclic_graph(dag):
        # 找到一条环并去掉其中一条边
        cycle = nx.find_cycle(dag)
        dag.remove_edge(*cycle[0])

    # 对弱连通分量分别提取路径
    paths = []
    for component in nx.weakly_connected_components(dag):
        sub = nx.DiGraph(dag.subgraph(component).copy())
        # 找所有从入度为 0 的节点出发的路径
        roots = [n for n in sub.nodes if sub.in_degree(n) == 0]
        for root in roots:
            for target in sub.nodes:
                if root == target:
                    continue
                for path in nx.all_simple_paths(sub, root, target):
                    if len(path) >= 2:
                        paths.append(path)

    # 去重并按长度降序排列（长路径先处理，短路径可能是长路径的子集）
    unique_paths = list({tuple(p): p for p in paths}.values())
    unique_paths.sort(key=len, reverse=True)
    return unique_paths





#以下为新添加
def generate_response(dllm: DLLMModel, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8) -> str:
    """
    使用离散扩散语言模型生成对话式回复。
    
    通过迭代式解码策略，逐步填充 [MASK] token 来生成回复内容。
    该方法模拟了商用对话模型的逐词生成行为。
    
    Args:
        dllm: DLLMModel 实例
        prompt: 用户输入的提示文本
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
    
    Returns:
        生成的回复文本
    """
    # 1. 将 prompt 编码为 token ids
    prompt_ids = dllm.tokenizer.encode(prompt, add_special_tokens=True)
    
    # 2. 初始化序列：添加 [MASK] 作为待生成区域
    # 格式：[CLS] prompt [SEP] [MASK] * max_new_tokens [SEP]
    generated_len = 0
    mask_region_start = len(prompt_ids)  # 记录 [MASK] 区域的起始位置
    
    # 构建完整序列
    full_ids = prompt_ids.copy()
    full_ids += [dllm.mask_token_id] * max_new_tokens
    full_ids += [dllm.sep_token_id]
    
    # 3. 迭代解码
    # 维护一个布尔列表，标记哪些位置仍是 [MASK]
    is_masked = [False] * len(full_ids)
    for i in range(mask_region_start, len(full_ids) - 1):
        is_masked[i] = True
    
    # 记录每个位置的置信度，用于决定填充顺序
    confidence_history = []
    
    while any(is_masked):
        # 前向传播获取 logits
        logits = dllm.forward(full_ids)
        
        # 找出当前所有 [MASK] 位置
        masked_positions = [i for i, masked in enumerate(is_masked) if masked]
        
        if not masked_positions:
            break
        
        # 为每个 [MASK] 位置计算置信度
        pos_confidences = []
        for pos in masked_positions:
            conf = dllm.get_confidence(logits, pos)
            pos_confidences.append((pos, conf))
        
        # 选择置信度最高的位置进行填充（类似并行解码策略）
        # 也可以选择一次填充所有位置，这里采用逐位置填充以更接近自回归行为
        pos_to_fill, _ = max(pos_confidences, key=lambda x: x[1])
        
        # 采样新 token
        new_token = dllm.sample(logits, pos_to_fill, temperature=temperature)
        
        # 更新序列
        full_ids[pos_to_fill] = new_token
        is_masked[pos_to_fill] = False
        generated_len += 1
        
        # 可选：记录生成进度
        # if generated_len % 10 == 0:
        #     print(f"Generated {generated_len}/{max_new_tokens} tokens")
    
    # 4. 解码生成的内容
    # 提取从第一个 [MASK] 开始到结束的部分
    response_ids = full_ids[mask_region_start:]
    # 去掉结尾的 [SEP]
    if response_ids and response_ids[-1] == dllm.sep_token_id:
        response_ids = response_ids[:-1]
    
    response_text = dllm.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    return response_text


def format_stories_to_string_simple(stories: list[dict]) -> str:
    """
    从 stories 列表中提取 role 和 action，格式化为带序号的字符串
    """
    return "\n".join([
        f"{idx}. {story['role']}: {story['action']}"
        for idx, story in enumerate(stories, start=1)
        if story.get('role') and story.get('action')
    ])

def two_stage_conversation(
    dllm: DLLMModel,
    user_input: str,  # 外部输入的部分
    max_pairs: int = 60,  # 有序数对的最大数目
    temperature: float = 0.7
) -> str:
    """
    两阶段对话生成
    
    Args:
        dllm: DLLMModel实例
        user_input: 用户输入的部分
        str1: 第一次prompt的前半部分
        str2: 第二次prompt的完整内容
        max_pairs: 有序数对的最大可能数目
        temperature: 采样温度
    
    Returns:
        (第一次对话的回复, 第二次对话解析出的有序数对列表)
    """
    
    # ==================== 第一阶段对话 ====================
    str1 = """
## Role Definition
You are a seasoned software requirements analysis expert, skilled at extracting semantic relationships from User Stories and constructing structured analysis results. 
## Task
Utilizing common sense from daily life and professional knowledge, using the role-action pairs in the **User-Story**, generate a **DEPENDENCY_RULES** table. 
## Definition
**Dependency Relationship**:
Refers to a situation where the implementation or execution of one user-story must be based on the completion or existence of another user-story, indicating a clear sequential dependency.
**DEPENDENCY_RULES**:
Refers to a rule vocabulary. An example of its structure is as follows. 
DEPENDENCY_RULES = {
"View Order":   ["Place Order", "Purchase"],
"Modify":       ["Create", "Register", "Add"],
"Delete":       ["Create", "Register"],
"Evaluate Product":   ["Purchase", "Place Order"],
"View Logistics":   ["Place Order", "Purchase"],
"Login":       ["Register"],
"Return":       ["Purchase"],
"Cancel Order":   ["Place Order"] }
I will give an example to illustrate its meaning. The action "view orders" depends on the action list ["place order", "purchase"], the action "modify" depends on the action list ["create", "register", "add"], and so on. 
## Constraints
You can only conduct the analysis based on the provided User-Story content. No additional assumptions can be introduced. The input content cannot be modified or optimized. Your output must strictly follow the format of **DEPENDENCY_RULES**. No other content is allowed to be generated. 
**User Story**:

"""
    # 拼接第一次的完整prompt
    first_prompt = str1 + user_input
    
    # 生成第一次回复
    first_response = generate_response(
        dllm=dllm,
        prompt=first_prompt,
        max_new_tokens=150,  # 根据实际需求调整
        temperature=temperature
    )
    
    # ==================== 第二阶段对话 ====================
    str2="""
## Task
Maintain your character and based on the **DEPENDENCY_RULES** you generated, determine the **Dependency Relationship** diagram between the output user stories. This directed graph should be presented in the form of an edge set, and the specific format is as described in the **Reference Dialogue**. 
## Definition
**Reference Dialogue**:
From this, you need to understand how to output and the meanings of the numbers in the binary result output. (a, b) is an edge in a **Dependency Relationship** graph, indicating that the user story with sequence number b depends on the user story with sequence number a.
An example is as follows. 
Number of user stories: 24
1.Customer: Log in to account
2.Customer: Register account
3.Customer: Search for products
4.Customer: Browse products
5.Customer: Add product to shopping cart
6.Customer: Purchase product
7.Customer: Create order
8.Customer: View order
9.Customer: Delete shopping cart item
10.Customer: Modify quantity of items in shopping cart
11.Customer: Select product specification
12.Customer: Delete order
13.Customer: Add contact information
14.Customer: Modify contact information
15.Customer: Delete contact information
16.Delivery person: Log in to account
17.Delivery person: Register account
18.Delivery person: View delivery order
19.Delivery person: Modify delivery order
20.Business user: Log in to account
21.Business user: Register account
22.Business user: Add product
23.Business user: Modify product
24.Business user: Delete product
Output:
(6, 11)
(4, 12)
(1, 5)
(7, 14) 
## Constraints
You can only conduct the analysis based on the provided User-Story content. No additional assumptions can be introduced. The input content cannot be modified or optimized. Your output must strictly follow the "output" format in the **Reference Dialogue**, consisting of several pairs of items. No other content is allowed to be generated.

"""

    # 构建第二次的完整prompt（包含第一次的回复作为上下文）
    second_prompt = f"Based on the previous conversation: \nUser: {first_prompt}\nAssistant:{first_response}\n\n{str2}"

    # 生成第二次回复
    second_response = generate_response(
        dllm=dllm,
        prompt=second_prompt,
        max_new_tokens=300,  # 可能需要更多token来生成有序数对
        temperature=temperature
    )
    
    return second_response

def parse_pairs_to_2d_array(pairs_str: str) -> List[List[int]]:
    """
    从包含有序数对的字符串中解析出所有数对，并转换为二维数组
    
    Args:
        pairs_str: 包含有序数对的字符串，格式如 "(1,2) (3,4) (5,6)" 或每行一个
    
    Returns:
        二维数组，arr[i][0] = 第一个数，arr[i][1] = 第二个数
        如果没有解析到任何数对，返回空列表
    """
    # 正则表达式匹配各种格式的有序数对
    # 匹配格式：(数字, 数字) 或 (数字,数字) 或 [数字, 数字] 等
    pattern = r'[\(\[（]\s*(\d+)\s*[,，]\s*(\d+)\s*[\)\]）]'
    
    # 查找所有匹配
    matches = re.findall(pattern, pairs_str)
    
    # 转换为二维数组
    result = []
    for match in matches:
        num1 = int(match[0])
        num2 = int(match[1])
        result.append([num1, num2])
    
    return result