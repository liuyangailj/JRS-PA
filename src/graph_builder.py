# src/graph_builder.py
import networkx as nx
from typing import Dict, Any

def build_graph(nodes: Dict[str, Any], links: list) -> nx.DiGraph:
    """
    根据节点和链接信息构建有向图。

    :param nodes: 节点信息字典，键为节点名称，值为节点属性。
    :param links: 链接信息列表，每个链接包含源节点和目标节点等属性。
    :return: 构建的有向图。
    """
    G = nx.DiGraph()
    
    # 添加节点
    for node_name, node_info in nodes.items():
        G.add_node(node_name, **node_info)
    
    # 添加边
    for link in links:
        G.add_edge(link['source'], link['destination'], **link)
    
    return G
