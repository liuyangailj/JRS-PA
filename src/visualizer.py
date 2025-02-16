# visualizer.py

import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, Dict

def draw_topology(G: nx.Graph, title: str = "Network Topology") -> None:
    """
    绘制网络拓扑图。

    参数:
        G (nx.Graph): NetworkX 图对象。
        title (str): 图的标题，默认为 "Network Topology"。
    """
    try:
        pos = nx.spectral_layout(G)  # 使用谱布局
    except Exception as e:
        print(f"布局计算错误: {e}")
        pos = nx.spring_layout(G)  # 回退到弹簧布局

    try:
        nx.draw(
            G, pos, with_labels=True, node_size=500,
            node_color='skyblue', font_size=10, font_weight='bold', arrows=True
        )
        edge_labels = nx.get_edge_attributes(G, 'transmission_rate')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title(title)
        plt.show()
    except Exception as e:
        print(f"绘图错误: {e}")
