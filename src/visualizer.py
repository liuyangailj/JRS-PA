import networkx as nx
import matplotlib.pyplot as plt

def draw_topology(
    graph: nx.Graph,
    title: str = "Network Topology",
    node_size: int = 700,
    node_color: str = "lightblue",
    edge_color: str = "gray",
    font_size: int = 10,
    layout: str = "spring"
) -> None:
    """
    绘制网络拓扑图。

    参数:
        graph (nx.Graph): NetworkX 图对象。
        title (str): 图的标题，默认为 "Network Topology"。
        node_size (int): 节点的大小，默认为 700。
        node_color (str): 节点的颜色，默认为 "lightblue"。
        edge_color (str): 边的颜色，默认为 "gray"。
        font_size (int): 节点标签的字体大小，默认为 10。
        layout (str): 布局方式 ("spring", "spectral", "circular")，默认为 "spring"。
    """
    try:
        # 布局选择
        if layout == "spring":
            pos = nx.spring_layout(graph)
        elif layout == "spectral":
            pos = nx.spectral_layout(graph)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        else:
            raise ValueError(f"未知布局类型: {layout}")
    except Exception as e:
        print(f"布局错误: {e}，使用默认 spring 布局。")
        pos = nx.spring_layout(graph)  # 回退到 spring 布局

    try:
        plt.figure(figsize=(10, 8))
        # 绘制节点与边
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color)
        nx.draw_networkx_edges(graph, pos, edge_color=edge_color, width=2)
        # 绘制节点标签与边标签
        nx.draw_networkx_labels(graph, pos, font_size=font_size, font_family="sans-serif")
        edge_labels = nx.get_edge_attributes(graph, "transmission_rate")
        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        plt.title(title)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
    except Exception as e:
        print(f"绘图错误: {e}")
