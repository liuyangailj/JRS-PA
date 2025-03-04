import networkx as nx
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
def build_graph(data: dict) -> nx.Graph:
    """
    从数据字典构建 NetworkX 图对象。
    
    参数:
        data (dict): 包含'nodes'和'links/ports'的数据字典。
        
    返回:
        nx.Graph: NetworkX 图对象。
    """
    try:
        graph = nx.Graph()
        if 'nodes' in data:
            for node in data['nodes']:
                graph.add_node(node['name'], **node)
        else:
            logger.error("JSON数据中未找到 'nodes' 字段")
            
        if 'links/ports' in data:
            for link in data['links/ports']:
                src = link.get('source')
                dst = link.get('destination')
                if src and dst:
                    weight = link.get('weight', 1)
                    graph.add_edge(src, dst, weight=weight, **link)
        else:
            logger.error("JSON数据中未找到 'links/ports' 字段")
        return graph
        
    except Exception as e:
        logger.error("构建图对象失败: %s", e)
        raise
    
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
    参数：
        graph (nx.Graph): NetworkX 图对象。
        title (str): 图的标题，默认为 "Network Topology"。
        node_size (int): 节点的大小，默认为 700。
        node_color (str): 节点的颜色，默认为 "lightblue"。
        edge_color (str): 边的颜色，默认为 "gray"。
        font_size (int): 节点标签的字体大小，默认为 10。
        layout (str): 布局方式 ("spring", "spectral", "circular")，默认为 "spring"。
    """
    try:
        if layout == "spring":
            pos = nx.spring_layout(graph)
        else:
            pos = nx.circular_layout(graph)

        nx.draw(graph, pos, with_labels=True, node_size=node_size,
                node_color=node_color, edge_color=edge_color, font_size=font_size)
        edge_labels = nx.get_edge_attributes(graph, "transmission_rate")
        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
        plt.title(title)
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        logger.error("绘图错误: %s", e)
        raise
    