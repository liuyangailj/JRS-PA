import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.simple_paths import shortest_simple_paths

class NetworkTopology:
    def __init__(self, json_file):
        self.json_file = json_file
        self.data = None

    def load_data(self):
        """加载JSON数据，仅读取一次文件内容"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def save_data(self, output_file):
        """将处理后的数据保存到指定文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4)

def build_graph(data):
    """
    构建网络拓扑图。
    假设 data 包含 'nodes', 'links/ports' 字段
    """
    graph = nx.Graph()
    # 添加节点
    if 'nodes' in data:
        for node in data['nodes']:
            graph.add_node(node['name'], **node)
    else:
        print("JSON数据中未找到'nodes'字段")
    # 添加边：这里假设每个连接都包含 'source' 与 'target' 字段
    if 'links/ports' in data:
        for link in data['links/ports']:
            src = link['source']
            dst = link['destination']
            weight = link.get('weight', 1)
            graph.add_edge(src, dst, weight=weight, **link)
    else:
        print("JSON数据中未找到'links/ports'字段")
    return graph

def compute_k_shortest_paths(graph, data, k=3):
    """
    针对 data 中的每个流计算 k 条最短路径，并保存到流数据中
    使用 networkx.algorithms.simple_paths.shortest_simple_paths 方法依赖权重 'weight'
    """
    if 'streams' not in data:
        print("JSON数据中未找到'streams'字段")
        return

    for stream in data['streams']:
        talker = stream.get("talker")
        listeners = stream.get("listeners", [])
        if not talker:
            continue

        stream_paths = {}
        for listener in listeners:
            try:
                paths_generator = shortest_simple_paths(graph, talker, listener, weight='weight')
                k_paths = []
                for idx, path in enumerate(paths_generator):
                    if idx >= k:
                        break
                    k_paths.append(path)
                stream_paths = k_paths
            except nx.NetworkXNoPath:
                stream_paths = []
        stream["path"] = stream_paths

def get_stream_phy_delay_on_path(data):
    '''
    计算流在其候选路径上无等待的端到端延时
    streams: maxFrameSize,framesPerPeriod,deadline,path
    nodes: isBridge = ture, 的 processingDelay
    links/ports: transmissionRate, propagationDelay
    
    '''
    # 获取节点中 isBridge 为 True 的处理延时

    processing_delay = None
    for node in data.get('nodes', []):
        if node.get('isBridge', False):
            processing_delay = node.get('processingDelay', 0)
            break
    if processing_delay is None:
        print("未找到 isBridge=True 的节点或 processingDelay 数据不可用")
        return data
    
    # 获取链路传输速率与传播延迟
    transmission_rate = None
    propagation_delay = None
    for link in data.get('links/ports', []):
        if 'transmissionRate' in link and 'propagationDelay' in link:
            transmission_rate = link['transmissionRate']
            propagation_delay = link['propagationDelay']
            break
    if transmission_rate is None or propagation_delay is None:
        print("未找到 transmissionRate 或 propagationDelay 数据")
        return data

    # 遍历每个流，计算各候选路径的物理延时，并过滤掉超过 deadline 的路径
    for stream in data.get('streams', []):
        framesize = stream.get('maxFrameSize', 0)
        framesperperiod = stream.get('framesPerPeriod', 0)
        deadline = stream.get('deadline', None)
        if deadline is None:
            print("流中缺少 deadline 数据，跳过该流")
            continue

        original_paths = stream.get('path', [])
        valid_paths = []
        # 计算传输延时 = framesize * framesperperiod / transmission_rate
        if transmission_rate == 0:
            print("transmission_rate 为 0，无法计算延时")
            continue
        transmission_delay = framesize * framesperperiod / transmission_rate

        # 对每个候选路径计算phy_delay
        for route in original_paths:
            # route 可能已有多个节点组成路径
            num_of_hops = len(route) - 1
            phy_delay = (transmission_delay + propagation_delay) * (num_of_hops + 1) + processing_delay * num_of_hops
            # 如果满足 deadline，则保留，并保存计算结果
            if phy_delay < deadline:
                valid_paths.append({
                    'route': route,
                    'phy_delay': phy_delay
                })
        # 更新 stream 的路径信息
        stream['path'] = valid_paths

    return data

    # -------------------------------------------------        

def sort_streams(data):
    """
    对流数据进行排序:
    - 首先按照 `period` 升序排列
    - 如果 `period` 相同，则按流名 `name` 的字母序升序排列
    """
    streams = data.get("streams", [])

    # 按规则排序
    sorted_streams = sorted(streams, key=lambda s: (s.get("period", 0), s.get("name", "")))

    # 更新排序后的顺序
    data["sorted_stream_order"] = [s.get("name", "") for s in sorted_streams]
    data["streams"] = sorted_streams

    return data

def select_optimal_routes(data):
    """
    筛选出每个streams中path里phy_delay最小的那个对应的route，
    并只保留这个route和phy_delay作为新的streams的path。
    """
    for stream in data.get("streams", []):
        # 检查流是否有路径信息
        if "path" in stream and stream["path"]:
            # 找到物理延迟最小的路径
            optimal_path = min(stream["path"], key=lambda x: x["phy_delay"])
            # 只保留最优路径
            stream["path"] = [{"route": optimal_path["route"], "phy_delay": optimal_path["phy_delay"]}]
        else:
            # 如果没有路径信息，保留空列表
            stream["path"] = []

    return data

def draw_topology(graph):
    """使用Matplotlib和NetworkX绘制网络拓扑"""
    # 绘制网络拓扑图
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)  # 使用 spring 布局，也可以选择其他布局
    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue')
    # 绘制边
    nx.draw_networkx_edges(graph, pos, width=2)
    # 绘制节点标签
    nx.draw_networkx_labels(graph, pos, font_size=10, font_family="sans-serif")
    # 若想在边上绘制权重值
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plt.title("Network Topology")
    plt.axis('off')  # 不显示坐标轴
    plt.show()

def main():
    json_file = "../data/input/test_1.json"  # 请修改为实际的JSON文件路径
    output_file = "../data/output/output.json"  # 输出文件路径
    
    test_output_file_1 = "../data/output/output_1.json"  # 输出文件路径
    test_output_file_2 = "../data/output/output_2.json"  # 输出文件路径
    test_output_file_3 = "../data/output/output_3.json"  # 输出文件路径
    
    k = 4  # 短路径数量

    nt = NetworkTopology(json_file)
    nt.load_data()  # 仅加载一次数据

    # 使用通用函数构建图和计算 k 最短路径，保证数据结构不改变
    graph = build_graph(nt.data)  # 构建图 
    compute_k_shortest_paths(graph, nt.data, k=k)  # 计算 k 最短路径 
    get_stream_phy_delay_on_path(nt.data)  # 计算流的物理延迟
    nt.save_data(test_output_file_1)
    
    sort_streams(nt.data)  # 对流数据排序
    nt.save_data(test_output_file_2)
    
    select_optimal_routes(nt.data)  # 选择最佳路径
    nt.save_data(test_output_file_3)

    # # 保存处理后的数据
    nt.save_data(output_file)

    # 绘制网络拓扑 
    draw_topology(graph)

if __name__ == '__main__':
    main()
