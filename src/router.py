
"""
模块说明：
    该模块实现了一个 Router 类，用于计算网络拓扑中的 k 条最短路径，并计算流在其候选路径上的端到端延时。
    Router 类提供了以下方法：
    - compute_k_shortest_paths: 计算 k 条最短路径
    - get_stream_phy_delay_on_path: 计算流在其候选路径上无等待的端到端延时
    - select_optimal_routes: 筛选出每个流中path里phy_delay最小的route
"""
import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths

class DataError(Exception):
    """当输入数据缺失必要字段或格式错误时抛出的异常"""
    pass

class Router:
    def __init__(self, graph):
        """
        初始化Router对象.
        
        参数：
            graph (nx.Graph): 网络拓扑图 
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph 参数必须是一个 NetworkX 图对象")
        self.graph = graph

    def compute_k_shortest_paths(self, data, k=3):
        """
        针对 data 中的每个流计算 k 条最短路径.
        
        参数：
            data
            k
            
        返回：
            dict
        """
        if 'streams' not in data:
            raise DataError("输入数据缺少 streams 字段")
        
        for stream in data['streams']:
            talker = stream.get("talker")
            listeners = stream.get("listeners", [])
            if not talker:
                continue
            
            # 针对每个监听节点计算路径，多个监听节点的路径暂存在同一字段内
            stream_paths = []
            for listener in listeners:
                try:
                    paths_generator = shortest_simple_paths(
                        self.graph, talker, listener, weight='weight'
                        )
                    # 生成k条最短路径
                    k_paths = []
                    for idx, path in enumerate(paths_generator):
                        if idx >= k:
                            break
                        k_paths.append(path)
                    stream_paths.extend(k_paths)
                except nx.NetworkXNoPath:
                    # 当前talker到listener没有路径，继续处理下一个listener
                    continue
                
                stream["path"] = stream_paths
        return data

    def get_stream_phy_delay_on_path(self, data):
        """
        计算流在其候选路径上无等待的端到端延时,并过滤出满足deadline要求的路径.
        计算公式为：
            transmission_delay = framesize * framesPerPeriod / transmission_rate
            phy_delay = (transmission_delay + propagation_delay) * (num_of_links) + processing_delay * (num_of_links - 1)
        其中 num_of_links 为跳数。

        参数:
            data (dict): 包含网络流、节点及链路数据的字典。

        返回:
            dict: 更新后的数据字典，每个流的 'path' 字段中存放符合 deadline 要求的路径及对应延时。

        异常:
            DataError: 当缺失必要节点信息或链路信息时抛出异常。
        """
        # 查找Bridge节点的处理延时
        processing_delay = None
        nodes = data.get('nodes', [])
        if not nodes or not isinstance(nodes, list):
            raise DataError("数据中缺失节点信息，或格式不正确")
        for node in nodes:
            if node.get('isBridge', False):
                processing_delay = node.get('processingDelay', 0)
                break
        if processing_delay is None:
            print("未找到 isBridge=True 的节点或 processingDelay 数据不可用")
        
        # 从链路信息获取传输速率和传播延时
        transmission_rate = None
        propagation_delay = None
        ports = data.get('links/ports', [])
        if not ports or not isinstance(ports, list):
            raise DataError("数据中缺失链路信息，或格式不正确")
        for link in ports:
            if 'transmissionRate' in link and 'propagationDelay' in link:
                transmission_rate = link['transmissionRate']
                propagation_delay = link['propagationDelay']
                break
            
        if transmission_rate is None or propagation_delay is None:
            raise DataError("数据中缺失 transmissionRate 或 propagationDelay 数据")
        if transmission_rate == 0:
            raise DataError("transmissionRate 为 0，无法计算延时")
        # 更新每个流中的path数据，计算物理延时筛选出满足deadline的路径
        for stream in data.get('streams', []):
            framesize = stream.get('maxFrameSize', 0)
            framesperperiod = stream.get('framesPerPeriod', 0)
            deadline = stream.get('deadline', None)
            if deadline is None:
                raise DataError("流中缺少 deadline 数据")
            
            # 计算transmission_delay
            transmission_delay = framesize * framesperperiod / transmission_rate
            
            original_paths = stream.get('path', [])
            valid_paths = []            
            for route in original_paths:
                # 路由上的跳数
                num_of_hops = len(route) - 1
                phy_delay = (transmission_delay + propagation_delay) * (num_of_hops + 1) + processing_delay * num_of_hops
                if phy_delay < deadline:
                    valid_paths.append({
                        'route': route,
                        'phy_delay': phy_delay
                    })
            stream['path'] = valid_paths
        return data

    def select_optimal_routes(self, data):
        """
        筛选出每个流中path里phy_delay最小的route
        
        参数:
            data (dict): 包含网络流、节点及链路数据的字典。
            
        返回:
            dict: 更新后的数据字典，每个流的 'path' 字段中只包含最优路径。
        """
        for stream in data.get("streams", []):
            candidate_paths = stream.get("path", [])
            if candidate_paths:
                optimal_path = min(candidate_paths, key=lambda x: x["phy_delay"])          
                stream["path"] = [{
                    "route": optimal_path["route"],
                    "phy_delay": optimal_path["phy_delay"]
                    }]
            else:
                stream["path"] = []
        return data
