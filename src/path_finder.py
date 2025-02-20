# src/path_finder.py
# 计算出每条流的 k 条最短路径，并返回结果。
# 筛选出满足延时条件的路径集合

import networkx as nx
import logging

logger = logging.getLogger(__name__)

def calculate_path_transmission_rate(G, path):
    
    total_rate = 0
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])
        total_rate += edge_data.get('transmission_rate', 0)
    return total_rate

def calculate_d_phy(streams, link_prop_delay = 100, bridge_proc_delay = 1000):
    """
    计算流在其候选路径上无等待的端到端延时
    :param M:  
    :param d_prop:  传播时延，默认  100 ns
    :param d_proc:  处理时延，默认 1000 ns
    :return: Di_phy  ns
    """
    for stream in streams:
        stream_name = stream['name']
        path = stream['path']
        
    # 跳数 M：传输链路数为 M+1，则交换机数量为 M
    M = len(path) - 1
    
    total_d_trans = sum(d_trans_list)
    total_d_prop = (M + 1) * d_prop
    total_d_proc = M * d_proc

    d_phy = total_d_trans + total_d_prop + total_d_proc
    return d_phy

# -------------------------计算候选路径集合-------------------------
# 需要的输入：G, streams(所有属性), k
# 需要的输出：path:
def find_k_shortest_paths(G: nx.DiGraph, streams: list, k: int = 4) -> dict:
    k_shortest_paths = {}
    no_path_info = []

    for stream in streams:
        # 从stream_data中获取计算要用的流的信息
        talker = stream['talker']
        listeners = stream['listeners']
        stream_name = stream['name']
        
        k_shortest_paths[stream_name] = {}
        

        for listener in listeners:
            try:
                paths = list(nx.shortest_simple_paths(G, source=talker, target=listener, weight='weight'))
                # 调整输出格式为编号形式，
                numbered_paths = {}
                for i,path in enumerate(paths[:k]):
                    path_key = f"path_{i+1}"
                    tansmission_rate = calculate_path_transmission_rate(G, path)
                    d_phy = calculate_d_phy(path, d_prop=0.05, d_proc=1)
                    numbered_paths[path_key] = {
                        'path': path,
                        'total_transmission_rate': tansmission_rate,
                        'd_phy': d_phy
                    }
                k_shortest_paths[stream_name][listener] = numbered_paths
                   
            except nx.NetworkXNoPath:
                k_shortest_paths[stream_name][listener] = []
                no_path_info.append((stream_name, listener))
                logger.warning(f"未找到路径: Stream={stream_name}, Listener={listener}")

    if no_path_info:
        logger.info("以下流和监听器未找到路径:")
        for stream_name, listener in no_path_info:
            logger.info(f"Stream: {stream_name}, Listener: {listener}")

    return k_shortest_paths

# # ---------------------路径筛选---------------------
# def filter_paths(k_shortest_paths):
#     filtered_paths = {}
#     for stream in streams:
#         stream_name = stream['name']        
#         listeners = stream['listeners']
#         filtered_paths[stream_name] = {}
#         for listener in listeners:
#             paths = k_shortest_paths[stream_name][listener]
#             filtered = []
#             for path_key, path_info in paths.items():
#                 path = path_info['path']
#                 total_transmission_rate =