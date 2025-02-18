# src/path_finder.py
import networkx as nx
import logging

logger = logging.getLogger(__name__)

def calculate_path_transmission_rate(G, path):
    
    total_rate = 0
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])
        total_rate += edge_data.get('transmission_rate', 0)
    return total_rate

def find_k_shortest_paths(G: nx.DiGraph, streams: list, k: int = 4) -> dict:
    k_shortest_paths = {}
    no_path_info = []

    for stream in streams:
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
                    numbered_paths[path_key] = {
                        'path': path,
                        'total_transmission_rate': tansmission_rate
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
