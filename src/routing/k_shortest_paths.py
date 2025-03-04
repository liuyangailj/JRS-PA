import networkx as nx
from networkx.algorithms.simple_paths import shortest_simple_paths
import logging

logger = logging.getLogger(__name__)

def compute_k_shortest_paths(graph: nx.Graph, data: dict, k: int = 3) -> None:
    """
    针对每个流计算 k 条最短路，使用 NetworkX 的 shortest_simple_paths 方法.
    :param graph: NetworkX 图对象
    :param data: 包含流信息的数据字典
    :param k: 最短路径数量
    """

    if "streams" not in data:
        logger.error("JSON数据中未找到 'streams' 字段")
        return

    for stream in data["streams"]:
        talker = stream.get("talker")
        listeners = stream.get("listeners", [])
        if not talker:
            continue
        k_paths = []
        for listener in listeners:
            try:
                paths_generator = shortest_simple_paths(graph, talker, listener, weight='weight')
                for idx, path in enumerate(paths_generator):
                    if idx >= k:
                        break
                    k_paths.append(path)
            except nx.NetworkXNoPath:
                logger.warning("无路径: %s 到 %s", talker, listener)
                continue
        stream["path"] = k_paths
