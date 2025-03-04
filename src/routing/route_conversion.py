import logging

logger = logging.getLogger(__name__)

def build_link_mapping(data: dict) -> dict:
    """
    根据"links/ports"字段构建连接映射字典。
    key 为(source.lower(), desination.lower()), value 为链路名称。
    参数：
        data (dict): 包含"links/ports"的数据字典。
    返回：
        dict: 连接映射字典。
    """
    mapping = {}
    for link in data.get("links/ports", []):
        src = link.get("source", "").lower()
        dst = link.get("destination", "").lower()
        name = link.get("name")
        if src and dst and name:
            mapping[(src, dst)] = name
    return mapping
        
def convert_stream_routes(data: dict) -> dict:
    """
    遍历所有 stream 的 path, 将route中的节点列表转化为链路表示。
    参数：
        data (dict): 包含"streams"的数据字典。
    返回：
        dict: 更新后的数据字典。
    """
    
    link_mapping = build_link_mapping(data)
    for stream in data.get("streams", []):
        for path in stream.get("path", []):
            route = path.get("route", [])
            new_route = []
            for i in range(len(route) - 1):
                nodeA = route[i].lower()
                nodeB = route[i+1].lower()
                link_name = link_mapping.get((nodeA, nodeB))
                if link_name:
                    new_route.append(link_name)
                else:
                    msg = f"未找到链路: {nodeA}->{nodeB}"
                    logger.warning(msg)
                    new_route.append(f"NotFound({nodeA}->{nodeB})")
            path["route"] = new_route
    return data
