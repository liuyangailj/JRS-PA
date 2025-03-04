

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