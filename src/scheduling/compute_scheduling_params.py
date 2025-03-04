import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

# ---------------------step 1 TDI计算开始--------------------------------

def calculate_tdi_and_ts(data):
    """
    Step 1: 计算时间分段间隔 (TDI)调度基本单元，即基周期
    和时间槽长度 (TS)一个最大帧从一个节点发出到在下一节点被完全接收的时间
    需获取参数：pdbase, dtrans, dprop, dproc, nl_speed, l_mtu
    
    """
    
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
    
    # 获取流的最大帧大小和每周期帧数
    first_stream = data.get('streams', [{}])[0] # 仅获取第一个流的信息
    framesize = first_stream.get('maxFrameSize', 0)
    framesperperiod = first_stream.get('framesPerPeriod', 0)
    
    # 计算传输延时 = framesize * framesperperiod / transmission_rate
    if transmission_rate == 0:
        print("transmission_rate 为 0，无法计算延时")
        return data
    
    transmission_delay = framesize * framesperperiod / transmission_rate

    # 获取流的最小周期
    # 提取所有的 period 并找到最小值
    streams = data.get('streams', [])
    periods = [stream.get('period', 0) for stream in streams]       
    pdbase = min(periods) # 根据公式 (5)      
    TDI = pdbase  # 根据公式 (8)
    
    GBI = 1542/ transmission_rate  # 根据公式 (9)
    
    TS = transmission_delay + propagation_delay + processing_delay  # 根据公式 (10)
    
    return TDI, TS, GBI

# ---------------------step 1 TDI计算结束--------------------------------

# ---------------------step 2 NTSTC计算开始--------------------------------
def calculate_tsai_and_ntstc(data):
    """
    Step 2: 
    计算： 
    1. TSAI (Time Slot Allocation Interval) 时间槽分配间隔
    2. NTSTC (Number of Time Slots to be Allocated) 需要分配的时间槽数量
    3. TCI (Time Slot Count Interval) 时间槽计数间隔
    4. NTCI (Number of Time Slot Count Intervals) 时间槽计数间隔数量
    
    需获取参数：
    1. 每条流的最优路径；
    2. 所有使用到的出端口UEP（used egress ports）列表, 和UEP的数量 
    3. 每个UEP的流量集合列表，和对应集合中的流量个数
    4. 每个流的周期，即TSAI_i
    5. 每个端口上的TDI


    stream_periods, hops, nl_speed, pdbase
    """
    streams = data.get("streams", [])
    _,TS,_ = calculate_tdi_and_ts(data)

    # 用于存储所有去重的端口，及端口对应的流信息
    used_ports = set()
    # 结构： { port: [ { "name": stream_name, "period": period, "hops": hops }, ... ] }
    port_to_streams = {}
    portTDI = calculate_tdi_and_ts(data)[0]

    # 遍历所有 streams 提取信息
    for stream in streams:
        stream_name = stream.get("name")
        stream_period = stream.get("period")
        stream_paths = stream.get("path", [])
        # 对于每个流，可能有多个 path，每个 path 中的 route 端口列表
        for path in stream_paths:
            routes = path.get("route", [])
            # 计算该流在此条路由上的 hop 数：取 route 长度减 1
            hops = max(len(routes) - 1, 0)
            for port in routes:
                port = port.strip()
                used_ports.add(port)
                if port not in port_to_streams:
                    port_to_streams[port] = []
                # 保存当前流的信息到该端口
                port_to_streams[port].append({
                    "name": stream_name,
                    "period": stream_period,
                    "hops": hops
                })

    # 构造结果的 Used_ports 列表
    used_ports_list = []
    for port in used_ports:
        streams_info = port_to_streams.get(port, [])
        if not streams_info:
            continue

        # 统计该端口上的流名称列表
        streams_on_this_port = [s["name"] for s in streams_info]
        streamsNum = len(streams_info)
        # portTDI 为该端口上所有流中 period 的最大值, 在此处的计算中，TDI是否还是保持全网的TDI=pdbase
        # portTDI = min(s["period"] for s in streams_info)
        
        # 最大跳数及对应流
        max_hops = -1
        stream_with_max_hops = None
        for s in streams_info:
            if s["hops"] > max_hops:
                max_hops = s["hops"]
                stream_with_max_hops = s
        
        # 计算第一部分 NTSTC：所有流占用 TS 数之和：portTDI / stream.period 累加
        ts_sum = 0.0
        for s in streams_info:
            # 为了避免除以0，这里认为 period 必须大于0
            if s["period"] > 0:
                ts_sum += portTDI / s["period"]
        
        # 计算第二部分 NTSTC：最大跳数对应流的 TS 数补偿
        # stream_with_max_hops 中 period 用于计算
        extra_ts = 0.0
        if stream_with_max_hops and stream_with_max_hops["period"] > 0:
            extra_ts = max_hops / (stream_with_max_hops["period"] / portTDI)
        
        
        NTSTC = int(np.ceil(ts_sum + extra_ts))
        TCI = NTSTC*TS
        
        used_ports_list.append({
            "port_name": port,
            "streams_on_this_port": streams_on_this_port,
            "streamsNum": streamsNum,
            "portTDI": portTDI,
            "max_hops_of_these_streams": max_hops,
            "stream_with_max_hops": stream_with_max_hops["name"] if stream_with_max_hops else None,
            "NTSTC": NTSTC,
            "TCI": TCI 
        })
        
    # 构造最顶层的输出字典
    used_ports_data = {
        "name": "Ports_data_streams",
        "PortsNum": len(used_ports_list),
        "Used_ports": used_ports_list
    }
    return used_ports_data
# ---------------------step2 NTSTC计算结束--------------------------------
