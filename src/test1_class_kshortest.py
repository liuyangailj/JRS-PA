import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
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
            
    def build_link_mapping(self, data):
        """从json数据中的 'links/ports' 构造链路映射字典.
        key为 (source.lower(), destination.lower())，
        value为对应的链路名称.
        """
        mapping = {}
        for link in data.get("links/ports", []):
            # 将source和destination都转为小写
            src = link.get("source", "").lower()
            dst = link.get("destination", "").lower()
            mapping[(src, dst)] = link.get("name")
        return mapping
        
    def convert_stream_routes(self, data):
        """
        遍历所有stream的path，将 route 节点顺序列表转化为链路
        表示形式，根据链路对应的 source/destination 值（转成小写）
        匹配link_mapping.转换前：
        "route": 
        ["ES8", 
        "Bridge3", 
        "Bridge4", 
        "Bridge5", 
        "ES14"]
        转换为：
        [
            "Link17-2/Port35",    # 对应 ("es8", "bridge3")
            "Link3-1/Port6",      # 对应 ("bridge3", "bridge4")
            "Link5-1/Port10",     # 对应 ("bridge4", "bridge5")
            "Link23-1/Port46"     # 对应 ("bridge5", "es14")
        ]
        """
        link_mapping = self.build_link_mapping(data)    
        # 遍历每个 stream
        for stream in data.get("streams", []):
            # 每个 stream 中可能包含多个 path
            for path in stream.get("path", []):
                route = path.get("route", [])
                new_route = []
                # 对于相邻节点对，查找对应的链路表示
                for i in range(len(route) - 1):
                    nodeA = route[i].lower()
                    nodeB = route[i+1].lower()
                    key = (nodeA, nodeB)
                    link_name = link_mapping.get(key)
                    if link_name:
                        new_route.append(link_name)
                    else:
                        # 若找不到对应链接，可以保留节点对或报错，此处选择报错信息
                        new_route.append(f"NotFound({nodeA}->{nodeB})")
                # 更新 path 中的 route 字段
                path["route"] = new_route        
        return data

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

# '''
# ------------------调度算法代码-------------------------------
import numpy as np
import math

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

    # 用于存储所有去重的端口，及端口对应的流信息
    used_ports = set()
    # 结构： { port: [ { "name": stream_name, "period": period, "hops": hops }, ... ] }
    port_to_streams = {}

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
        # portTDI 为该端口上所有流中 period 的最大值
        portTDI = min(s["period"] for s in streams_info)
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
        
        NTSTC = np.ceil(ts_sum + extra_ts)
        # NTSTC = ts_sum
        
        used_ports_list.append({
            "port_name": port,
            "streams_on_this_port": streams_on_this_port,
            "streamsNum": streamsNum,
            "portTDI": portTDI,
            "max_hops_of_these_streams": max_hops,
            "stream_with_max_hops": stream_with_max_hops["name"] if stream_with_max_hops else None,
            "NTSTC": NTSTC
        })
        
    # 构造最顶层的输出字典
    used_ports_data = {
        "name": "Used_Ports",
        "PortsNum": len(used_ports_list),
        "Used_ports": used_ports_list
    }
    return used_ports_data
# ---------------------step2 NTSTC计算结束--------------------------------


# ---------------------step 3调度--------------------------------
def schedule_flows(flow_periods, optimal_routes, hops, pdbase, TDI, TSAIs, NTSTCs):
    """
    Step 3: 为每个流调度分配时间槽
    """
    schedule = {}
    allocation_sequence = sorted(range(len(flow_periods)), key=lambda i: (hops[i], flow_periods[i]))  # 按跳数和周期排序

    for n, flow_id in enumerate(allocation_sequence):
        flow_schedule = []
        current_tdi = 0
        tdi_slots = []
        
        while current_tdi < TDI:
            timeslot_start = current_tdi
            for hop in range(hops[flow_id]):
                ts_entry = {
                    "flow": flow_id,
                    "hop": hop,
                    "start_time": timeslot_start,
                    "end_time": timeslot_start + TSAIs[flow_id]
                }
                flow_schedule.append(ts_entry)
                timeslot_start += TSAIs[flow_id]
            current_tdi += TSAIs[flow_id]
        
        schedule[flow_id] = flow_schedule

    return schedule


def derive_gcl(schedule, TDI, NTSTCs):
    """
    Step 4: 提取为TSN交换机生成GCL
    """
    gcls = {}
    for flow_id, flow_schedule in schedule.items():
        gcl = {
            "flow": flow_id,
            "time_slots": [],
        }
        for entry in flow_schedule:
            gcl["time_slots"].append({
                "hop": entry["hop"],
                "start": entry["start_time"],
                "end": entry["end_time"]
            })
        gcls[flow_id] = gcl
    return gcls


# -----------------调度算法函数结束--------------------------------
# '''

def main():
    # 执行代码使用
    # json_file = "../data/input/test_1.json"  # 请修改为实际的JSON文件路径
    json_file = "../data/input/bridge3_es9_line_example.json" 
    output_file = "../data/output/output.json"  # 输出文件路径
    
    test_output_file_1 = "../data/output/output_1.json"  # 输出文件路径 "k 最短路径 + 物理延迟"  
    test_output_file_2 = "../data/output/output_2.json"  # 输出文件路径 "排序"
    test_output_file_3 = "../data/output/output_3.json"  # 输出文件路径 "选择最佳路径"
    test_output_file_4 = "../data/output/output_4.json"  # 输出文件路径 "路径转化为链路"
    test_output_file_5 = "../data/output/output_5.json"  # 输出文件路径 "计算NTSTC"
    
    # # 调试代码使用路径
    # json_file = "./data/input/test_1.json"  # 请修改为实际的JSON文件路径
    # output_file = "./data/output/output.json"  # 输出文件路径
    
    # test_output_file_1 = "./data/output/output_1.json"  # 输出文件路径 "k 最短路径 + 物理延迟"  
    # test_output_file_2 = "./data/output/output_2.json"  # 输出文件路径 "排序"
    # test_output_file_3 = "./data/output/output_3.json"  # 输出文件路径 "选择最佳路径"
    # test_output_file_4 = "./data/output/output_4.json"  # 输出文件路径 "路径转化为链路"
    # test_output_file_5 = "./data/output/output_5.json"  # 输出文件路径 "计算NTSTC"
    
    
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
    
    # 路径转化为链路
    nt.convert_stream_routes(nt.data)
    nt.save_data(test_output_file_4)
    
    # 计算TDI, TS, GBI
    TDI, TS, GBI = calculate_tdi_and_ts(nt.data)
    print("TDI:", TDI, "TS:", TS, "GBI:", GBI)
    
    # 计算NTSTC
    output_data = calculate_tsai_and_ntstc(nt.data)
    # 输出到结果文件 output_5.json，可以调整路径    
    with open(test_output_file_5, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("结果已写入文件：", output_file)


    # # 保存处理后的数据
    nt.save_data(output_file)

    # 绘制网络拓扑 
    draw_topology(graph, layout="spectral")

if __name__ == '__main__':
    main()
