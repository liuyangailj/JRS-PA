import json
import networkx as nx
import matplotlib.pyplot as plt
import math
from copy import deepcopy

# ---------- 定义流属性 ----------
class Stream:
    def __init__(self, name, stream_type, period, frames_per_period, max_frame_size, max_latency, deadline, talker, listeners, path, priority, earliest_offset, latest_offset):
        self.name = name
        self.stream_type = stream_type
        self.period = period
        self.frames_per_period = frames_per_period
        self.max_frame_size = max_frame_size
        self.max_latency = max_latency
        self.deadline = deadline
        self.talker = talker
        self.listeners = listeners
        self.path = path
        self.priority = priority
        self.earliest_offset = earliest_offset
        self.latest_offset = latest_offset
        
    def to_dict(self):
        return {
            'name': self.name,
            'stream_type': self.stream_type,
            'period': self.period,
            'frames_per_period': self.frames_per_period,
            'max_frame_size': self.max_frame_size,
            'max_latency': self.max_latency,
            'deadline': self.deadline,
            'talker': self.talker,
            'listeners': self.listeners,
            'path': self.path,
            'priority': self.priority,
            'earliestOffset': self.earliest_offset,
            'latestOffset': self.latest_offset
        }

# ---------- 数据加载函数 ----------
# 模拟读取输入数据文件，文件内需包含：网络拓扑、N flows、初始候选路由数 K 参数
def load_json(file_path):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Invalid JSON file: {file_path}")
        return None
    
# ---------- 数据提取函数 ----------
def extract_network_data(data):
        
    nodes = {node['name']: node for node in data['nodes']}
    links = data['links/ports']
    streams = data['streams']
    return nodes, links, streams

# ---------- 图构建函数 ----------
def build_graph(nodes, links):
    
    G = nx.DiGraph()
    
    # Add nodes
    for node_name, node_info in nodes.items():
        G.add_node(node_name, **node_info)
    
    # Add edges
    for link in links:
        G.add_edge(link['source'], link['destination'], **link)
    
    return G

# ---------- 可视化函数 ----------
def draw_topology(G, title="Network Topology"):
    
    # Choose a layout for the graph
    pos = nx.spectral_layout(G)    
    # Draw network
    nx.draw(
        G, pos, with_labels=True, node_size=500, 
        node_color='skyblue', font_size=10, font_weight='bold', arrows=True
    )    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'transmission_rate')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    # Display the graph
    plt.title(title)
    plt.show()
    
# ---------- 最短路径计算函数 ----------
def calculate_path_transmission_rate(G, path):
    
    total_rate = 0
    for i in range(len(path) - 1):
        edge_data = G.get_edge_data(path[i], path[i + 1])
        total_rate += edge_data.get('transmission_rate', 0)
    return total_rate

# ---------- 路由路径表示方式转化（节点-边） ----------    
def convert_nodes_to_links(G, path):
    
    link_path = []
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        edge_data = G.get_edge_data(src, dst)
        
        if edge_data:
            link_name = edge_data.get('name')
            if link_name:
                link_path.append(link_name)
            else:
                link_path.append(f"{src} -> {dst}")
        else:
            # 如果找不到对应的边，记录为未定义的链路
            link_path.append(f"Ubdefinded link: {src} -> {dst}")
    return link_path

# ---------- K条最短路径计算函数 ----------
def find_k_shortest_paths(G, streams, k=4):
    
    # Compute K shortest paths for each stream.

    k_shortest_paths = {}
    no_path_info = []
    
    for stream in streams:
        talker = stream['talker']
        listeners = stream['listeners']
        stream_name = stream['name']
        k_shortest_paths[stream_name] = {}

        for listener in listeners:
            # Find k shortest paths
            try:
                paths = list(nx.shortest_simple_paths(G, source=talker, target=listener, weight='weight'))
                k_shortest_paths[stream_name][listener] = [

                    {'path': path, 'total_transmission_rate': calculate_path_transmission_rate(G, path)}
                    for path in paths[:k]
                ]
            except nx.NetworkXNoPath:
                k_shortest_paths[stream_name][listener] = []  # No path found
                no_path_info.append((stream_name, listener))
                
    if no_path_info:
        print("No path found for the following streams and listeners:")
        for stream_name, listener in no_path_info:
            print(f"Stream: {stream_name}, Listener: {listener}")

    return k_shortest_paths

# ---------- 结果输出函数 ----------
def print_k_shortest_paths(k_shortest_paths):
   
    for stream_name, listener_paths in k_shortest_paths.items():
        print(f"Stream: {stream_name}")
        for listener, paths in listener_paths.items():
            print(f"  Listener: {listener}")
            for idx, path_info in enumerate(paths, start=1):
                path = path_info['path']
                total_rate = path_info['total_transmission_rate']
                print(f"    Path {idx}: {path} (Total Transmission Rate: {total_rate})")

# ---------- 结果保存函数 ----------
def save_k_shortest_paths(k_shortest_paths, output_file = "k_shortest_paths.json"):
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(k_shortest_paths, file, indent=4)
    print(f"K shortest paths saved to: {output_file}")




# 解析输入文件数据，

# ---------- 路由算法相关函数 ----------

def compute_Diphy(route, flow):
    """
    根据公式 Eq.(1) 计算 Di_phy，表示从始端到终端的传输时延
    这里假设 route 是包含链路时延的列表，flow 包含相关消息参数
    """
    # 例如：简单相加各链路传输延时
    return sum(link['delay'] for link in route)

def is_valid_route(route, flow):
    """
    检查路由是否有效：当 Di_phy 小于流的最大允许延时 φ 时才有效
    """
    Di_phy = compute_Diphy(route, flow)
    return Di_phy <= flow['phi']

def generate_candidate_routes(G, streams, K):
    """
    生成流 flow 在网络 network 中的候选路由，
    这里直接采用networkx自带求k-shortest path，实际中可调用如 Zoobi’s sidetrack-based variant algorithm.
    """
    k_shortest_paths = {}
    no_path_info = []
    
    for stream in streams:
        talker = stream['talker']
        listeners = stream['listeners']
        stream_name = stream['name']
        k_shortest_paths[stream_name] = {}

        for listener in listeners:
            # Find k shortest paths
            try:
                paths = list(nx.shortest_simple_paths(G, source=talker, target=listener, weight='weight'))
                k_shortest_paths[stream_name][listener] = {
                    f"path{idx+1}": {'path': path}
                    for idx, path in enumerate(paths[:k])
                    }
                
            except nx.NetworkXNoPath:
                k_shortest_paths[stream_name][listener] = [] # No path found
                no_path_info.append((stream_name, listener))
    if no_path_info:
        print("No path found for the following streams and listeners:")
        for stream_name, listener in no_path_info:
            print(f"Stream: {stream_name}, Listener: {listener}")
    return k_shortest_paths
                        
    # 示例：假设 network['paths'][flow['id']]中已包含预计算的候选路由列表
    candidate_routes = network.get('paths', {}).get(str(flow['id']), [])
    # 只返回较前K条候选路由
    return candidate_routes[:K]

# ---------- 最优路径选择算法 ----------

def optimal_routing(V_Ri, flow, network_links, weights=(0.5, 0.5)):
    """
    对流 flow 的有效路由集合 V_Ri 选择最优路由
    weights 对应 ω1 与 ω2, 计算公式参考 Eq.(4)
    network_links 是当前链路带宽信息，用于计算残留带宽
    """
    if not V_Ri:
        return None

    # 计算所有候选路由的 Di_phy 和残留带宽（最小链路带宽）
    metrics = []
    for route in V_Ri:
        Di_phy = compute_Diphy(route['links'], flow)
        # Rj.B 取路由上所有链路的残余带宽的最小值
        Rj_B = min(link['residual_bw'] for link in route['links'])
        metrics.append({'route': route, 'Di_phy': Di_phy, 'B': Rj_B})
    
    # 找到所有路由中最小的 Di_phy 和最大的残留带宽
    min_Di_phy = min(m['Di_phy'] for m in metrics)
    max_B = max(m['B'] for m in metrics)
    
    # 计算综合评价函数
    for m in metrics:
        # 防止除零错误
        delay_ratio = (min_Di_phy / m['Di_phy']) if m['Di_phy'] > 0 else 1.0
        bw_ratio = m['B'] / max_B if max_B > 0 else 1.0
        m['func'] = weights[0] * delay_ratio + weights[1] * bw_ratio

    # 选择函数值最大的路由作为最优路由
    best = max(metrics, key=lambda x: x['func'])
    # 更新经过链路的残余带宽，假设消耗一个固定量（此处简化处理）
    for link in best['route']['links']:
        link['residual_bw'] -= flow.get('bw_requirement', 0)
    return best['route']

def joint_routing_scheduling(G, streams, initial_K):
    """
    联合路由和调度算法
    network: 包含网络拓扑，预先计算的候选路由等数据结构
    flows: 流列表，每个流包含 phi（最大允许延时）及其它参数
    initial_K: 初始候选路由数量
    """
    K = initial_K
    # 为每个流初始化有效路由集合
    valid_routes = {stream['id']: [] for stream in streams}
    old_valid_routes = {stream['id']: [] for stream in streams}

    candidate_routes = {}
    # 按 max_latency 升序排序流（保证先处理时延最紧的流） 
    streams_sorted = sorted(streams, key=lambda f: f['max_letancy'])  # 此处的streams应当为列表 
    
    while True:
        # 对每个流，计算候选路由并筛选有效路由
        for flow in streams_sorted:
            cand = generate_candidate_routes(G, streams,  K)
            candidate_routes[flow['id']] = cand
            valid_routes[flow['id']] = []
            for route in cand:
                if is_valid_route(route['links'], flow):
                    valid_routes[flow['id']].append(route)
        
        # 针对时延最紧的流（第一个流），选取其有效路由
        first_flow = flows_sorted[0]
        V_Ri = valid_routes[first_flow['id']]
        # 如果没有有效路由，则调度失败，退出或做其他处理
        if not V_Ri:
            print(f"流 {first_flow['id']} 无有效路由，调度失败")
            return None
        
        # 对最紧流选取最优路由
        opt_route = optimal_routing(V_Ri, first_flow, network.get('links', {}))
        if not opt_route:
            print(f"流 {first_flow['id']} 无法选出最优路由")
            return None
        
        # 对其他流，依次选择最优路由
        optimal_routes = {first_flow['id']: opt_route}
        scheduling_possible = True
        for flow in flows_sorted[1:]:
            V_Ri = valid_routes[flow['id']]
            # 若无有效路由，则标记此轮调度失败
            if not V_Ri:
                scheduling_possible = False
                break
            opt = optimal_routing(V_Ri, flow, network.get('links', {}))
            if opt is None:
                scheduling_possible = False
                break
            optimal_routes[flow['id']] = opt

        # 尝试进行调度，根据调度算法（下面函数调用simulate_scheduling）
        schedule = simulate_scheduling(optimal_routes, flows, network)
        if schedule is not None:
            print("调度成功")
            return optimal_routes, schedule
        else:
            # 若调度失败，对第一流的有效路由进行下一候选尝试
            # 或若所有路由都已尝试，则增加 K 并更新旧有效路由集合
            if valid_routes[first_flow['id']] != old_valid_routes[first_flow['id']]:
                old_valid_routes = deepcopy(valid_routes)
                # 此处可以尝试更换第一流的候选路由
                # 重新循环以选取下一条路由
                print("调度失败，尝试更换最紧流的候选路由")
                continue
            else:
                # 无新候选路由，则增加K后重试
                K += 1
                old_valid_routes = deepcopy(valid_routes)
                print(f"调度失败，增加候选数 K 至 {K}")
                continue

# ---------- 调度算法相关函数 ----------

def simulate_scheduling(optimal_routes, flows, network):
    """
    模拟调度算法，依据 Algorithm 3 的伪代码
    主要步骤：
      - 使用公式 Eq.(8)确定时间分片 TDI = pdbase（pdbase为 flows 的最小周期，公式 Eq.(5)）
      - 依据公式 Eq.(9)计算 TS
      - 根据每个流的周期确定 TSAI（公式 Eq.(10)）和调度周期 TSAI_max（公式 Eq.(11)）
      - 分配各流的时间槽（简单模拟，不做复杂时隙冲突检测）
    返回调度结果（例如一个字典描述各流的时间槽分配）
    """
    # 获取 flows 周期列表
    periods = [f['period'] for f in flows if 'period' in f]
    if not periods:
        print("无有效周期信息")
        return None
    pdbase = min(periods)  # 基本周期 [T42](8)
    # 设定 TDI = pdbase
    TDI = pdbase
    # 计算 TS，假设 ts = dtrans + dprop + dproc, 此处简单取固定值
    TS = network.get('TS_fixed', 1)
    
    # TSAI_i = pdi, TSAI_max 为所有 TSAI 的最小公倍数（此处简化取 max）
    scheduling = {}
    for flow in flows:
        TSAI_i = flow['period']
        # 假设每个流在 TDI 内需要 n_slots 个 TS, n_slots = TDI // TSAI_i
        n_slots = TDI // TSAI_i
        scheduling[flow['id']] = {
            'TDI': TDI,
            'TS': TS,
            'allocated_slots': n_slots,
            'optimal_route': optimal_routes.get(flow['id'], None)
        }
    # 此处再根据具体限制（如避免时隙冲突）做进一步优化
    # 如果发现冲突或不满足调度要求，则返回 None
    return scheduling

# ---------- 主函数 ----------

def main(file_path, k=4):
    # Step1: 加载JSON数据
    data = load_json(file_path)
    if not data:
        print("加载数据失败，程序退出。")
        return
    
    # Step2: 提取网络信息
    nodes, links, streams = extract_network_data(data)
    
    # Step3: 创建图
    G = build_graph(nodes, links)
    
    # Step4: 绘制网络拓扑
    draw_topology(G)
    
    streams = []
    for stream_data in data['streams']:
        stream = Stream(
            name=stream_data['name'],
            stream_type=stream_data['type'],
            period=stream_data['period'],
            frames_per_period=stream_data['framesPerPeriod'],
            max_frame_size=stream_data['maxFrameSize'],
            max_latency=stream_data['maxLatency'],
            deadline=stream_data['deadline'],
            talker=stream_data['talker'],
            listeners=stream_data['listeners'],
            path=stream_data['path'],
            priority=stream_data['priority'],
            earliest_offset=stream_data['earliestOffset'],
            latest_offset=stream_data['latestOffset']
        )
        
    # 计算每个流的K条最短路径
    
    for listener in stream.listeners:
        try:                
            paths = list(nx.shortest_simple_paths(G, source=stream.talker, 
                                                target=listener, weight='weight'))
            selected_paths = paths[:k] # 选择前K条最短路径
            
            for path in selected_paths:
                link_path = convert_nodes_to_links(G, path) 
                stream.path.append(link_path)
        except nx.NetworkXNoPath:
            print(f"No path found between {stream.talker} and {listener}")
    streams.append(stream)
        
    # Save the output to a JSON file
    output_data = {
        "streams": [stream.to_dict() for stream in streams]
    }
    with open('../data/output/output.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print("路径计算完成，结果已保存到 output.json")    

    # 假设数据中包含 network, flows, initial_K 的字段
    network = data.get('network', {})
    flows = data.get('flows', [])
    initial_K = data.get('K', 3)

    result = joint_routing_scheduling(network, flows, initial_K)
    if result is not None:
        optimal_routes, schedule = result
        print("最优路由选择结果:")
        print(json.dumps(optimal_routes, indent=2, ensure_ascii=False))
        print("调度结果:")
        print(json.dumps(schedule, indent=2, ensure_ascii=False))
    else:
        print("联合路由调度失败")

if __name__ == "__main__":
    main()
