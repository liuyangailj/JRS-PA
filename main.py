# ---------------------------------------------------------------
# 文件：main.py
# ---------------------------------------------------------------
import logging
from src.data_handler import json_handler
from src.graph import topology
from src.routing.route_conversion import convert_stream_routes
from src.routing.k_shortest_paths import compute_k_shortest_paths
from src.routing.optimal_routing import optimal_routing # 假设 optimal_routing 在 src/routing 目录下
from src.scheduling.compute_scheduling_params import calculate_tdi_and_ts
from src.scheduling.allocate_time_slot import allocate_time_slots
from src.scheduling import derive_gcl # 如果你需要这个模块
from src.sorting import sort_stream


def main():
    input_file = "./data/input/bridge3_es9_line_example.json" 
    output_file = "./data/output/output.json"
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 加载数据
    data = json_handler.read_json(input_file)
    
    # 构建网络拓扑图
    graph = topology.build_graph(data)
    
    # 转换流中的路由表示
    data = convert_stream_routes(data)
    
    # 计算最短路径（例如 k=3）
    compute_k_shortest_paths(graph, data, k=3)
    
    # 对流进行排序
    data = sort_stream.sort_streams(data)
    
    # 计算流的物理延迟，筛选出有效路径
    # 选出最优路径
    
    optimal_routing(data)
    
    
    # 计算时间调度参数
    data = calculate_tdi_and_ts(data)
    
    # 初始化端口分配数据 (此处示例假设端口列表来源于 JSON 或预设)
    used_ports_data = calculate_tsai_and_ntstc(data)
    data = allocate_time_slots(data, used_ports_data)
    
    # # 推导 GCL
    # data = derive_gcl.derive_gcl(data)
    
    # # 对流进行排序
    # data = stream_sort.sort_streams(data)
    
    # 绘制拓扑图以便确认拓扑信息
    topology.draw_topology(graph, title="Network Topology")
    
    # 保存最终结果
    json_handler.write_json(data, output_file)

if __name__ == "__main__":
    main()
