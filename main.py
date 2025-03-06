# ---------------------------------------------------------------
# 文件：main.py
# ---------------------------------------------------------------
import logging
import time

from src.data_manager import DataManager
from src.router import Router
from src.scheduler import Scheduler
from src.sort_streams import sort_streams
from src.draw_topology import draw_topology
from src.ts_allocation_gantt import plot_gantt_chart

def main():
    # input_file = "./data/input/bridge3_es9_line_example.json" 
    # input_file = "./data/input/test_1.json" 
    input_file = "./data/input/ring_multipath.json"
    
    output_file = "./data/output/allocated_ts_output.json"
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)    
    
    # 实例化类
    data_manager = DataManager(input_file)
        
    # 加载数据
    input_data = data_manager.load_data()
    
    # 构建网络拓扑图
    graph = data_manager.build_graph()
    
    # 绘制拓扑图以便确认拓扑信息
    # draw_topology(graph)
    
    # 开始计时：
    start_time = time.perf_counter()
    
    # 计算候选k条路径（例如 k=3）
    router = Router(graph)
    find_k_path_data = router.compute_k_shortest_paths(input_data, k=3)
    # 计算流的物理延迟，筛选出有效路径
    valid_path_data = router.get_stream_phy_delay_on_path(find_k_path_data)   
    # 排序
    sorted_streams = sort_streams(valid_path_data)
    # 选最优路径
    select_optimal_routes_data = router.select_optimal_routes(sorted_streams)  
    
    # 转换流中的路由表示
    converted_stream_routes = data_manager.convert_stream_routes(select_optimal_routes_data)
    
    # 实例化调度
    scheduler = Scheduler(select_optimal_routes_data)
    # # 计算时间调度参数
    # TDI, TS, GBI = scheduler.calculate_tdi_and_ts(converted_stream_routes)
    
    # 初始化端口分配数据 (此处示例假设端口列表来源于 JSON 或预设)
    used_ports_data = scheduler.calculate_tsai_and_ntstc(converted_stream_routes)
    allocated_streams = scheduler.allocate_time_slots(converted_stream_routes, used_ports_data)
    
    # 结束计时
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"代码执行时间（allocate_time_slots 前后的代码块）: {elapsed_time:.6f} 秒")
    
    # # 推导 GCL
    # data = derive_gcl.derive_gcl(data)
    
    # # 对流进行排序
    # data = stream_sort.sort_streams(data)    
    
    # 保存最终结果
    data_manager.save_data(output_file, allocated_streams)    
    
    # 绘制结果gantt图
    # plot_gantt_chart(allocated_streams, used_ports_data, TSAI_max, TDI)
    TDI, TS, GBI, TSAI_max = scheduler.calculate_tdi_and_ts(converted_stream_routes)
    plot_gantt_chart(allocated_streams, used_ports_data, TSAI_max, TDI, TS)

if __name__ == "__main__":
    main()
