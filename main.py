# main.py
import logging
from src import config
from src.data_loader import load_json
from src.topo_builder import build_graph
from src.k_shortest_paths import find_k_shortest_paths
# from src.visualizer import draw_topology
from src.stream_dict import Stream

def extract_network_data(data):
        
    nodes = {node['name']: node for node in data['nodes']}
    links = data['links/ports']
    streams = data['streams']
    return nodes, links, streams

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 加载配置
    file_path = config.INPUT_FILE_PATH
    k = config.DEFAULT_K
    
    # 步骤1：加载JSON数据
    data = load_json(file_path)
    if not data:
        logger.error("加载数据失败，程序退出。")
        return
    
    # 步骤2：提取网络信息
    nodes, links, streams_data = extract_network_data(data)
    
    # 步骤3：创建图
    G = build_graph(nodes, links)
    
    # 步骤4：绘制网络拓扑
    draw_topology(G)
    
    # 步骤5：计算K条最短路径
    k_shortest_paths = find_k_shortest_paths(G, streams_data, k)
    
    # 步骤6：保存结果
    save_k_shortest_paths(k_shortest_paths, config.OUTPUT_FILE_PATH)
    logger.info("路径计算完成，结果已保存。")

if __name__ == "__main__":
    main()
