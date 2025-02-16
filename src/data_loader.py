# src/data_loader.py
import json
import logging

logger = logging.getLogger(__name__)

def load_json(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"无效的JSON文件: {file_path}")
    return {}


def extract_network_data(data):
    
    nodes = {node['name']: node for node in data['nodes']}
    links = data['links/ports']
    streams = data['streams']
    return nodes, links, streams

def save_k_shortest_paths(k_shortest_paths, output_file = "k_shortest_paths.json"):
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(k_shortest_paths, file, indent=4)
    print(f"K shortest paths saved to: {output_file}")