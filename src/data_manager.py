import networkx as nx
import json

class DataManager:
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.data = None

    def load_data(self):
        """加载JSON数据，仅读取一次文件内容"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return self.data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"加载JSON数据失败: {e}")
        return None         

    def save_data(self, output_file, data):
        """将处理后的数据保存到指定文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            print(f"保存数据失败: {e}")

    def build_link_mapping(self, data)->dict:
        """从JSON数据中的 'links/ports' 构造链路映射字典"""
        mapping = {}
        for link in data.get("links/ports", []):
            src = link.get("source", "").lower()
            dst = link.get("destination", "").lower()
            mapping[(src, dst)] = link.get("name")
        return mapping

    def convert_stream_routes(self,data):
        """遍历所有stream的path，将 route 节点顺序列表转化为链路表示形式"""
        link_mapping = self.build_link_mapping(data)
        for stream in data.get("streams", []):
            for path in stream.get("path", []):
                route = path.get("route", [])
                new_route = []
                for i in range(len(route) - 1):
                    nodeA = route[i].lower()
                    nodeB = route[i+1].lower()
                    key = (nodeA, nodeB)
                    link_name = link_mapping.get(key)
                    if link_name:
                        new_route.append(link_name)
                    else:
                        new_route.append(f"NotFound({nodeA}->{nodeB})")
                path["route"] = new_route
        return data

    def build_graph(self) -> nx.Graph:
        """构建网络拓扑图"""
        graph = nx.Graph()
        if 'nodes' in self.data:
            for node in self.data['nodes']:
                graph.add_node(node['name'], **node)
        else:
            print("JSON数据中未找到'nodes'字段")
        if 'links/ports' in self.data:
            for link in self.data['links/ports']:
                src = link['source']
                dst = link['destination']
                weight = link.get('weight', 1)
                graph.add_edge(src, dst, weight=weight, **link)
        else:
            print("JSON数据中未找到'links/ports'字段")
        return graph
