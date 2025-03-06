import json
import re

def parse_inner_json(content):
    """
    因为原始数据中 "content" 部分是一个包含换行符和转义字符的字符串，
    如果可以保证其内容为合法 JSON，则直接转换。
    """
    try:
        return json.loads(content)
    except Exception as e:
        print("解析内部 JSON 失败:", e)
        return None

def check_and_convert(links_ports):
    """
    传入 links_ports 为包含所有链接项的列表，其中每个项应为字典，
    如 {"name": "Link0-1/Port0", "source": "...", "destination": "...", ...}.
    """
    # 1. 按 name 匹配 Link 和 Port 部分
    link_pattern = re.compile(r'^Link(?P<link_num>\d+)-(?P<link_seq>\d+)/Port(?P<port_num>\d+)$')
    
    # 按 Link 编号分组，每个 Link 对应成对的数据记录
    link_groups = {}
    port_numbers = []
    for item in links_ports:
        name = item.get("name", "")
        m = link_pattern.match(name)
        if not m:
            print("名称格式错误:", name)
            continue
        link_num = int(m.group("link_num"))
        link_seq = int(m.group("link_seq"))
        port_num = int(m.group("port_num"))
        port_numbers.append(port_num)
        
        group = link_groups.setdefault(link_num, [])
        group.append((link_seq, item))
    
    # 检查 Port 序号连续从 0 开始
    if port_numbers:
        port_numbers.sort()
        for expected, num in enumerate(port_numbers):
            if expected != num:
                raise ValueError(f"Port 序号错误，预期 {expected} 但发现 {num}")
    
    # 检查每个 Link 组是否为成对数据，以及 source 和 destination 互为对调
    for link_num, items in link_groups.items():
        if len(items) != 2:
            raise ValueError(f"Link{link_num} 不存在成对记录")
        # 对比两条记录
        record1 = items[0][1]
        record2 = items[1][1]
        if not (record1["source"] == record2["destination"] and record1["destination"] == record2["source"]):
            raise ValueError(f"Link{link_num} 的 source/destination 未互为对调")
    
    # 若检查无误, 修改数字格式：统一从1开始
    # 说明：Link 序号原始从0开始，需加1；Link 后缀修改为 -0 和 -1：将原先 link_seq 按大小排序后重置为 0,1；
    # Port 序号同理，加 1.
    new_links_ports = []
    for link_num, items in link_groups.items():
        new_link_num = link_num + 1
        # 对当前 Link 对内按原有 link_seq 排序
        items.sort(key=lambda x: x[0])
        for new_seq, (_, item) in enumerate(items):
            m = link_pattern.match(item["name"])
            if m:
                new_name = f"Link{new_link_num}-{new_seq}/Port{int(m.group('port_num'))+1}"
                item["name"] = new_name
            new_links_ports.append(item)
    
    # 同时将所有孤立的 Port 名称（如果有单独 Port 记录）也加 1（这里假设仅存在与 Link 绑定的）
    return new_links_ports

def main():
    # 假设 JSON 文件是一个数组，每个元素含有 field "content" 内部嵌入
    network_data_file = "../data/input/test_1.json"
    with open(network_data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 假设所有记录中包含 "links/ports" 信息已经被整合成一个列表
    links_ports = data.get("links/ports", [])
    # for link_port in data.get("links/ports", []):
    #     # 如果 entry["content"] 本身是 JSON 字符串，则解析之
    #     # inner = parse_inner_json(link_port["content"])
    #     if link_port is None:
    #         continue
    #     # 如果内部 JSON 本身为字典，且包含 "links/ports" 键
    #     if "links/ports" in inner:
    #         links_ports.extend(inner["links/ports"])
    #     else:
    #         # 当 entry["content"] 直接为一个链接项，则直接加入
    #         links_ports.append(inner)
    
    try:
        new_links_ports = check_and_convert(links_ports)
        print("所有检查都通过，转换后的数据如下：")
        print(json.dumps(new_links_ports, indent=2, ensure_ascii=False))
    except Exception as e:
        print("检查失败:", e)

if __name__ == "__main__":
    main()
