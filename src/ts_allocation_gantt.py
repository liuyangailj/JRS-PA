import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

def generate_random_color():
    """生成随机颜色"""
    return "#%06x" % random.randint(0, 0xFFFFFF)

def calculate_x(stdin, ssn, TDI, TS):
    """
    计算x坐标，单位统一表示成TS刻度
    x = (stdin - 1) * (TDI / TS) + ssn
    """
    return (stdin - 1) * (TDI / TS) + ssn-1

def plot_gantt_chart(allocation_result, used_ports_data, TSAI_max, TDI, TS):
    """
    绘制甘特图：
    - 纵坐标：端口
    - 横坐标：时间（以TS为最小单位，长度为TSAI_max）
    - 每个流在对应端口上的TS位置用不同颜色表示，占用1个TS

    Args:

    allocation_result (dict): 每个流在各端口上的分配结果
    used_ports_data (dict): 包含端口数据，至少包含"Used_ports"，格式如：
    {"Used_ports": [{"port_name": "port1", ...}, {"port_name": "port2", ...}, ...]}
    TSAI_max (int): STDIN 的上界，横坐标的最大值
    TDI (int): 时间槽间隔，用于横坐标刻度转换（假设1个STDIN对应 TDI 个TS，如果需要调整，可以修改此处）
    """
    """
    绘制Gantt图
    y轴表示端口，x轴表示时间（单位 TS），每个流使用不同颜色表示
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    # 为不同流生成颜色
    color_map = {}
    colors = plt.cm.tab20.colors
    for idx, stream in enumerate(allocation_result.keys()):
        color_map[stream] = colors[idx % len(colors)]

    # y 轴按照端口顺序排列
    ports = [port_info["port_name"] for port_info in used_ports_data["Used_ports"]]
    port_y = {port: idx for idx, port in enumerate(ports)}  # 端口对应y坐标

    # 遍历 allocation_result，为每个流每个端口绘制条形
    for stream, port_allocs in allocation_result.items():
        for port, alloc in port_allocs.items():
            stdin = alloc["time_slot_position"][0]
            ssn = alloc["time_slot_position"][1]
            # 计算x坐标（单位 TS）和条形宽度（固定 1 TS）
            x = calculate_x(stdin, ssn, TDI, TS)
            y = port_y.get(port, None)
            # 仅当在端口列表中才绘制
            if y is not None:
                ax.barh(y, width=1, left=x, height=0.5, color=color_map[stream], edgecolor='black')
                # 显示流名称在条上
                ax.text(x + 0.1, y, stream, va='center', fontsize=8, color='white')

    # 设置y轴刻度与标签
    ax.set_yticks(list(port_y.values()))
    ax.set_yticklabels(list(port_y.keys()))
    ax.set_xlabel(f"时间槽 (单位: TS)")
    ax.set_ylabel("端口")
    ax.set_title("TSN 流时间槽分配 Gantt 图")

    # # 绘图美化
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)
    # ax.set_xlim(0, (TSAI_max/ TS) + 1)  # 根据STDIN上界调整x轴范围

    # 创建图例
    patches = [mpatches.Patch(color=color_map[stream], label=stream) for stream in color_map]
    ax.legend(handles=patches, title="流名称", loc='center left',
          bbox_to_anchor=(1, 0.5), ncol=2)

    
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    plt.show()
