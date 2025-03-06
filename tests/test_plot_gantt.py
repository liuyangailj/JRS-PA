from src.ts_allocation_gantt import plot_gantt_chart

def main():
    # 模拟 allocation_result 数据
    allocation_result = {
        "流1": {
            "端口A": {"time_slot_position": [1, 0]},
            "端口B": {"time_slot_position": [3, 0]}
        },
        "流2": {
            "端口A": {"time_slot_position": [2, 0]},
            "端口C": {"time_slot_position": [4, 0]}
        }
    }
    
    # 模拟 used_ports_data 数据
    used_ports_data = {
        "Used_ports": [
            {"port_name": "端口A"},
            {"port_name": "端口B"},
            {"port_name": "端口C"}
        ]
    }
    
    # TS 相关参数
    TSAI_max = 5     # STDIN上界
    TDI = 2          # 每个STDIN对应2个TS
    TS = 1           # TS的比例因子

    # 执行绘图测试
    plot_gantt_chart(allocation_result, used_ports_data, TSAI_max, TDI, TS)
    
if __name__ == "__main__":
    main()

    