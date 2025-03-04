# ---------------------step 3调度 分配TS--------------------------------

'''
算法描述：
1. 对于每个流 stream：
   a. 遍历其路由上的所有端口（for port in stream.route）。
   b. 若是第一个端口，则：
      i. 遍历 STDIN 从 1 到 TASImax/TDI（注意 TDI 为端口单独的时间间隔）：
          - 计算 RTS = NTSTC - SSN 。
          - 如果满足 1 ≤ RTS < NTSTC（即当前 SSN 处有足够槽位供流分配），则将当前 STDIN 和 SSN 分配给流在该端口上的时间槽，并将 Used_TS_allocation[STDIN][SSN] 更新为 True，同时将 SSN 自增 1 。
          - 如果 RTS 小于 1（槽位不足），则 STDIN 自增 1，SSN 重置为 1，然后重新判断（直到找到满足条件的位置或 STDIN 达到上限且 SSN 等于 NTSTC）。
      ii. 保存当前分配的 STDIN 和 SSN 作为该流在第一个端口的初始值 .
   c. 对于后续端口：
      i. 根据上一个端口的分配递推：
         - 若上一个端口的 SSN < NTSTC，则当前端口分配：STDIN 与上一个端口相同，而 SSN 更新为上一个 SSN + 1 ；
         - 如果上一个端口的 SSN 已经等于 NTSTC，则当前端口 STDIN 加 1，SSN 重新置为 1。
      ii. 将递推后的 STDIN 和 SSN 记录为当前端口的分配位置 .

下面是详细的代码设计：
--------------------------------------------------------------
'''

# 参数说明
# streams: 流集合，每个流有 route（端口列表）和其他属性
# TASImax: 最大时隙分配区间（对应 TSAImax）
# NTSTC: 每个端口的时间临界数据区间内槽位数（TS的总槽数）
# Used_TS_allocation: 字典或二维数组，记录每个 (STDIN, SSN) 是否已被使用

# 假设每个端口独立保存自己的 STDIN 和 SSN 初始值，即：
# 每个流在初始端口上的分配为 (port_allocation[port].STDIN, port_allocation[port].SSN)
# 后续端口的分配基于上一端口的结果递推
import logging
from src.scheduling.compute_scheduling_params import calculate_tdi_and_ts

logger = logging.getLogger(__name__)
def allocate_time_slots(data, used_ports_data):
    # 初始化一个矩阵记录所有端口每个 (STDIN, SSN) 是否被分配
    # 假设 STDIN 取值 1 到 TASImax_TDI，SSN取值 1 到 NTSTC
    # 这里表示为二维字典：Used_TS_allocation[port][STDIN][SSN] = True/False
    # 如果所有端口共用同一分配矩阵，则对应端口的实例分别保存分配信息
    used_TS_allocation = {}  # 用于每个端口存储分配情况， key为端口标识
   
    # 假设存在函数 init_allocation(port) 用于初始化某端口的 allocation 状态
    def init_allocation(port, NTSTC, TASI_max_TDI):
        if port not in used_TS_allocation:
            used_TS_allocation[port] = {}
            # STDIN 范围为 1 到 TASImax/TDI (上界值)
            for stdin in range(1, TASI_max_TDI+1):
                used_TS_allocation[port][stdin] = {}
                for ssn in range(1, NTSTC + 1):
                    used_TS_allocation[port][stdin][ssn] = False

    # 保存每个流每个端口的分配结果
    allocation_result = {}  # allocation_result[stream_id][port] = (STDIN, SSN)
    # 获取流数据信息
    streams = data.get("streams", [])
    if streams:
        TASI_max = max([stream.get("period", 0) for stream in streams if stream.get("period", 0) > 0])
    else:
        TASI_max = 0    
    
    # 调用函数获取 TDI 和 TS
    # # 计算 一个调度周期TASImax内的TDI数量
    TDI = calculate_tdi_and_ts(data)[0]    
    TASI_max_TDI = int(TASI_max / TDI) if TDI != 0 else 0
    
    # 保存每个流在每个端口的分配结果 allocation_result[stream_id][port] = (STDIN, SSN)
    allocation_result = {}
    
    

    # 对于每个流
    for stream in streams:
        stream_id = stream['name']
        stream_path = stream['path']
        allocation_result[stream_id] = {}
        ports_list = stream["path"][0]['route'] # 获取流经过的端口列表 stream.path.route

        # 处理第一个端口：逐个遍历 STDIN 寻找合适位置
        first_port = ports_list[0]
        for port in used_ports_data["Used_ports"]:
            if port["port_name"] == first_port:
                first_port_NTSTC = port["NTSTC"]
                break
        if first_port_NTSTC is None:
            raise Exception(f"端口 {first_port} 的 NTSTC 未找到！")
        
        init_allocation(first_port, first_port_NTSTC, TASI_max_TDI)
        allocated = False
        # 从 STDIN=1 开始检查，注意上界为 TASImax
        stdin = 1
        ssn = 1
        
        # 循环外层，直到找到合适槽位或遍历完所有可能位置
        while stdin <= TASI_max_TDI and not allocated:
            # 计算剩余槽位 RTS = NTSTC - 当前 SSN + 1 （因为 SSN 是当前未被分配槽位起点）
            RTS = first_port_NTSTC - ssn + 1
            # 判断条件：需要至少 1 个槽位（通常条件可以是1 ≤ RTS < NTSTC，但实际判断槽位是否足够）
            if RTS >= 1:
                # 若当前槽位未被分配，则分配该位置
                if not used_TS_allocation[first_port][stdin][ssn]:
                    allocation_result[stream_id][first_port] = (stdin, ssn)
                    used_TS_allocation[first_port][stdin][ssn] = True
                    allocated = True
                    # 更新 ssn 为下一槽位供下一次使用（如有需要后续再分配当前流在同一端口的其他 TS）
                    ssn += 1
                else:
                    # 若已使用，则 ssn 后移
                    ssn += 1
            else:
                # RTS 小于1时，说明当前 STDIN 已无足够槽位，转向下一个 STDIN，将 ssn 重置为 1
                stdin += 1
                ssn = 1
        if not allocated:
            raise Exception(f"流 {stream_id} 在第一个端口分配失败，请检查 TASImax 或 NTSTC 参数！")
        # 对于后续端口，采用递推方式分配
        # 注意每个端口都需要初始化其 Used_TS_allocation 状态
        previous_stdin, previous_ssn = allocation_result[stream_id][first_port]
        for i in range(1, len(ports_list)):
            curr_port = ports_list[i]
            curr_port_NTSTC = None
            for port in used_ports_data["Used_ports"]:
                if port["port_name"] == curr_port:
                    curr_port_NTSTC = port["NTSTC"]
                    break
            if curr_port_NTSTC is None: 
                raise Exception(f"端口 {curr_port} 的 NTSTC 未找到！")
            
            init_allocation(curr_port, curr_port_NTSTC, TASI_max_TDI)
            # 递推规则：如果上一端口的 ssn 小于 NTSTC，则当前端口的分配与上一端口相同 STDIN, SSN+1
            if previous_ssn < curr_port_NTSTC:
                curr_stdin = previous_stdin
                curr_ssn = previous_ssn + 1
            else:
                # 如果上一端口 ssn 已等于 NTSTC，则 STDIN 自增，SSN 重置为 1
                curr_stdin = previous_stdin + 1
                curr_ssn = 1
            # 检查当前 slot 是否已经被分配
            # 若已分配，则需要在当前端口内找第一个未分配槽位，类似第一个端口的逻辑
            allocated_curr = False
            temp_stdin = curr_stdin
            temp_ssn = curr_ssn
            
            while temp_stdin <= TASI_max_TDI and not allocated_curr:
                if not used_TS_allocation[curr_port][temp_stdin][temp_ssn]:
                    # 分配给当前端口
                    allocation_result[stream_id][curr_port] = (temp_stdin, temp_ssn)
                    used_TS_allocation[curr_port][temp_stdin][temp_ssn] = True
                    allocated_curr = True
                    # 更新递推变量以便下一端口使用
                    previous_stdin = temp_stdin
                    previous_ssn = temp_ssn
                else:
                    # 如果当前槽位被占用，则往后找
                    temp_ssn += 1
                    if temp_ssn > curr_port_NTSTC:
                        temp_stdin += 1
                        temp_ssn = 1
            if not allocated_curr:
                raise Exception(f"流 {stream_id} 在端口 {curr_port} 分配失败！")
    return allocation_result