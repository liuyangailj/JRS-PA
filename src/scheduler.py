import numpy as np
import json


class Scheduler:
    def __init__(self, data):
        self.data = data

    # ---------------------step 1 TDI计算开始--------------------------------

    def calculate_tdi_and_ts(self, data):
        """
        Step 1: 计算时间分段间隔 (TDI)和时间槽长度 (TS)
        
        参数: 
            data
            
        返回: 
            tuple: (TDI, TS, GBI)
        """
        
        # 获取节点中 isBridge 为 True 的处理延时
        processing_delay = None
        for node in data.get('nodes', []):
            if node.get('isBridge', False):
                processing_delay = node.get('processingDelay', 0)
                break
        if processing_delay is None:
            print("未找到 isBridge=True 的节点或 processingDelay 数据不可用")
            return None, None, None, None
        
        # 获取链路传输速率与传播延迟
        transmission_rate = None
        propagation_delay = None
        for link in data.get('links/ports', []):
            if 'transmissionRate' in link and 'propagationDelay' in link:
                transmission_rate = link['transmissionRate']
                propagation_delay = link['propagationDelay']
                break
        if transmission_rate is None or propagation_delay is None:
            print("未找到 transmissionRate 或 propagationDelay 数据")
            return None, None, None, None
        
        # 获取流的最大帧大小和每周期帧数
        streams = data.get('streams', [])
        if not streams:
            print("未找到流数据")
            return None, None, None, None
        first_stream = streams[0] # 仅获取第一个流的信息
        framesize = first_stream.get('maxFrameSize', 0)
        framesperperiod = first_stream.get('framesPerPeriod', 0)
        
        
        if transmission_rate == 0:
            print("transmission_rate 为 0，无法计算延时")
            return None, None, None, None
        
        # 计算传输延时 = framesize * framesperperiod / transmission_rate
        transmission_delay = framesize * framesperperiod / transmission_rate
        
        # 获取所有流周期，并找到最小值作为调度基准pdbase        
        periods = [stream.get('period', 0) for stream in streams]       
        pdbase = min(periods) # 根据公式 (5) 
        # 获取 TSAI_max        
        # TSAI_max = max([stream.get("period", 0) for stream in streams if stream.get("period", 0) > 0])     
        TSAI_max = max(periods) # 根据公式 (6)   
        TDI = pdbase  # 根据公式 (8)        
        GBI = 1542/ transmission_rate  # 根据公式 (9)        
        TS = transmission_delay + propagation_delay + processing_delay  # 根据公式 (10)        
        return TDI, TS, GBI, TSAI_max   

    # ---------------------step 2 NTSTC计算开始--------------------------------
    def calculate_tsai_and_ntstc(self, data):
        """
        Step 2: 计算调度所需相关参数
            1. TSAI (Time Slot Allocation Interval) 时间槽分配间隔
            2. NTSTC (Number of Time Slots to be Allocated) 需要分配的时间槽数量
            3. TCI (Time Slot Count Interval) 时间槽计数间隔
            4. NTCI (Number of Time Slot Count Intervals) 时间槽计数间隔数量
        
        参数：
            data

        返回：
            dict: 包含计算结果的used_ports_data数据结构
        """
        # 获取流数据和TDI、TS、参数
        streams = data.get("streams", [])
        TDI,TS,_,_ = self.calculate_tdi_and_ts(data)
        if TDI is None or TS is None:
            raise Exception("TDI 未能正确计算 或 TS 数据未找到！")

        used_ports = set()
        port_to_streams = {}
        portTDI = self.calculate_tdi_and_ts(data)[0]

        # 遍历所有 streams ，统计每个端口上的流信息和hop数
        for stream in streams:
            stream_name = stream.get("name")
            stream_period = stream.get("period")
            stream_paths = stream.get("path", [])
            # 对于每个流，可能有多个 path，每个 path 中的 route 端口列表
            for path in stream_paths:
                routes = path.get("route", [])
                # 计算该流在此条路由上的 hop 数：取 route 长度减 1
                hops = max(len(routes) - 1, 0)
                for port in routes:
                    port = port.strip()
                    used_ports.add(port)
                    if port not in port_to_streams:
                        port_to_streams[port] = []
                    # 保存当前流的信息到该端口
                    port_to_streams[port].append({
                        "name": stream_name,
                        "period": stream_period,
                        "hops": hops
                    })

        # 构造结果的 Used_ports 列表
        used_ports_list = []
        for port in used_ports:
            streams_info = port_to_streams.get(port, [])
            if not streams_info:
                continue

            streams_on_this_port = [s["name"] for s in streams_info]
            streamsNum = len(streams_info)
            # portTDI 为该端口上所有流中 period 的最大值, 在此处的计算中，TDI是否还是保持全网的TDI=pdbase
            # portTDI = min(s["period"] for s in streams_info)
            
            # 最大跳数及对应流
            max_hops = -1
            stream_with_max_hops = None
            for s in streams_info:
                if s["hops"] > max_hops:
                    max_hops = s["hops"]
                    stream_with_max_hops = s
            
            # 计算第一部分 NTSTC：所有流占用 TS 数之和：portTDI / stream.period 累加
            ts_sum = 0.0
            for s in streams_info:
                # 为了避免除以0，这里认为 period 必须大于0
                if s["period"] > 0:
                    ts_sum += portTDI / s["period"]
            
            # 计算第二部分 NTSTC：最大跳数对应流的 TS 数补偿
            # stream_with_max_hops 中 period 用于计算
            extra_ts = 0.0
            if stream_with_max_hops and stream_with_max_hops["period"] > 0:
                extra_ts = max_hops / (stream_with_max_hops["period"] / portTDI)
            
            
            NTSTC = int(np.ceil(ts_sum + extra_ts))
            TCI = NTSTC*TS
            
            used_ports_list.append({
                "port_name": port,
                "streams_on_this_port": streams_on_this_port,
                "streamsNum": streamsNum,
                "portTDI": portTDI,
                "max_hops_of_these_streams": max_hops,
                "stream_with_max_hops": stream_with_max_hops["name"] if stream_with_max_hops else None,
                "NTSTC": NTSTC,
                "TCI": TCI 
            })
            
        # 构造最顶层的输出字典
        used_ports_data = {
            "name": "Ports_data_streams",
            "PortsNum": len(used_ports_list),
            "Used_ports": used_ports_list
        }
        return used_ports_data

    # ---------------------step 3调度--------------------------------    
    def init_allocation(self, used_time_slot_allocation, port, NTSTC, TSAI_max_TDI):
        """初始化某个端口的时间槽分配状态

        Args:
            used_time_slot_allocation (dict): 记录端口分配状态的字典
            port (str): 端口标识
            NTSTC (int): 当前端口允许的槽位数上限
            TSAI_max_TDI (int): STDIN的上限值
        """
        if port not in used_time_slot_allocation:
            used_time_slot_allocation[port] = {}
            # STDIN 范围为 1 到 TASImax/TDI (上界值)
            for stdin in range(1, TSAI_max_TDI+1):
                used_time_slot_allocation[port][stdin] = {}
                for ssn in range(1, NTSTC + 1):
                    used_time_slot_allocation[port][stdin][ssn] = False

    def allocate_time_slots(self, data, used_ports_data):
        """
        Step 3: 分配时间槽
        
        算法说明：
            1. 第一个端口采用遍历查找空闲槽位进行分配。
            2. 后续端口则基于前一端口的分配结果递推分配槽位。
        参数：
            data 包含流数据，
            used_ports_data 包含个端口计算
        返回：
            dict: allocation_result: 每个流在每个端口的分配结果
        """
        if not used_ports_data:
            raise Exception("未找到端口分配数据！")
        
        # 初始化所有端口的分配状态记录，用于标记每个端口的 STDIN 和 SSN 是否已被使用
        used_time_slot_allocation = {}  # 用于每个端口存储分配情况， key为端口标识
        allocation_result = {}  # 保存每个流每个端口的分配结果
        streams = data.get("streams", []) # 获取流数据信息
        # 获取 TSAI_max
        TSAI_max = self.calculate_tdi_and_ts(data)[3]    
        
        # 获取 TDI 和 TS
        # # 计算 一个调度周期TASImax内的TDI数量
        TDI = self.calculate_tdi_and_ts(data)[0]
        if TDI == 0 or TDI is None:
            raise Exception("TDI 计算错误！")
        TSAI_max_TDI = int(TSAI_max / TDI) 
        
        # 对于每个流开始时间槽分配
        for stream in streams:
            stream_id = stream['name']
            # stream_path = stream['path']
            allocation_result[stream_id] = {}
            ports_list = stream["path"][0]['route'] # 获取流经过的端口列表 stream.path.route

            # 处理第一个端口：逐个遍历 STDIN 寻找合适位置
            first_port = ports_list[0]
            first_port_NTSTC = None
            for port in used_ports_data["Used_ports"]:
                if port["port_name"] == first_port:
                    first_port_NTSTC = port["NTSTC"]
                    break
            if first_port_NTSTC is None:
                raise Exception(f"端口 {first_port} 的 NTSTC 未找到！")
            
            # 初始化第一个端口的分配状态
            self.init_allocation(used_time_slot_allocation, first_port, first_port_NTSTC, TSAI_max_TDI)
            allocated = False
            # 从 STDIN=1 开始检查，注意上界为 TSAImax
            stdin = 1
            ssn = 1
            
            # 直到找到合适槽位或遍历完所有可能位置
            while stdin <= TSAI_max_TDI and not allocated:
                # 计算剩余槽位 RTS = NTSTC - 当前 SSN + 1 （因为 SSN 是当前未被分配槽位起点）
                RTS = first_port_NTSTC - ssn + 1
                # 判断条件：需要至少 1 个槽位（通常条件可以是1 ≤ RTS < NTSTC，但实际判断槽位是否足够）
                if RTS >= 1:
                    # 若当前槽位未被分配，则分配该位置
                    if not used_time_slot_allocation[first_port][stdin][ssn]:
                        allocation_result[stream_id][first_port] = {
                            "time_slot_position": (stdin, ssn),
                            "curr_port_NTSTC": first_port_NTSTC
                        }
                        used_time_slot_allocation[first_port][stdin][ssn] = True
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
            # 注意每个端口都需要初始化其 used_time_slot_allocation 状态
            # 递推规则：如果上一端口的 ssn 小于 NTSTC，则当前端口的分配与上一端口相同 STDIN, SSN+1
            # 如果上一端口 ssn 已等于 NTSTC，则 STDIN 自增，SSN 重置为 1
            previous_stdin = allocation_result[stream_id][first_port]["time_slot_position"][0]
            previous_ssn = allocation_result[stream_id][first_port]["time_slot_position"][1]

            for i in range(1, len(ports_list)):
                curr_port = ports_list[i]
                curr_port_NTSTC = None
                for port in used_ports_data["Used_ports"]:
                    if port["port_name"] == curr_port:
                        curr_port_NTSTC = port["NTSTC"]
                        break
                if curr_port_NTSTC is None: 
                    raise Exception(f"端口 {curr_port} 的 NTSTC 未找到！")
                
                self.init_allocation(used_time_slot_allocation, curr_port, curr_port_NTSTC, TSAI_max_TDI)
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
                
                while temp_stdin <= TSAI_max_TDI and not allocated_curr:
                    if not used_time_slot_allocation[curr_port][temp_stdin][temp_ssn]:
                        # 分配给当前端口
                        allocation_result[stream_id][curr_port] = {
                            "time_slot_position": (temp_stdin, temp_ssn),
                            "curr_port_NTSTC": curr_port_NTSTC
                        }
                        used_time_slot_allocation[curr_port][temp_stdin][temp_ssn] = True
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

    def derive_gcl(schedule, TDI, NTSTCs):
        """
        Step 4: 提取为TSN交换机生成GCL
        """
        gcls = {}
        for stream_id, flow_schedule in schedule.items():
            gcl = {
                "flow": stream_id,
                "time_slots": [],
            }
            for entry in flow_schedule:
                gcl["time_slots"].append({
                    "hop": entry["hop"],
                    "start": entry["start_time"],
                    "end": entry["end_time"]
                })
            gcls[stream_id] = gcl
        return gcls 

