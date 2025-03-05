# test_scheduler.py

import pytest
from src.scheduler import Scheduler  # 假定 Scheduler 类定义在 scheduler.py 模块中

# 示例网络数据，可根据项目实际情况修改
@pytest.fixture
def sample_network_data():
    return {
        "nodes": ["A", "B", "C"],
        "links": [
            {"from": "A", "to": "B", "capacity": 100},
            {"from": "B", "to": "C", "capacity": 150},
            {"from": "A", "to": "C", "capacity": 200},
        ],
        "traffic": [
            {"source": "A", "destination": "C", "demand": 80},
            {"source": "B", "destination": "C", "demand": 120},
        ]
    }

# 构建一个 Scheduler 对象的 fixture
@pytest.fixture
def scheduler(sample_network_data):
    sch = Scheduler(sample_network_data)
    sch.init_allocation()  # 假设初始化分配
    return sch

# 测试 calculate_tdi_and_ts 方法
def test_calculate_tdi_and_ts(scheduler):
    # 调用方法进行计算
    scheduler.calculate_tdi_and_ts()
    
    # 断言计算后的属性不为空或符合预期
    # 假定计算后，scheduler 对象中存在属性 tdi 和 ts
    assert hasattr(scheduler, "tdi"), "Scheduler 应该包含属性 tdi"
    assert hasattr(scheduler, "ts"), "Scheduler 应该包含属性 ts"
    # 可以加入具体值的断言，例如:
    # assert scheduler.tdi == expected_tdi_value
    # assert scheduler.ts == expected_ts_value

# 测试 calculate_tsai_and_ntstc 方法
def test_calculate_tsai_and_ntstc(scheduler):
    scheduler.calculate_tsai_and_ntstc()
    
    # 假设该方法会产生 tsai 和 ntstc 两个属性
    assert hasattr(scheduler, "tsai"), "Scheduler 应该包含属性 tsai"
    assert hasattr(scheduler, "ntstc"), "Scheduler 应该包含属性 ntstc"
    # 可以进一步检查值是否落在预期范围
    # assert scheduler.tsai > 0
    # assert scheduler.ntstc >= 0

# 测试 allocate_time_slots 方法
def test_allocate_time_slots(scheduler):
    scheduler.allocate_time_slots()
    
    # 假定 allocate_time_slots 会设置一个 allocation 属性，存储时隙分配信息
    assert hasattr(scheduler, "allocation"), "Scheduler 应该包含属性 allocation"
    # 验证 allocation 数据结构符合预期（例如非空字典）
    assert isinstance(scheduler.allocation, dict), "allocation 应该是字典类型"
    assert scheduler.allocation, "allocation 不应为空"

# 测试 derive_gcl 方法
def test_derive_gcl(scheduler):
    # 首先调用可能影响 derive_gcl 方法的其他计算
    scheduler.calculate_tdi_and_ts()
    scheduler.calculate_tsai_and_ntstc()
    scheduler.allocate_time_slots()
    
    gcl = scheduler.derive_gcl()
    # 假定 gcl 应该是一个列表或者特定数据结构
    assert isinstance(gcl, list), "GCL 应该是一个列表"
    assert len(gcl) > 0, "GCL 列表中应有至少一个元素"
    
    # 可以进一步检查第一个元素的结构是否符合预期
    # expected_item_keys = {"start_time", "end_time", "node"}
    # for key in expected_item_keys:
    #     assert key in gcl[0]

# 更多的测试用例可以根据 Scheduler 的业务逻辑添加
