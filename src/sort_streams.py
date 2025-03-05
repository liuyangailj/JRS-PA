import logging

logger = logging.getLogger(__name__)

def sort_streams(data):
    """
    对流数据进行排序:
    - 首先按照 `period` 升序排列
    - 如果 `period` 相同，则按流名 `name` 的字母序升序排列
    """
    streams = data.get("streams", [])

    # 按规则排序
    sorted_streams = sorted(streams, key=lambda s: (s.get("period", 0), s.get("name", "")))

    # 更新排序后的顺序
    data["sorted_stream_order"] = [s.get("name", "") for s in sorted_streams]
    data["streams"] = sorted_streams
    logger.info("流数据排序完成")
    
    return data