"""
import logging

logger = logging.getLogger(__name__)

def sort_streams(data):
    
    # 对流数据进行排序:
    # - 首先按照 `period` 升序排列
    # - 如果 `period` 相同，则按流名 `name` 的字母序升序排列
    
    streams = data.get("streams", [])

    # 按规则排序
    sorted_streams = sorted(streams, key=lambda s: (s.get("period", 0), s.get("name", "")))

    # 更新排序后的顺序
    data["sorted_stream_order"] = [s.get("name", "") for s in sorted_streams]
    data["streams"] = sorted_streams
    logger.info("流数据排序完成")
    
    return data
"""
# --- 以上是原始代码 ---
import logging

logger = logging.getLogger(__name__)

class StreamSorter:
    def __init__(self, data):
        self.data = data

    def sort_by_period_then_name(self):
        """
        对流数据进行排序：
        - 首先按照 `period` 升序排列
        - 如果 `period` 相同，则按流名 `name` 的字母序升序排列
        """
        streams = self.data.get("streams", [])
        sorted_streams = sorted(streams, key=lambda s: (s.get("period", 0), s.get("name", "")))
        self.data["sorted_stream_order"] = [s.get("name", "") for s in sorted_streams]
        self.data["streams"] = sorted_streams
        logger.info("流数据按 `period` 和 `name` 排序完成")
        return self.data

    def sort_by_name_length(self):
        """
        对流数据进行排序：
        - 按流名 `name` 的长度升序排列
        """
        streams = self.data.get("streams", [])
        sorted_streams = sorted(streams, key=lambda s: len(s.get("name", "")))
        self.data["sorted_stream_order"] = [s.get("name", "") for s in sorted_streams]
        self.data["streams"] = sorted_streams
        logger.info("流数据按流名长度排序完成")
        return self.data

    def sort_by_period_desc(self):
        """
        对流数据进行排序：
        - 按 `period` 降序排列
        """
        streams = self.data.get("streams", [])
        sorted_streams = sorted(streams, key=lambda s: s.get("period", 0), reverse=True)
        self.data["sorted_stream_order"] = [s.get("name", "") for s in sorted_streams]
        self.data["streams"] = sorted_streams
        logger.info("流数据按 `period` 降序排序完成")
        return self.data

    def sort_by_PDV(self):
        """
        对流数据进行排序：
        - 按 `PDV` 升序排列
        """
        streams = self.data.get("streams", [])
        sorted_streams = sorted(streams, key=lambda s: s.get("PDV", 0))
        self.data["sorted_stream_order"] = [s.get("name", "") for s in sorted_streams]
        self.data["streams"] = sorted_streams
        logger.info("流数据按 `PDV` 排序完成")
        return self.data