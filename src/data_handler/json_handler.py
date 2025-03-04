import json
import logging
logger = logging.getLogger(__name__)
def read_json(input_file_path: str) -> dict:
    '''
    加载json数据。
    :param json_file: JSON 文件路径
    :return: 加载后的数据字典
    :raises: FileNotFoundError, json.JSONDecodeError
    '''
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info("成功加载 JSON 数据: %s", input_file_path)
        return data
    except Exception as e:
        logger.error("加载 JSON 数据失败: %s", e)
        raise

def write_json(data: dict, output_file_path: str) -> None:
    '''
    写入json数据。
    :param data: 待写入的数据字典
    :param json_file: JSON 文件路径
    :raises: FileNotFoundError, json.JSONDecodeError
    '''
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info("成功写入 JSON 数据: %s", output_file_path)
    except Exception as e:
        logger.error("写入 JSON 数据失败: %s", e)
        raise
    
   