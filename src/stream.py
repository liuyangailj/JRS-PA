# src/stream.py
from typing import List

class Stream:
    def __init__(
        self, 
        name: str, 
        stream_type: str, 
        period: float, 
        frames_per_period: int, 
        max_frame_size: int, 
        max_latency: float, 
        deadline: float, 
        talker: str, 
        listeners: List[str], 
        path: List[str], 
        priority: int, 
        earliest_offset: int, 
        latest_offset: int
    ):
        self.name = name
        self.stream_type = stream_type
        self.period = period
        self.frames_per_period = frames_per_period
        self.max_frame_size = max_frame_size
        self.max_latency = max_latency
        self.deadline = deadline
        self.talker = talker
        self.listeners = listeners
        self.path = path
        self.priority = priority
        self.earliest_offset = earliest_offset
        self.latest_offset = latest_offset

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'stream_type': self.stream_type,
            'period': self.period,
            'frames_per_period': self.frames_per_period,
            'max_frame_size': self.max_frame_size,
            'max_latency': self.max_latency,
            'deadline': self.deadline,
            'talker': self.talker,
            'listeners': self.listeners,
            'path': self.path,
            'priority': self.priority,
            'earliestOffset': self.earliest_offset,
            'latestOffset': self.latest_offset
        }
