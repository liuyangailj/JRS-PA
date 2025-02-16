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
        earliest_offset: float, 
        latest_offset: float
    ):
        self.name = name
        self.stream_type = stream_type
        self.period = period
        self.framesPerPeriod = frames_per_period
        self.maxFrameSize = max_frame_size
        self.maxLatency = max_latency
        self.deadline = deadline
        self.talker = talker
        self.listeners = listeners
        self.path = path
        self.priority = priority
        self.earliestOffset = earliest_offset
        self.latestOffset = latest_offset

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'stream_type': self.stream_type,
            'period': self.period,
            'framesPerPeriod': self.framesPerPeriod,
            'maxFrameSize': self.maxFrameSize,
            'maxLatency': self.maxLatency,
            'deadline': self.deadline,
            'talker': self.talker,
            'listeners': self.listeners,
            'path': self.path,
            'priority': self.priority,
            'earliestOffset': self.earliestOffset,
            'latestOffset': self.latestOffset
        }
