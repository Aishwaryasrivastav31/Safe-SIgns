"""The 43 GTSRB classes, what each one asks the driver to do, and how fast that allows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sign:
    cid: int
    name: str
    kind: str
    speed: int | None = None


SIGNS = [
    Sign(0, "speed limit 20", "speed", 20), Sign(1, "speed limit 30", "speed", 30),
    Sign(2, "speed limit 50", "speed", 50), Sign(3, "speed limit 60", "speed", 60),
    Sign(4, "speed limit 70", "speed", 70), Sign(5, "speed limit 80", "speed", 80),
    Sign(6, "end of speed limit 80", "end80"), Sign(7, "speed limit 100", "speed", 100),
    Sign(8, "speed limit 120", "speed", 120), Sign(9, "no passing", "nopass"),
    Sign(10, "no passing for trucks", "nopass"), Sign(11, "right of way at next junction", "junction"),
    Sign(12, "priority road", "neutral"), Sign(13, "yield", "yield"), Sign(14, "stop", "stop"),
    Sign(15, "no vehicles", "no_entry"), Sign(16, "no trucks", "neutral"), Sign(17, "no entry", "no_entry"),
    Sign(18, "general caution", "warning"), Sign(19, "dangerous curve left", "warning"),
    Sign(20, "dangerous curve right", "warning"), Sign(21, "double curve", "warning"),
    Sign(22, "bumpy road", "warning"), Sign(23, "slippery road", "warning"),
    Sign(24, "road narrows on the right", "warning"), Sign(25, "road work", "warning"),
    Sign(26, "traffic signals", "junction"), Sign(27, "pedestrians", "people"),
    Sign(28, "children crossing", "people"), Sign(29, "bicycles crossing", "people"),
    Sign(30, "beware of ice or snow", "warning"), Sign(31, "wild animals crossing", "warning"),
    Sign(32, "end of all limits", "end_all"), Sign(33, "turn right ahead", "neutral"),
    Sign(34, "turn left ahead", "neutral"), Sign(35, "ahead only", "neutral"),
    Sign(36, "go straight or right", "neutral"), Sign(37, "go straight or left", "neutral"),
    Sign(38, "keep right", "neutral"), Sign(39, "keep left", "neutral"),
    Sign(40, "roundabout mandatory", "junction"), Sign(41, "end of no passing", "end_nopass"),
    Sign(42, "end of no passing for trucks", "end_nopass"),
]
N_CLASSES = len(SIGNS)
NAMES = [s.name for s in SIGNS]
DEFAULT_SPEED = {"urban": 50, "rural": 100, "motorway": 130}
MAX_LEGAL = {"urban": 60, "rural": 100, "motorway": 130}
