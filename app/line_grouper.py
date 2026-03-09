from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from config import OCRConfig
from utils import merge_rects, rect_center, rect_size, sort_boxes_top_left


@dataclass
class LineGroup:
    rects: List[Tuple[int, int, int, int]] = field(default_factory=list)
    members: List[dict] = field(default_factory=list)

    def add(self, item: dict) -> None:
        self.rects.append(item["rect"])
        self.members.append(item)

    @property
    def rect(self) -> Tuple[int, int, int, int]:
        return merge_rects(self.rects)


def _same_line(candidate_rect, line_rect, cfg: OCRConfig) -> bool:
    _, cy = rect_center(candidate_rect)
    _, ly = rect_center(line_rect)

    cw, ch = rect_size(candidate_rect)
    lw, lh = rect_size(line_rect)

    if ch <= 0 or lh <= 0:
        return False

    y_close = abs(cy - ly) <= cfg.line_merge_y_threshold
    height_compatible = min(ch, lh) / max(ch, lh) >= cfg.line_merge_height_ratio

    c_x1, _, c_x2, _ = candidate_rect
    l_x1, _, l_x2, _ = line_rect

    horizontal_gap = min(abs(c_x1 - l_x2), abs(l_x1 - c_x2))
    overlap = not (c_x2 < l_x1 or c_x1 > l_x2)
    x_close = overlap or horizontal_gap <= cfg.line_merge_x_gap

    return y_close and height_compatible and x_close


def group_boxes_into_lines(boxes: List[dict], cfg: OCRConfig) -> List[dict]:
    lines: List[LineGroup] = []

    for box in sort_boxes_top_left(boxes):
        matched = False
        for line in lines:
            if _same_line(box["rect"], line.rect, cfg):
                line.add(box)
                matched = True
                break
        if not matched:
            lg = LineGroup()
            lg.add(box)
            lines.append(lg)

    grouped = []
    for idx, line in enumerate(lines):
        grouped.append(
            {
                "line_id": idx,
                "rect": line.rect,
                "members": line.members,
            }
        )

    return grouped