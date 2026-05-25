from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    VOICE_CHECK_IN = "voice_check_in"
    NOTIFY_CAREGIVER = "notify_caregiver"
    OPEN_VIDEO_FEED = "open_video_feed"
    REROUTE_HOSPITAL = "reroute_hospital"
    PULL_OVER = "pull_over"
    UNLOCK_DOORS = "unlock_doors"
    DISPATCH_911 = "dispatch_911"
