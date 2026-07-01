# Copyright (C) 2025 Xiaomi Corporation
# This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.
"""RTSP camera configuration schema."""

from miot.rtsp_camera import RtspCameraInfo
from miot.types import MIoTCameraCodec
from pydantic import BaseModel, Field


class RtspCameraConfig(BaseModel):
    """RTSP camera configuration loaded from YAML."""

    did: str = Field(..., description="Camera unique id")
    name: str = Field(..., description="Camera display name")
    rtsp_url: str = Field(..., description="RTSP url with optional credential")
    codec: str | None = Field(
        default=None, description="Codec hint: h264 | h265 | auto"
    )
    enable_audio: bool = Field(default=False, description="Enable audio decoding")
    transport: str | None = Field(
        default=None, description="Transport protocol: tcp or udp"
    )
    home_id: str | None = Field(default=None, description="Bound home id")
    home_name: str | None = Field(default=None, description="Home/area name")
    room_name: str | None = Field(default=None, description="Room name")
    vendor: str | None = Field(default=None, description="Vendor or brand name")
    model: str | None = Field(
        default="rtsp_camera", description="Model name for display"
    )
    icon: str | None = Field(default=None, description="Icon url or path")

    def _parse_codec(self) -> MIoTCameraCodec | None:
        """Parse codec hint; return None to allow runtime autodetect."""
        if self.codec is None:
            return None
        codec_lower = str(self.codec).strip().lower()
        if codec_lower in ("", "auto", "detect", "auto-detect", "autodetect"):
            return None
        if "265" in codec_lower or "hevc" in codec_lower:
            return MIoTCameraCodec.VIDEO_H265
        if "264" in codec_lower or "avc" in codec_lower:
            return MIoTCameraCodec.VIDEO_H264
        return None

    def to_rtsp_camera_info(self) -> RtspCameraInfo:
        """Convert to runtime RtspCameraInfo."""
        transport_lower = (self.transport or "").strip().lower()
        return RtspCameraInfo(
            did=self.did,
            name=self.name,
            rtsp_url=self.rtsp_url,
            codec=self._parse_codec(),
            channel_count=1,
            enable_audio=self.enable_audio,
            use_tcp=transport_lower == "tcp",
            home_id=self.home_id,
            home_name=self.home_name,
            room_name=self.room_name,
            vendor=self.vendor,
            model=self.model or "rtsp_camera",
            icon=self.icon,
        )
