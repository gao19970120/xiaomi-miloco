# Copyright (C) 2025 Xiaomi Corporation
# This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.

"""
Camera vision handler utility for managing camera image streams.
Provides functionality to handle camera image queues and vision processing.
"""

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Coroutine
from typing import Any

from av.audio.frame import AudioFrame
from av.video.frame import VideoFrame
from miot.camera import MIoTCamera, MIoTCameraInstance
from miot.rtsp_camera import RTSPCamera, RtspCameraInfo, RTSPCameraInstance
from miot.types import MIoTCameraCodec, MIoTCameraInfo

from miloco.miot.schema import CameraImgInfo, CameraImgSeq, CameraInfo

logger = logging.getLogger(__name__)


def _decode_g711(data: bytes, codec: MIoTCameraCodec):
    """Decode G.711 A-law/μ-law and upsample 8 kHz PCM to 16 kHz."""
    import numpy as np

    encoded = np.frombuffer(data, dtype=np.uint8)
    if codec == MIoTCameraCodec.AUDIO_G711A:
        values = np.bitwise_xor(encoded, 0x55).astype(np.int32)
        mantissa = values & 0x0F
        exponent = (values >> 4) & 0x07
        linear = (mantissa << 4) + 8
        linear = np.where(exponent == 1, linear + 0x100, linear)
        linear = np.where(
            exponent > 1,
            (linear + 0x100) << (exponent - 1),
            linear,
        )
        pcm8 = np.where(values & 0x80, linear, -linear)
    elif codec == MIoTCameraCodec.AUDIO_G711U:
        values = np.bitwise_xor(encoded, 0xFF).astype(np.int32)
        linear = ((values & 0x0F) << 3) + 0x84
        linear <<= (values >> 4) & 0x07
        pcm8 = np.where(values & 0x80, 0x84 - linear, linear - 0x84)
    else:
        raise ValueError(f"Unsupported G.711 codec: {codec}")

    pcm8 = np.clip(pcm8, -32768, 32767).astype(np.int16)
    if pcm8.size == 0:
        return pcm8

    # Linear 2× interpolation is sufficient for the 8 kHz speech-band source
    # and avoids the deprecated stdlib audioop module (removed in Python 3.13).
    pcm16 = np.empty(pcm8.size * 2, dtype=np.int16)
    pcm16[0::2] = pcm8
    if pcm8.size > 1:
        mixed = (
            pcm8[:-1].astype(np.int32) + pcm8[1:].astype(np.int32)
        ) // 2
        pcm16[1:-1:2] = mixed.astype(np.int16)
    pcm16[-1] = pcm8[-1]
    return pcm16


class SizeLimitedQueue:
    """Size-limited queue that automatically removes oldest elements"""

    def __init__(self, max_size: int, ttl: int):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        self.max_size = max_size
        self.ttl = ttl
        self.queue = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def _filter_old_items(self) -> None:
        """Filter old items"""
        current_time = time.time()
        while self.queue and current_time - self.queue[0][1] > self.ttl:
            self.queue.popleft()

    def clear(self) -> None:
        """Clear queue"""
        with self._lock:
            self.queue.clear()

    def put(self, item: Any) -> None:
        """Add element, automatically removes oldest element if queue is full"""
        with self._lock:
            self._filter_old_items()
            self.queue.append((item, time.time()))

    def get(self) -> Any:
        """Get and remove the oldest element"""
        with self._lock:
            if not self.queue:
                raise IndexError("Queue is empty")
            self._filter_old_items()
            if not self.queue:
                raise IndexError("Queue is empty after filtering")
            return self.queue.popleft()[0]

    def peek(self) -> Any:
        """View the oldest element without removing it"""
        with self._lock:
            if not self.queue:
                raise IndexError("Queue is empty")
            self._filter_old_items()
            if not self.queue:
                raise IndexError("Queue is empty after filtering")
            return self.queue[0][0]

    def size(self) -> int:
        """Return current queue size"""
        with self._lock:
            self._filter_old_items()
            return len(self.queue)

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        with self._lock:
            self._filter_old_items()
            return len(self.queue) == 0

    def is_full(self) -> bool:
        """Check if queue is full"""
        with self._lock:
            self._filter_old_items()
            return len(self.queue) == self.max_size

    def to_list(self) -> list[Any]:
        """Convert to list, from oldest to newest"""
        with self._lock:
            self._filter_old_items()
            return [item[0] for item in self.queue]

    def get_recent(self, n: int) -> list[Any]:
        """Get the most recent n elements, sorted by time from old to new

        Args:
            n: Number of elements to get

        Returns:
            List of the most recent n elements, returns all elements if queue has fewer than n elements
        """
        if n <= 0:
            return []

        with self._lock:
            self._filter_old_items()
            actual_n = min(n, len(self.queue))
            recent_items = [entry[0] for entry in self.queue][-actual_n:]
            return recent_items


# ============ RTSP SUPPORT: Base Strategy ============

class BaseCameraVisionHandler:
    """Base camera vision handler strategy - common interface for MIoT + RTSP cameras."""

    async def register_raw_stream(
        self,
        callback: Callable[[str, bytes, int, int, int], Coroutine],
        channel: int,
    ):
        raise NotImplementedError

    async def unregister_raw_stream(self, channel: int):
        raise NotImplementedError

    async def update_camera_info(self, camera_info: Any) -> None:
        raise NotImplementedError

    def get_recents_camera_img(self, channel: int, n: int) -> CameraImgSeq:
        raise NotImplementedError

    async def destroy(self) -> None:
        raise NotImplementedError


# ============ Original MIoT Camera Handler ============

class CameraVisionHandler(BaseCameraVisionHandler):
    """Camera vision handler for managing camera image streams"""

    _CODEC_ID_MAP: dict = {
        MIoTCameraCodec.AUDIO_OPUS: "opus",
        MIoTCameraCodec.AUDIO_G711A: "g711a",
        MIoTCameraCodec.AUDIO_G711U: "g711u",
    }

    def __init__(
        self,
        camera_info: MIoTCameraInfo,
        miot_camera_instance: MIoTCameraInstance,
        miot_camera_manager: MIoTCamera,
        max_size: int,
        ttl: int,
    ):
        # ttl seconds
        self.camera_info = camera_info
        self.miot_camera_instance = miot_camera_instance
        # 需要 manager 引用走 destroy_camera_async 清 _camera_map cache;
        # 直调 instance.destroy_async() 不 evict cache,下次 create_camera_async
        # 会 "camera already exists" 短路返回已 free 的 instance,register/start 全部
        # 返回 -1。
        self._miot_camera_manager = miot_camera_manager
        self.camera_img_queues: dict[int, SizeLimitedQueue] = {}
        self._audio_codec: dict[int, str | None] = {}

        for channel in range(self.camera_info.channel_count or 1):
            self.camera_img_queues[channel] = SizeLimitedQueue(
                max_size=max_size, ttl=ttl
            )
            self._audio_codec[channel] = None
            asyncio.create_task(
                self.miot_camera_instance.register_decode_jpg_async(
                    self.add_camera_img, channel
                )
            )

        logger.info(
            "CameraImgManager init success, camera did: %s", self.camera_info.did
        )

    async def register_raw_stream(
        self, callback: Callable[[str, bytes, int, int, int], Coroutine], channel: int
    ):
        await self.miot_camera_instance.register_raw_video_async(callback, channel)

    async def unregister_raw_stream(self, channel: int):
        await self.miot_camera_instance.unregister_raw_video_async(channel)

    async def add_camera_img(self, did: str, data: bytes, ts: int, channel: int):
        logger.debug(
            "add_camera_img camera_id: %s, camera timestamp: %d, image_size: %d",
            did,
            ts,
            len(data),
        )
        self.camera_img_queues[channel].put(
            CameraImgInfo(data=data, timestamp=int(time.time()))
        )

    async def update_camera_info(self, camera_info: MIoTCameraInfo) -> None:
        self.camera_info = camera_info
        if self.camera_info.connected:
            for channel in range(self.camera_info.channel_count or 1):
                await self.miot_camera_instance.register_decode_jpg_async(
                    self.add_camera_img, channel
                )
        else:
            for channel in range(self.camera_info.channel_count or 1):
                await self.miot_camera_instance.unregister_decode_jpg_async(channel)
                self.camera_img_queues[channel].clear()

    def get_recent_camera_img(self, channel: int, n: int) -> CameraImgSeq:
        if self.camera_info.connected:
            return CameraImgSeq(
                camera_info=CameraInfo.model_validate(self.camera_info.model_dump()),
                channel=channel,
                img_list=self.camera_img_queues[channel].get_recent(n),
            )
        else:
            return CameraImgSeq(
                camera_info=CameraInfo.model_validate(self.camera_info.model_dump()),
                channel=channel,
                img_list=[],
            )

    def get_recents_camera_img(self, channel: int, n: int) -> CameraImgSeq:
        """Alias for BaseCameraVisionHandler interface compatibility."""
        return self.get_recent_camera_img(channel, n)

    def get_audio_codec(self, channel: int) -> str | None:
        """Get detected audio codec for a channel."""
        return self._audio_codec.get(channel)

    async def register_raw_audio_stream(
        self, callback: Callable[[str, bytes, int, int, int], Coroutine], channel: int
    ):
        async def _detecting_wrapper(
            did: str, data: bytes, ts: int, seq: int, ch: int, codec_id: MIoTCameraCodec
        ):
            if self._audio_codec.get(ch) is None:
                self._audio_codec[ch] = self._CODEC_ID_MAP.get(codec_id, "opus")
                logger.info(
                    "Detected audio codec for camera %s channel %d: %s",
                    did,
                    ch,
                    self._audio_codec[ch],
                )
            await callback(did, data, ts, seq, ch)

        await self.miot_camera_instance.register_raw_audio_async(
            _detecting_wrapper, channel
        )

    async def unregister_raw_audio_stream(self, channel: int):
        await self.miot_camera_instance.unregister_raw_audio_async(channel)
        self._audio_codec[channel] = None

    async def register_decode_video_frame_stream(
        self, callback: Callable[[str, VideoFrame, int, int, int, int], Coroutine], channel: int
    ) -> int:
        """Register decoded VideoFrame callback (multi_reg, coexists with internal decode_jpg)."""
        return await self.miot_camera_instance.register_decode_video_frame_async(
            callback, channel, multi_reg=True
        )

    async def unregister_decode_video_frame_stream(self, channel: int, reg_id: int):
        await self.miot_camera_instance.unregister_decode_video_frame_async(
            channel, reg_id
        )

    async def register_decode_audio_frame_stream(
        self, callback: Callable[[str, AudioFrame, int, int, int, int], Coroutine], channel: int
    ) -> int:
        """Register decoded AudioFrame callback (multi_reg)."""
        return await self.miot_camera_instance.register_decode_audio_frame_async(
            callback, channel, multi_reg=True
        )

    async def unregister_decode_audio_frame_stream(self, channel: int, reg_id: int):
        await self.miot_camera_instance.unregister_decode_audio_frame_async(
            channel, reg_id
        )

    async def destroy(self) -> None:
        for channel in range(self.camera_info.channel_count or 1):
            await self.miot_camera_instance.unregister_decode_jpg_async(channel=channel)
            await self.miot_camera_instance.unregister_raw_video_async(channel=channel)
            await self.miot_camera_instance.unregister_raw_audio_async(channel=channel)
            self.camera_img_queues[channel].clear()

        # 走 manager 入口让 SDK 从 _camera_map cache 里 evict,否则下次
        # create_camera_async 会短路返回这个已 free 的 instance("camera already
        # exists"),register/start 全部 -1,无法重新拉流。
        await self._miot_camera_manager.destroy_camera_async(
            did=self.camera_info.did
        )


# ============ RTSP SUPPORT: RTSP Camera Vision Handler ============

class RtspCameraVisionHandler(BaseCameraVisionHandler):
    """RTSP camera vision handler using libcamera_rtsp native library.

    Mirrors CameraVisionHandler interface so the perception engine can consume
    RTSP cameras without knowing the source is not MIoT.
    """

    def __init__(
        self,
        camera_info: RtspCameraInfo,
        rtsp_camera_instance: RTSPCameraInstance,
        rtsp_camera_manager: RTSPCamera,
        max_size: int,
        ttl: int,
    ):
        self.camera_info = camera_info
        self.rtsp_camera_instance = rtsp_camera_instance
        self._rtsp_camera_manager = rtsp_camera_manager
        self.camera_img_queues: dict[int, SizeLimitedQueue] = {}
        self._max_size = max_size
        self._ttl = ttl

        # Start the RTSP stream first, then register decode callbacks
        asyncio.create_task(self._start_and_register())

        logger.info(
            "RtspCameraVisionHandler init success, camera did: %s", self.camera_info.did
        )

    async def _start_and_register(self):
        """Start RTSP stream then register decode callbacks."""
        try:
            await self.rtsp_camera_instance.start_async(
                enable_audio=self.camera_info.enable_audio,
                enable_reconnect=True,
            )
            logger.info("RTSP camera started: %s", self.camera_info.did)
        except Exception as e:
            logger.error("Failed to start RTSP camera %s: %s", self.camera_info.did, e)
            self.camera_info.online = False
            return

        max_sz = self._max_size
        for channel in range(self.camera_info.channel_count or 1):
            self.camera_img_queues[channel] = SizeLimitedQueue(max_size=max_sz, ttl=self._ttl)
            await self.rtsp_camera_instance.register_decode_jpg_async(
                self.add_camera_img, channel
            )
        logger.info("RTSP camera decode registered: %s", self.camera_info.did)

    async def register_raw_stream(
        self, callback: Callable[[str, bytes, int, int, int], Coroutine], channel: int
    ):
        await self.rtsp_camera_instance.register_raw_video_async(callback, channel)

    async def unregister_raw_stream(self, channel: int):
        await self.rtsp_camera_instance.unregister_raw_video_async(channel)

    async def add_camera_img(self, did: str, data: bytes, ts: int, channel: int):
        logger.debug(
            "rtsp add_camera_img camera_id: %s, camera timestamp: %d, image_size: %d",
            did,
            ts,
            len(data),
        )
        self.camera_img_queues[channel].put(
            CameraImgInfo(data=data, timestamp=int(time.time()))
        )

    async def update_camera_info(self, camera_info: RtspCameraInfo) -> None:
        self.camera_info = camera_info
        if self.camera_info.online:
            for channel in range(self.camera_info.channel_count or 1):
                await self.rtsp_camera_instance.register_decode_jpg_async(
                    self.add_camera_img, channel
                )
        else:
            for channel in range(self.camera_info.channel_count or 1):
                await self.rtsp_camera_instance.unregister_decode_jpg_async(channel)
                self.camera_img_queues[channel].clear()

    def get_recents_camera_img(self, channel: int, n: int) -> CameraImgSeq:
        if self.camera_info.online:
            return CameraImgSeq(
                camera_info=CameraInfo.model_validate(self.camera_info.model_dump()),
                channel=channel,
                img_list=self.camera_img_queues[channel].get_recent(n),
            )
        else:
            return CameraImgSeq(
                camera_info=CameraInfo.model_validate(self.camera_info.model_dump()),
                channel=channel,
                img_list=[],
            )

    def get_recent_camera_img(self, channel: int, n: int) -> CameraImgSeq:
        """Alias for interface compatibility (same as get_recents_camera_img)."""
        return self.get_recents_camera_img(channel, n)

    async def destroy(self) -> None:
        for channel in range(self.camera_info.channel_count or 1):
            await self.rtsp_camera_instance.unregister_decode_jpg_async(channel=channel)
            await self.rtsp_camera_instance.unregister_raw_video_async(channel=channel)
            self.camera_img_queues[channel].clear()

        await self._rtsp_camera_manager.destroy_camera_async(self.camera_info.did)

    # ── Video/Audio frame stream for live watch (WebSocket) ──
    # RTSP 摄像头直接复用 decoder 的 BGR 输出,再走统一 H.264 编码路径。

    async def register_decode_video_frame_stream(
        self,
        callback: Callable[[str, VideoFrame, int, int, int, int], Coroutine],
        channel: int,
    ) -> int:
        reg_id = await self.rtsp_camera_instance.register_decode_video_frame_async(
            callback, channel=channel, multi_reg=True
        )
        logger.info(
            "RTSP decode_video_frame_stream registered for %s ch=%d reg=%d",
            self.camera_info.did, channel, reg_id,
        )
        return reg_id

    async def unregister_decode_video_frame_stream(self, channel: int, reg_id: int):
        await self.rtsp_camera_instance.unregister_decode_video_frame_async(
            channel, reg_id
        )
        logger.debug(
            "RTSP decode_video_frame_stream unregistered for %s reg=%d",
            self.camera_info.did, reg_id,
        )

    async def register_decode_audio_frame_stream(
        self,
        callback: Callable[[str, AudioFrame, int, int, int, int], Coroutine],
        channel: int,
    ) -> int:
        """Register a normalized s16/mono/16kHz callback for RTSP audio."""
        import numpy as np

        async def _pcm_bridge(
            did: str, data: bytes, timestamp: int, ch: int
        ) -> None:
            """Decode the source codec and normalize to 16 kHz mono PCM."""
            try:
                codec = self.rtsp_camera_instance.get_audio_codec(ch)
                if codec in (
                    MIoTCameraCodec.AUDIO_G711A,
                    MIoTCameraCodec.AUDIO_G711U,
                ):
                    pcm = _decode_g711(data, codec)
                elif codec == MIoTCameraCodec.AUDIO_OPUS:
                    # MIoTMediaDecoder already emits s16/mono/16k for Opus.
                    pcm = np.frombuffer(data, dtype="<i2").copy()
                else:
                    logger.debug(
                        "RTSP audio codec not known yet for %s channel %d",
                        did, ch,
                    )
                    return
                if pcm.size == 0:
                    return
            except Exception as e:
                logger.error("PCM bridge error for %s: %s", did, e)
                return
            await callback(did, pcm, timestamp, ch, 0, 0)

        return await self.rtsp_camera_instance.register_decode_pcm_async(
            _pcm_bridge, channel=channel, multi_reg=True
        )

    async def register_raw_audio_stream(
        self, callback, channel: int = 0
    ):
        """Expose the encoded RTSP audio track to the live-view WebSocket."""
        return await self.rtsp_camera_instance.register_raw_audio_async(
            callback, channel
        )

    async def unregister_raw_audio_stream(self, channel: int):
        await self.rtsp_camera_instance.unregister_raw_audio_async(channel)

    def get_audio_codec(self, channel: int) -> str | None:
        """Return the live-view codec name expected by the web client."""
        from miot.types import MIoTCameraCodec

        codec = self.rtsp_camera_instance.get_audio_codec(channel)
        return {
            MIoTCameraCodec.AUDIO_OPUS: "opus",
            MIoTCameraCodec.AUDIO_G711A: "g711a",
            MIoTCameraCodec.AUDIO_G711U: "g711u",
        }.get(codec)

    async def unregister_decode_audio_frame_stream(self, channel: int, reg_id: int):
        await self.rtsp_camera_instance.unregister_decode_pcm_async(
            channel, reg_id
        )
