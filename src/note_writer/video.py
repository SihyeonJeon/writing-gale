"""
Look inside a post's video instead of judging it from one thumbnail.

For the first video on a post: download the smallest MP4 the X API lists for it, pull a few evenly spaced
frames (sent to the model as images) and transcribe the first minutes of audio locally with Whisper.
Everything runs on the machine running the writer; nothing is stored after the run.

Every failure returns a reason instead of a guess ("no mp4 variant", "download failed", ...), so the
prompt can say what was and wasn't seen.
"""

import base64
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests

from data_models import Media, PostWithContext

MAX_VIDEO_BYTES = 60_000_000
MAX_SECONDS = 180  # only the first 3 minutes are looked at
N_FRAMES = 4
FRAME_WIDTH = 768
MAX_TRANSCRIPT_CHARS = 1500
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
MIN_LANGUAGE_PROB = 0.6
MAX_NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0

_whisper = None


@dataclass
class VideoInfo:
    frames: List[str] = field(default_factory=list)  # base64 data URLs
    transcript: str = ""
    language: str = ""
    seconds_covered: float = 0.0
    status: str = "ok"  # "ok" or the reason nothing (or only part) was extracted


def pick_variant(media: Media) -> Optional[str]:
    """Smallest MP4: enough to read on-screen text and hear speech, cheapest to download."""
    mp4s = [v for v in media.variants if v.get("content_type") == "video/mp4" and v.get("url")]
    if not mp4s:
        return None
    return min(mp4s, key=lambda v: v.get("bit_rate") or 0)["url"]


def first_video(pwc: PostWithContext) -> Optional[Media]:
    for post in [pwc.post, pwc.quoted_post, pwc.in_reply_to_post]:
        for m in post.media if post else []:
            if m.media_type in ("video", "animated_gif"):
                return m
    return None


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: List[str], timeout: int = 120) -> bool:
    try:
        return subprocess.run(args, capture_output=True, timeout=timeout).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _download(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            if r.status_code != 200:
                return False
            size = 0
            with dest.open("wb") as f:
                for chunk in r.iter_content(1 << 16):
                    size += len(chunk)
                    if size > MAX_VIDEO_BYTES:
                        break  # a truncated MP4 still decodes from the start
                    f.write(chunk)
        return size > 0
    except requests.RequestException:
        return False


def _frames(ffmpeg: str, video: Path, seconds: float, out_dir: Path) -> List[str]:
    urls = []
    for i in range(N_FRAMES):
        t = seconds * (i + 0.5) / N_FRAMES
        out = out_dir / f"f{i}.jpg"
        ok = _run([ffmpeg, "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "4", str(out)])
        if ok and out.exists() and out.stat().st_size > 0:
            urls.append("data:image/jpeg;base64," + base64.b64encode(out.read_bytes()).decode())
    return urls


def _transcribe(ffmpeg: str, video: Path, out_dir: Path) -> tuple[str, str]:
    global _whisper
    wav = out_dir / "a.wav"
    if not _run([ffmpeg, "-y", "-i", str(video), "-t", str(MAX_SECONDS), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]) or not wav.exists():
        return "", ""  # no audio track
    from faster_whisper import WhisperModel

    if _whisper is None:
        _whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    # Read the 16 kHz mono PCM ourselves; the library's own file decoding depends on a PyAV version.
    import wave

    import numpy as np

    with wave.open(str(wav), "rb") as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if audio.size == 0:
        return "", ""
    segments, info = _whisper.transcribe(audio, vad_filter=True, beam_size=1)
    # Whisper invents fluent text for music and noise. Keep only segments it is confident are speech,
    # and nothing at all when it can't tell the language.
    if (info.language_probability or 0) < MIN_LANGUAGE_PROB:
        return "", ""
    text = " ".join(
        s.text.strip() for s in segments if s.no_speech_prob < MAX_NO_SPEECH_PROB and s.avg_logprob > MIN_AVG_LOGPROB
    ).strip()
    return text[:MAX_TRANSCRIPT_CHARS], (info.language or "") if text else ""


def inspect_file(video: Path, duration_s: float) -> VideoInfo:
    info = VideoInfo(seconds_covered=min(duration_s or MAX_SECONDS, MAX_SECONDS))
    try:
        ffmpeg = _ffmpeg()
    except Exception as e:
        return VideoInfo(status=f"ffmpeg unavailable: {e}")
    with tempfile.TemporaryDirectory() as d:
        info.frames = _frames(ffmpeg, video, info.seconds_covered, Path(d))
        try:
            info.transcript, info.language = _transcribe(ffmpeg, video, Path(d))
        except Exception as e:
            info.status = f"frames only, transcription failed: {type(e).__name__}"
    if not info.frames and not info.transcript:
        info.status = "could not decode the video"
    return info


def inspect_video(pwc: PostWithContext) -> Optional[VideoInfo]:
    """None when the post has no video; otherwise what could be extracted, with a status."""
    media = first_video(pwc)
    if media is None:
        return None
    url = pick_variant(media)
    if not url:
        return VideoInfo(status="no mp4 variant from the API")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "v.mp4"
        if not _download(url, path):
            return VideoInfo(status="video download failed")
        return inspect_file(path, (media.duration_ms or 0) / 1000)


def describe_for_prompt(info: Optional[VideoInfo]) -> str:
    if info is None:
        return ""
    if not info.frames and not info.transcript:
        return f"\nThe post has a video that could not be inspected ({info.status}). Do not describe what it shows unless your search verifies it.\n"
    out = f"\nThe post has a video. Attached after the post's own images: {len(info.frames)} frames taken evenly across its first {info.seconds_covered:.0f} seconds.\n"
    if info.transcript:
        out += (
            "Machine transcript of the audio (may contain errors and may miss parts; quote it only if a source "
            f"confirms the wording):\n```\n{info.transcript}\n```\n"
        )
    else:
        out += "No speech was transcribed from it.\n"
    return out
