from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import whisperx

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "My Own Heart (7).mp3"
VIDEO = ROOT / "My Own Heart.mp4"
LYRICS = ROOT / "MyOwnHeart_Lyrics.txt"
OUTPUT = ROOT / "My_Own_Heart_Synchronized.mp4"
ASS_FILE = ROOT / "my_own_heart.ass"
TIMINGS_FILE = ROOT / "my_own_heart_timings.json"
VOCAL_ANCHOR = 11.0
SECOND_LINE_DELAY = 10.0
DEVICE = "cpu"


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def media_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    data = json.loads(subprocess.check_output(cmd, text=True))
    return float(data["format"]["duration"])


def lyric_lines() -> list[str]:
    return [line.strip() for line in LYRICS.read_text(encoding="utf-8").splitlines() if line.strip()]


def forced_align_words(lines: list[str]) -> list[dict]:
    duration = media_duration(AUDIO)
    transcript = " ".join(lines)
    audio = whisperx.load_audio(str(AUDIO))
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    segments = [{"start": VOCAL_ANCHOR - 0.25, "end": duration - 0.05, "text": transcript}]
    result = whisperx.align(segments, model_a, metadata, audio, DEVICE, return_char_alignments=False)

    words: list[dict] = []
    for item in result.get("word_segments", []):
        token = norm(str(item.get("word", "")))
        start = item.get("start")
        end = item.get("end")
        if token and start is not None and end is not None:
            words.append({"word": token, "start": float(start), "end": float(end)})
    if len(words) < 10:
        raise RuntimeError(f"Forced alignment returned too few words: {len(words)}")
    return words


def build_line_timings(lines: list[str], aligned_words: list[dict]) -> list[dict]:
    counts = [len([w for w in re.findall(r"[A-Za-z0-9’']+", line) if norm(w)]) for line in lines]
    total_expected = sum(counts)
    total_aligned = len(aligned_words)
    duration = media_duration(AUDIO)
    if total_expected < 2 or total_aligned < 2:
        raise RuntimeError("Not enough words for alignment.")

    def aligned_index(expected_index: int) -> int:
        ratio = expected_index / (total_expected - 1)
        return max(0, min(total_aligned - 1, round(ratio * (total_aligned - 1))))

    timings: list[dict] = []
    cursor = 0
    for line_no, (line, count) in enumerate(zip(lines, counts), start=1):
        first = aligned_index(cursor)
        last = aligned_index(cursor + count - 1)
        start = aligned_words[first]["start"] - 0.08
        end = aligned_words[last]["end"] + 0.22

        if line_no == 1:
            start = VOCAL_ANCHOR
        else:
            start += SECOND_LINE_DELAY
            end += SECOND_LINE_DELAY

        if timings:
            start = max(start, timings[-1]["end"] + 0.03)
        end = max(end, start + 1.0)
        end = min(end, duration - 0.10)
        timings.append({"text": line, "start": start, "end": end})
        cursor += count

    TIMINGS_FILE.write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")
    return timings


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(timings: list[dict]) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyrics,DejaVu Sans,54,&H00FFFFFF,&H000000FF,&H00141414,&H78000000,-1,0,0,0,100,100,0,0,1,3,1,2,120,120,95,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for item in timings:
        safe = item["text"].replace("{", "(").replace("}", ")")
        events.append(f"Dialogue: 0,{ass_time(item['start'])},{ass_time(item['end'])},Lyrics,,0,0,0,,{{\\fad(150,150)}}{safe}")
    ASS_FILE.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def render() -> None:
    subprocess.run([
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(VIDEO),
        "-i", str(AUDIO),
        "-vf", f"ass={ASS_FILE.name}",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "320k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-shortest", str(OUTPUT),
    ], check=True)


def main() -> None:
    for path in (AUDIO, VIDEO, LYRICS):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.name}")
    lines = lyric_lines()
    timings = build_line_timings(lines, forced_align_words(lines))
    write_ass(timings)
    render()
    print(f"Created: {OUTPUT.name}")


if __name__ == "__main__":
    main()
