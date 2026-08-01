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
OUTPUT = ROOT / "My_Own_Heart_FIXED_AT_33_SECONDS.mp4"
ASS_FILE = ROOT / "my_own_heart_fixed_33.ass"
TIMINGS_FILE = ROOT / "my_own_heart_fixed_33_timings.json"
DEVICE = "cpu"
FIRST_VOCAL = 11.0
VERSE_ANCHOR = 33.0
ANCHOR_TEXT = "you said you'd stay but i stood alone"


def norm(text: str) -> str:
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9']+", " ", text).strip()


def media_duration(path: Path) -> float:
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ], text=True))
    return float(data["format"]["duration"])


def load_lines() -> list[str]:
    raw = [line.strip() for line in LYRICS.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Remove accidental consecutive duplicate lines from the previous test.
    lines: list[str] = []
    for line in raw:
        if lines and norm(lines[-1]) == norm(line):
            continue
        lines.append(line)
    return lines


def forced_align(lines: list[str]) -> list[dict]:
    duration = media_duration(AUDIO)
    audio = whisperx.load_audio(str(AUDIO))
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
    result = whisperx.align(
        [{"start": FIRST_VOCAL - 0.25, "end": duration - 0.05, "text": " ".join(lines)}],
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )
    words = []
    for item in result.get("word_segments", []):
        word = norm(str(item.get("word", "")))
        if word and item.get("start") is not None and item.get("end") is not None:
            words.append({"word": word, "start": float(item["start"]), "end": float(item["end"])})
    if len(words) < 10:
        raise RuntimeError(f"Too few aligned words: {len(words)}")
    return words


def build_timings(lines: list[str], words: list[dict]) -> list[dict]:
    counts = [len(re.findall(r"[A-Za-z0-9’']+", line)) for line in lines]
    total_expected = sum(counts)
    total_heard = len(words)

    def heard_index(expected_index: int) -> int:
        if total_expected <= 1:
            return 0
        ratio = expected_index / (total_expected - 1)
        return max(0, min(total_heard - 1, round(ratio * (total_heard - 1))))

    timings: list[dict] = []
    cursor = 0
    for line, count in zip(lines, counts):
        first = heard_index(cursor)
        last = heard_index(cursor + count - 1)
        start = words[first]["start"] - 0.08
        end = words[last]["end"] + 0.20
        timings.append({"text": line, "start": max(0.0, start), "end": max(start + 1.0, end)})
        cursor += count

    # First lyric starts exactly at the verified vocal entrance.
    first_delta = FIRST_VOCAL - timings[0]["start"]
    for item in timings:
        item["start"] += first_delta
        item["end"] += first_delta

    # Hard anchor: this exact verse line must start at 33.00 seconds.
    anchor_index = next(
        (i for i, item in enumerate(timings) if norm(item["text"]).rstrip(",") == ANCHOR_TEXT),
        None,
    )
    if anchor_index is None:
        raise RuntimeError("Could not find the verse anchor line in the lyrics.")

    delta = VERSE_ANCHOR - timings[anchor_index]["start"]
    for i in range(anchor_index, len(timings)):
        timings[i]["start"] += delta
        timings[i]["end"] += delta

    # Enforce strict order after the anchor without moving it away from 33 seconds.
    timings[anchor_index]["start"] = VERSE_ANCHOR
    timings[anchor_index]["end"] = max(timings[anchor_index]["end"], VERSE_ANCHOR + 3.2)
    for i in range(anchor_index + 1, len(timings)):
        if timings[i]["start"] < timings[i - 1]["end"] + 0.04:
            shift = timings[i - 1]["end"] + 0.04 - timings[i]["start"]
            timings[i]["start"] += shift
            timings[i]["end"] += shift

    duration = media_duration(AUDIO)
    for item in timings:
        item["start"] = round(max(0.0, min(item["start"], duration - 0.2)), 2)
        item["end"] = round(max(item["start"] + 0.8, min(item["end"], duration - 0.05)), 2)

    print(f"ANCHOR CONFIRMED: {timings[anchor_index]['text']} -> {timings[anchor_index]['start']:.2f}s")
    return timings


def ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(timings: list[dict]) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Lyrics,DejaVu Sans,54,&H00FFFFFF,&H000000FF,&H00141414,&H78000000,-1,0,0,0,100,100,0,0,1,3,1,2,120,120,95,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events = []
    for item in timings:
        safe = item["text"].replace("{", "(").replace("}", ")")
        events.append(
            f"Dialogue: 0,{ass_time(item['start'])},{ass_time(item['end'])},Lyrics,,0,0,0,,{{\\fad(120,120)}}{safe}"
        )
    ASS_FILE.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def render() -> None:
    subprocess.run([
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(VIDEO), "-i", str(AUDIO),
        "-vf", f"ass={ASS_FILE.name}", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "320k", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-shortest", str(OUTPUT),
    ], check=True)


def main() -> None:
    for required in (AUDIO, VIDEO, LYRICS):
        if not required.exists():
            raise FileNotFoundError(required)
    lines = load_lines()
    timings = build_timings(lines, forced_align(lines))
    TIMINGS_FILE.write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")
    write_ass(timings)
    render()
    print(f"Created {OUTPUT.name}")


if __name__ == "__main__":
    main()
