from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "My Own Heart (7).mp3"
VIDEO = ROOT / "My Own Heart.mp4"
LYRICS = ROOT / "MyOwnHeart_Lyrics.txt"
OUTPUT = ROOT / "My_Own_Heart_Synchronized.mp4"
ASS_FILE = ROOT / "my_own_heart.ass"
TIMINGS_FILE = ROOT / "my_own_heart_timings.json"
VOCAL_ANCHOR = 11.0


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def media_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    return float(data["format"]["duration"])


def transcribe() -> list[tuple[str, float, float]]:
    model = WhisperModel("medium.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(AUDIO),
        language="en",
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        best_of=5,
        temperature=0,
        condition_on_previous_text=True,
        initial_prompt="I learned the hard way. Sometimes you save yourself. My Own Heart by Minoo Lakani.",
    )

    words: list[tuple[str, float, float]] = []
    for segment in segments:
        for item in segment.words or []:
            token = norm(item.word)
            if token and item.start is not None and item.end is not None:
                start = float(item.start)
                end = float(item.end)
                if end >= VOCAL_ANCHOR - 0.75:
                    words.append((token, start, end))

    if not words:
        raise RuntimeError("No vocal words were detected after the 11-second intro.")

    # Lock the first detected vocal to the verified 11-second start.
    shift = VOCAL_ANCHOR - words[0][1]
    if abs(shift) <= 3.0:
        words = [(word, max(VOCAL_ANCHOR, start + shift), max(VOCAL_ANCHOR + 0.05, end + shift)) for word, start, end in words]
    else:
        words[0] = (words[0][0], VOCAL_ANCHOR, max(VOCAL_ANCHOR + 0.25, words[0][2]))
    return words


def lyric_lines() -> list[str]:
    return [line.strip() for line in LYRICS.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_line_timings(transcribed: list[tuple[str, float, float]]) -> list[dict]:
    lines = lyric_lines()
    counts = [len([word for word in line.split() if norm(word)]) for line in lines]
    total_lyric_words = sum(counts)
    total_detected_words = len(transcribed)

    if total_lyric_words < 2 or total_detected_words < 2:
        raise RuntimeError("Not enough lyric or detected vocal words.")

    duration = media_duration(AUDIO)

    def detected_index(lyric_word_index: int) -> int:
        ratio = lyric_word_index / (total_lyric_words - 1)
        return max(0, min(total_detected_words - 1, round(ratio * (total_detected_words - 1))))

    timings: list[dict] = []
    cursor = 0
    previous_end = VOCAL_ANCHOR - 0.03

    for line_no, (line, count) in enumerate(zip(lines, counts), start=1):
        first_lyric_word = cursor
        last_lyric_word = cursor + count - 1
        first_detected = detected_index(first_lyric_word)
        last_detected = detected_index(last_lyric_word)

        start = transcribed[first_detected][1] - 0.08
        end = transcribed[last_detected][2] + 0.22

        if line_no == 1:
            start = VOCAL_ANCHOR
        else:
            start = max(start, previous_end + 0.03)

        if end <= start:
            end = start + max(1.6, min(5.8, 0.48 * count + 0.8))

        max_hold = max(3.0, min(8.5, 0.62 * count + 1.6))
        end = min(end, start + max_hold, duration - 0.12)
        if end <= start:
            end = min(duration - 0.12, start + 1.8)

        timings.append({"text": line, "start": start, "end": end})
        print(f"LINE {line_no:02d}: {start:7.2f} -> {end:7.2f} | {line}")
        previous_end = end
        cursor += count

    last = timings[-1]
    last["end"] = min(duration - 0.12, max(last["end"], transcribed[-1][2] + 0.25))
    return timings


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


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
        events.append(
            f"Dialogue: 0,{ass_time(item['start'])},{ass_time(item['end'])},Lyrics,,0,0,0,,{{\\fad(150,150)}}{safe}"
        )
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

    timings = build_line_timings(transcribe())
    TIMINGS_FILE.write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")
    write_ass(timings)
    render()
    print(f"Created: {OUTPUT.name}")


if __name__ == "__main__":
    main()
