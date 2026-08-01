from __future__ import annotations

import json
import re
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "My Own Heart (7).mp3"
VIDEO = ROOT / "My Own Heart.mp4"
LYRICS = ROOT / "MyOwnHeart_Lyrics.txt"
OUTPUT = ROOT / "My_Own_Heart_Synchronized.mp4"
ASS_FILE = ROOT / "my_own_heart.ass"
TIMINGS_FILE = ROOT / "my_own_heart_timings.json"


def normalize_word(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", text.lower().replace("’", "'"))


def lyric_lines() -> list[str]:
    lines = [line.strip() for line in LYRICS.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def lyric_words(lines: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    words: list[str] = []
    spans: list[tuple[int, int]] = []
    for line in lines:
        start = len(words)
        words.extend(re.findall(r"[A-Za-z0-9’']+", line))
        spans.append((start, len(words)))
    return words, spans


def transcribed_words() -> list[dict]:
    model = WhisperModel("medium.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(AUDIO),
        language="en",
        beam_size=5,
        best_of=5,
        temperature=0,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=True,
    )

    result: list[dict] = []
    for segment in segments:
        for word in segment.words or []:
            token = normalize_word(word.word)
            if token and word.start is not None and word.end is not None:
                result.append({"word": token, "start": float(word.start), "end": float(word.end)})
    return result


def align_words(expected_words: list[str], heard_words: list[dict]) -> list[dict | None]:
    expected = [normalize_word(word) for word in expected_words]
    heard = [item["word"] for item in heard_words]
    matcher = SequenceMatcher(None, expected, heard, autojunk=False)
    aligned: list[dict | None] = [None] * len(expected)

    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            aligned[block.a + offset] = heard_words[block.b + offset]

    matched = [i for i, item in enumerate(aligned) if item is not None]
    if not matched:
        raise RuntimeError("Whisper could not align any lyric words to the audio.")

    # Interpolate missing lyric words between reliably matched words.
    for index in range(len(aligned)):
        if aligned[index] is not None:
            continue
        previous = max((i for i in matched if i < index), default=None)
        following = min((i for i in matched if i > index), default=None)
        if previous is not None and following is not None:
            left = aligned[previous]
            right = aligned[following]
            assert left is not None and right is not None
            fraction = (index - previous) / (following - previous)
            center = left["end"] + (right["start"] - left["end"]) * fraction
            duration = max(0.18, min(0.55, (right["start"] - left["end"]) / max(1, following - previous)))
            aligned[index] = {"word": expected[index], "start": center, "end": center + duration}
        elif previous is not None:
            left = aligned[previous]
            assert left is not None
            start = left["end"] + 0.12 * (index - previous)
            aligned[index] = {"word": expected[index], "start": start, "end": start + 0.35}
        elif following is not None:
            right = aligned[following]
            assert right is not None
            end = max(0.1, right["start"] - 0.12 * (following - index))
            aligned[index] = {"word": expected[index], "start": max(0.0, end - 0.35), "end": end}

    return aligned


def line_timings(lines: list[str], spans: list[tuple[int, int]], aligned: list[dict | None]) -> list[dict]:
    timings: list[dict] = []
    for line, (start_index, end_index) in zip(lines, spans):
        entries = [item for item in aligned[start_index:end_index] if item is not None]
        if not entries:
            continue
        start = max(0.0, entries[0]["start"] - 0.12)
        end = entries[-1]["end"] + 0.28
        if timings:
            start = max(start, timings[-1]["end"] + 0.03)
        end = max(end, start + 0.8)
        timings.append({"text": line, "start": start, "end": end})
    return timings


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def escape_ass(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")


def write_ass(timings: list[dict]) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Lyrics,Arial,54,&H00FFFFFF,&H000000FF,&H00141414,&H78000000,-1,0,0,0,100,100,0,0,1,3,1,2,120,120,95,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for item in timings:
        events.append(
            f"Dialogue: 0,{ass_time(item['start'])},{ass_time(item['end'])},Lyrics,,0,0,0,,{escape_ass(item['text'])}"
        )
    ASS_FILE.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def render() -> None:
    # Use the supplied MP4 as the visual source, replace its audio with the final MP3,
    # and burn the synchronized ASS subtitles into the picture.
    subtitle_filter = f"ass={ASS_FILE.name}"
    run([
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", str(VIDEO),
        "-i", str(AUDIO),
        "-vf", subtitle_filter,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "320k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-shortest", str(OUTPUT),
    ])


def main() -> None:
    for path in (AUDIO, VIDEO, LYRICS):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.name}")

    lines = lyric_lines()
    expected_words, spans = lyric_words(lines)
    heard_words = transcribed_words()
    aligned = align_words(expected_words, heard_words)
    timings = line_timings(lines, spans, aligned)

    TIMINGS_FILE.write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")
    write_ass(timings)
    render()
    print(f"Created: {OUTPUT.name}")


if __name__ == "__main__":
    main()
