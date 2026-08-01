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


def normalize_word(text: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", text.lower().replace("’", "'"))


def lyric_lines() -> list[str]:
    return [line.strip() for line in LYRICS.read_text(encoding="utf-8").splitlines() if line.strip()]


def lyric_words(lines: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    words: list[str] = []
    spans: list[tuple[int, int]] = []
    for line in lines:
        start = len(words)
        words.extend(re.findall(r"[A-Za-z0-9’']+", line))
        spans.append((start, len(words)))
    return words, spans


def transcribed_words(prompt: str) -> list[dict]:
    model = WhisperModel("medium.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        str(AUDIO),
        language="en",
        beam_size=5,
        temperature=0,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=prompt,
    )
    result: list[dict] = []
    for segment in segments:
        for word in segment.words or []:
            token = normalize_word(word.word)
            if token and word.start is not None and word.end is not None:
                result.append({"word": token, "start": float(word.start), "end": float(word.end)})
    return result


def match_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a.rstrip("s") == b.rstrip("s"):
        return 0.35
    if a[:4] == b[:4] and len(a) >= 4 and len(b) >= 4:
        return 0.55
    return 1.2


def align_words(expected_words: list[str], heard_words: list[dict]) -> list[dict | None]:
    expected = [normalize_word(w) for w in expected_words]
    heard = [w["word"] for w in heard_words]
    n, m = len(expected), len(heard)
    gap = 0.9
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap
        back[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = j * gap
        back[0][j] = "L"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + match_cost(expected[i - 1], heard[j - 1])
            up = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap
            best = min(diag, up, left)
            dp[i][j] = best
            back[i][j] = "D" if best == diag else ("U" if best == up else "L")

    aligned: list[dict | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "D":
            if match_cost(expected[i - 1], heard[j - 1]) <= 0.65:
                aligned[i - 1] = heard_words[j - 1]
            i -= 1
            j -= 1
        elif move == "U":
            i -= 1
        else:
            j -= 1

    matched = [k for k, item in enumerate(aligned) if item is not None]
    if len(matched) < max(8, n // 5):
        raise RuntimeError("Not enough lyric words matched the vocal track.")

    for k in range(n):
        if aligned[k] is not None:
            continue
        prev = max((x for x in matched if x < k), default=None)
        nxt = min((x for x in matched if x > k), default=None)
        if prev is not None and nxt is not None:
            left = aligned[prev]
            right = aligned[nxt]
            assert left and right
            fraction = (k - prev) / (nxt - prev)
            start = left["end"] + (right["start"] - left["end"]) * fraction
            aligned[k] = {"word": expected[k], "start": max(left["end"], start), "end": max(left["end"] + 0.18, start + 0.28)}
        elif prev is not None:
            left = aligned[prev]
            assert left
            start = left["end"] + 0.22 * (k - prev)
            aligned[k] = {"word": expected[k], "start": start, "end": start + 0.3}
        elif nxt is not None:
            right = aligned[nxt]
            assert right
            end = max(0.1, right["start"] - 0.22 * (nxt - k))
            aligned[k] = {"word": expected[k], "start": max(0.0, end - 0.3), "end": end}
    return aligned


def line_timings(lines: list[str], spans: list[tuple[int, int]], aligned: list[dict | None]) -> list[dict]:
    timings: list[dict] = []
    for line, (a, b) in zip(lines, spans):
        entries = [x for x in aligned[a:b] if x is not None]
        if not entries:
            continue
        start = max(0.0, entries[0]["start"] - 0.08)
        end = entries[-1]["end"] + 0.18
        if timings:
            start = max(start, timings[-1]["end"] + 0.02)
        end = max(end, start + 0.65)
        timings.append({"text": line, "start": start, "end": end})
    return timings


def ass_time(seconds: float) -> str:
    cs = max(0, round(seconds * 100))
    h, rem = divmod(cs, 360000)
    minute, rem = divmod(rem, 6000)
    sec, cent = divmod(rem, 100)
    return f"{h}:{minute:02d}:{sec:02d}.{cent:02d}"


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
    events = [f"Dialogue: 0,{ass_time(x['start'])},{ass_time(x['end'])},Lyrics,,0,0,0,,{escape_ass(x['text'])}" for x in timings]
    ASS_FILE.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def render() -> None:
    subprocess.run([
        "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(VIDEO), "-i", str(AUDIO),
        "-vf", f"ass={ASS_FILE.name}", "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "320k", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-shortest", str(OUTPUT)
    ], check=True)


def main() -> None:
    for path in (AUDIO, VIDEO, LYRICS):
        if not path.exists():
            raise FileNotFoundError(path.name)
    lines = lyric_lines()
    words, spans = lyric_words(lines)
    heard = transcribed_words(" ".join(lines))
    aligned = align_words(words, heard)
    timings = line_timings(lines, spans, aligned)
    TIMINGS_FILE.write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")
    write_ass(timings)
    render()
    print(f"Created: {OUTPUT.name}")


if __name__ == "__main__":
    main()
