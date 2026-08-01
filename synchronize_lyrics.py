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
DEVICE = "cpu"


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower().replace("’", "'"))


def media_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(cmd, text=True))
    return float(data["format"]["duration"])


def lyric_lines() -> list[str]:
    return [line.strip() for line in LYRICS.read_text(encoding="utf-8").splitlines() if line.strip()]


def forced_align_words(lines: list[str]) -> list[dict]:
    """Align the exact supplied lyrics to the audio with a phoneme CTC model."""
    duration = media_duration(AUDIO)
    transcript = " ".join(lines)
    audio = whisperx.load_audio(str(AUDIO))
    model_a, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

    # The transcript is known exactly. Give the aligner the full vocal region and
    # let its phoneme model find every word boundary in the performance.
    segments = [{
        "start": VOCAL_ANCHOR - 0.25,
        "end": duration - 0.05,
        "text": transcript,
    }]
    result = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        DEVICE,
        return_char_alignments=False,
    )

    words: list[dict] = []
    for item in result.get("word_segments", []):
        token = norm(str(item.get("word", "")))
        start = item.get("start")
        end = item.get("end")
        if token and start is not None and end is not None:
            words.append({
                "word": token,
                "start": float(start),
                "end": float(end),
                "score": float(item.get("score", 0.0) or 0.0),
            })

    if len(words) < 10:
        raise RuntimeError(f"Forced alignment returned too few words: {len(words)}")
    return words


def build_line_timings(lines: list[str], aligned_words: list[dict]) -> list[dict]:
    expected_counts = [len([w for w in re.findall(r"[A-Za-z0-9’']+", line) if norm(w)]) for line in lines]
    total_expected = sum(expected_counts)

    if len(aligned_words) < total_expected * 0.70:
        raise RuntimeError(
            f"Forced alignment found only {len(aligned_words)} of about {total_expected} lyric words."
        )

    # WhisperX aligns the exact supplied transcript in order. Use word counts to
    # convert its word boundaries into line boundaries without fuzzy matching.
    timings: list[dict] = []
    cursor = 0
    previous_end = VOCAL_ANCHOR - 0.05
    duration = media_duration(AUDIO)

    for line_no, (line, count) in enumerate(zip(lines, expected_counts), start=1):
        remaining_lines = len(lines) - line_no
        remaining_words_needed = sum(expected_counts[line_no:])
        available = len(aligned_words) - cursor

        # Keep enough words for every later lyric line even if a few aligner tokens
        # were omitted because of sung pronunciation.
        usable_count = min(count, max(1, available - remaining_words_needed))
        start_item = aligned_words[min(cursor, len(aligned_words) - 1)]
        end_index = min(len(aligned_words) - 1, cursor + usable_count - 1)
        end_item = aligned_words[end_index]

        start = float(start_item["start"]) - 0.10
        end = float(end_item["end"]) + 0.20
        if line_no == 1:
            start = VOCAL_ANCHOR
        else:
            start = max(start, previous_end + 0.02)
        end = min(duration - 0.08, max(end, start + 0.75))

        timings.append({
            "line": line_no,
            "text": line,
            "start": round(start, 3),
            "end": round(end, 3),
            "first_aligned_word": start_item["word"],
            "last_aligned_word": end_item["word"],
        })
        print(f"LINE {line_no:02d}: {start:7.2f} -> {end:7.2f} | {line}")
        previous_end = end
        cursor += usable_count

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
            f"Dialogue: 0,{ass_time(item['start'])},{ass_time(item['end'])},Lyrics,,0,0,0,,{{\\fad(100,100)}}{safe}"
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

    lines = lyric_lines()
    aligned_words = forced_align_words(lines)
    timings = build_line_timings(lines, aligned_words)
    TIMINGS_FILE.write_text(
        json.dumps({"method": "whisperx-forced-phoneme-alignment", "words": aligned_words, "lines": timings}, indent=2),
        encoding="utf-8",
    )
    write_ass(timings)
    render()
    print(f"Created: {OUTPUT.name}")


if __name__ == "__main__":
    main()
