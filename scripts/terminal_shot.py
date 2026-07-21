"""터미널 출력 텍스트를 스크린샷처럼 보이는 PNG 이미지로 렌더링하는 헬퍼.

실제 화면 캡처 대신, stdout으로 받은 텍스트를 macOS 터미널과 비슷한
스타일(어두운 배경, 모노스페이스, 상단 타이틀바)로 그려서 PDF 제출용
산출물 이미지를 만든다.
"""

import unicodedata

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 글리프가 없는 기본 폰트(DejaVu Sans)로는 한글이 깨져서(tofu box) 렌더링되므로
# 모노스페이스 + 한글을 모두 지원하는 폰트를 우선순위로 지정한다.
KOREAN_MONO_FONT = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Menlo", "monospace"]
BG_COLOR = "#1e1e1e"
TEXT_COLOR = "#d4d4d4"
TITLEBAR_COLOR = "#3a3a3a"


def _display_width(line: str) -> int:
    """한글 등 전각 문자를 2칸으로 계산한 표시 너비(모노스페이스 기준)."""
    width = 0
    for ch in line:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def save_terminal_screenshot(text: str, output_path: str, title: str = "Terminal") -> None:
    """text를 터미널 스크린샷 스타일의 PNG로 저장한다."""
    lines = text.rstrip("\n").split("\n")
    n_lines = max(len(lines), 1)
    max_len = max((_display_width(line) for line in lines), default=1)

    # 한글 폰트(AppleGothic 등)는 Menlo보다 줄 높이가 커서 여유 폭(1.2배)과
    # 바닥 여백을 넉넉히 잡아야 마지막 줄이 캔버스 밖으로 잘리지 않는다.
    char_w, char_h = 0.078, 0.22
    width = max(6.0, max_len * char_w + 1.0)
    height = n_lines * char_h * 1.2 + 1.0

    fig = plt.figure(figsize=(width, height), facecolor=BG_COLOR)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(BG_COLOR)

    titlebar_h = 0.55 / height
    ax.add_patch(plt.Rectangle((0, 1 - titlebar_h), 1, titlebar_h, color=TITLEBAR_COLOR, zorder=1))
    for i, color in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(plt.Circle((0.02 + i * 0.025, 1 - titlebar_h / 2), 0.006, color=color, zorder=2))
    ax.text(
        0.5,
        1 - titlebar_h / 2,
        title,
        color="#bbbbbb",
        fontsize=10,
        family=KOREAN_MONO_FONT,
        ha="center",
        va="center",
        zorder=2,
    )

    body_text = "\n".join(lines)
    ax.text(
        0.012,
        1 - titlebar_h - 0.02,
        body_text,
        color=TEXT_COLOR,
        fontsize=10,
        family=KOREAN_MONO_FONT,
        ha="left",
        va="top",
        linespacing=1.4,
        wrap=False,
    )

    fig.savefig(output_path, dpi=150, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"[screenshot] 저장 완료: {output_path}")
