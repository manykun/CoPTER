#!/usr/bin/env python3
"""Plot epoch vs mean_reward curves from JSONL metric files."""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
SERIES = {
    "acc": Path("/root/paddlejob/workspace/yangziwen/CoPTER/sor/eval_acc_baseline_metrics.jsonl"),
    "sor": Path("/root/paddlejob/workspace/yangziwen/CoPTER/sor/eval_models_sor/eval_sor_metrics.jsonl"),
}

WIDTH = 1200
HEIGHT = 760
MARGIN_LEFT = 145
MARGIN_RIGHT = 50
MARGIN_TOP = 90
MARGIN_BOTTOM = 95
GRID_COLOR = (220, 224, 230)
AXIS_COLOR = (40, 40, 40)
TEXT_COLOR = (25, 25, 25)
BACKGROUND = (255, 255, 255)
COLORS = {
    "acc": (40, 105, 190),
    "sor": (220, 95, 35),
}


def load_metrics(path: Path) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    rewards: list[float] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if "epoch" not in record or "mean_reward" not in record:
                raise ValueError(f"{path}:{line_number} missing epoch or mean_reward")

            epochs.append(int(record["epoch"]))
            rewards.append(float(record["mean_reward"]))

    return epochs, rewards


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_ticks(min_value: float, max_value: float, count: int = 6) -> list[float]:
    if math.isclose(min_value, max_value):
        return [min_value]
    step = (max_value - min_value) / (count - 1)
    return [min_value + step * index for index in range(count)]


def plot_combined(series_data: dict[str, tuple[list[int], list[float]]]) -> None:
    all_epochs = [epoch for epochs, _ in series_data.values() for epoch in epochs]
    all_rewards = [reward for _, rewards in series_data.values() for reward in rewards]
    if not all_epochs or not all_rewards:
        raise ValueError("No metrics found")

    min_epoch, max_epoch = min(all_epochs), max(all_epochs)
    min_reward, max_reward = min(all_rewards), max(all_rewards)
    reward_padding = max((max_reward - min_reward) * 0.08, 0.005)
    y_min = min_reward - reward_padding
    y_max = max_reward + reward_padding

    plot_left = MARGIN_LEFT
    plot_right = WIDTH - MARGIN_RIGHT
    plot_top = MARGIN_TOP
    plot_bottom = HEIGHT - MARGIN_BOTTOM
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    def x_to_px(epoch: int) -> int:
        if min_epoch == max_epoch:
            return plot_left + plot_width // 2
        return int(plot_left + (epoch - min_epoch) / (max_epoch - min_epoch) * plot_width)

    def y_to_px(reward: float) -> int:
        if math.isclose(y_min, y_max):
            return plot_top + plot_height // 2
        return int(plot_bottom - (reward - y_min) / (y_max - y_min) * plot_height)

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    label_font = load_font(24)
    tick_font = load_font(18)
    legend_font = load_font(22)

    for tick in make_ticks(y_min, y_max):
        y = y_to_px(tick)
        draw.line((plot_left, y, plot_right, y), fill=GRID_COLOR, width=1)
        label = f"{tick:.3f}"
        text_box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((plot_left - 15 - (text_box[2] - text_box[0]), y - 10), label, fill=TEXT_COLOR, font=tick_font)

    unique_epochs = sorted(set(all_epochs))
    x_tick_count = min(8, len(unique_epochs))
    x_tick_indices = sorted({round(index * (len(unique_epochs) - 1) / max(x_tick_count - 1, 1)) for index in range(x_tick_count)})
    for index in x_tick_indices:
        epoch = unique_epochs[index]
        x = x_to_px(epoch)
        draw.line((x, plot_top, x, plot_bottom), fill=GRID_COLOR, width=1)
        label = str(epoch)
        text_box = draw.textbbox((0, 0), label, font=tick_font)
        draw.text((x - (text_box[2] - text_box[0]) / 2, plot_bottom + 12), label, fill=TEXT_COLOR, font=tick_font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS_COLOR, width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS_COLOR, width=2)

    for name, (epochs, rewards) in series_data.items():
        color = COLORS[name]
        points = [(x_to_px(epoch), y_to_px(reward)) for epoch, reward in zip(epochs, rewards)]
        if len(points) > 1:
            draw.line(points, fill=color, width=4, joint="curve")
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=BACKGROUND, width=2)

    title = "acc vs sor"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((WIDTH - (title_box[2] - title_box[0])) / 2, 24), title, fill=TEXT_COLOR, font=title_font)

    legend_x = plot_right - 190
    legend_y = plot_top + 18
    for name in SERIES:
        color = COLORS[name]
        draw.line((legend_x, legend_y + 13, legend_x + 44, legend_y + 13), fill=color, width=5)
        draw.ellipse((legend_x + 17, legend_y + 6, legend_x + 27, legend_y + 16), fill=color, outline=BACKGROUND, width=2)
        draw.text((legend_x + 58, legend_y), name, fill=TEXT_COLOR, font=legend_font)
        legend_y += 36

    x_label = "epoch"
    x_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((WIDTH - (x_box[2] - x_box[0])) / 2, HEIGHT - 50), x_label, fill=TEXT_COLOR, font=label_font)

    y_label = "mean_reward"
    y_label_image = Image.new("RGBA", (220, 45), (255, 255, 255, 0))
    y_label_draw = ImageDraw.Draw(y_label_image)
    y_label_draw.text((0, 0), y_label, fill=TEXT_COLOR, font=label_font)
    y_label_image = y_label_image.rotate(90, expand=True)
    image.paste(y_label_image, (18, plot_top + plot_height // 2 - y_label_image.height // 2), y_label_image)

    output_path = BASE_DIR / "acc_sor.png"
    image.save(output_path)
    print(f"Saved {output_path}")


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    series_data = {name: load_metrics(path) for name, path in SERIES.items()}
    plot_combined(series_data)


if __name__ == "__main__":
    main()
