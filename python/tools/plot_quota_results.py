#!/usr/bin/env python3
"""Draw five reproducible SVG figures from ferro-sim's quota summary CSV."""

from __future__ import annotations

import argparse
import csv
from html import escape
from pathlib import Path

QUOTAS = ["", "1", "2", "4"]
QUOTA_LABELS = ["Unlimited", "1", "2", "4"]
COLORS = ["#b33b3b", "#2463a6", "#548235", "#8856a7", "#dd7e00", "#008b8b"]


def read_summary(path: Path) -> dict[tuple[str, str, str], dict[str, float | None]]:
    result: dict[tuple[str, str, str], dict[str, float | None]] = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            quota = row["quota_gpus"]
            key = (row["scenario"], quota, row["metric"])
            result[key] = {
                name: float(row[field]) if row[field] else None
                for name, field in (
                    ("mean", "mean"),
                    ("low", "ci95_low"),
                    ("high", "ci95_high"),
                )
            }
    return result


def chart(
    output: Path,
    title: str,
    y_label: str,
    data: dict[tuple[str, str, str], dict[str, float | None]],
    series: list[tuple[str, str, str, str]],
) -> None:
    width, height = 1120, 650
    left, top, right, bottom = 100, 86, 330, 112
    plot_w, plot_h = width - left - right, height - top - bottom
    points: list[tuple[float, float, float, float, int]] = []
    maximum = 0.0
    for series_index, (scenario, metric, label, _) in enumerate(series):
        for x, quota in enumerate(QUOTAS):
            value = data.get((scenario, quota, metric))
            if not value or value["mean"] is None:
                continue
            mean = float(value["mean"])
            low = float(value["low"] if value["low"] is not None else mean)
            high = float(value["high"] if value["high"] is not None else mean)
            points.append((x, mean, low, high, series_index))
            maximum = max(maximum, high)
    if not points:
        raise ValueError(f"no data available for {output.name}")
    y_max = maximum * 1.12 if maximum > 0 else 1.0
    y_min = 0.0

    def px(x: float) -> float:
        return left + x * plot_w / (len(QUOTAS) - 1)

    def py(value: float) -> float:
        return top + plot_h * (1.0 - (value - y_min) / (y_max - y_min))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#333;stroke-width:1.5}</style>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-size="24" font-weight="bold">{escape(title)}</text>',
    ]

    tick_count = 5
    for i in range(tick_count + 1):
        value = y_max * i / tick_count
        y = py(value)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="14">{value:.0f}</text>')
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<text x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle" font-size="17">Per-user hard quota (GPUs; unlimited baseline)</text>',
            f'<text x="26" y="{top + plot_h / 2}" text-anchor="middle" font-size="17" transform="rotate(-90 26 {top + plot_h / 2})">{escape(y_label)}</text>',
        ]
    )
    for index, label in enumerate(QUOTA_LABELS):
        svg.append(f'<text x="{px(index):.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-size="14">{label}</text>')

    for series_index, (scenario, metric, label, color) in enumerate(series):
        points_for_series = [p for p in points if p[4] == series_index]
        path = " ".join(
            ("M" if i == 0 else "L") + f" {px(x):.2f} {py(mean):.2f}"
            for i, (x, mean, _, _, _) in enumerate(points_for_series)
        )
        if path:
            svg.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, mean, low, high, _ in points_for_series:
            cx, y_low, y_high = px(x), py(low), py(high)
            svg.append(f'<line x1="{cx:.2f}" y1="{y_low:.2f}" x2="{cx:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<line x1="{cx - 6:.2f}" y1="{y_low:.2f}" x2="{cx + 6:.2f}" y2="{y_low:.2f}" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<line x1="{cx - 6:.2f}" y1="{y_high:.2f}" x2="{cx + 6:.2f}" y2="{y_high:.2f}" stroke="{color}" stroke-width="2"/>')
            svg.append(f'<circle cx="{cx:.2f}" cy="{py(mean):.2f}" r="5" fill="{color}"/>')

    legend_x, legend_y = left + plot_w + 36, top + 12
    for i, (_, _, label, color) in enumerate(series):
        y = legend_y + i * 34
        svg.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 30}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{legend_x + 40}" y="{y + 5}" font-size="14">{escape(label)}</text>')

    svg.append('<text x="100" y="620" font-size="12" fill="#555">Points are across-seed means; bars show two-sided 95% Student-t confidence intervals.</text>')
    svg.append("</svg>")
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = read_summary(args.input)
    args.out.mkdir(parents=True, exist_ok=True)

    chart(
        args.out / "quota_heavy_user_wait.svg",
        "Heavy-user mean waiting time under dominant-user workloads",
        "Mean per-user wait (seconds)",
        data,
        [
            ("B-heavy", "heavy_user_mean_wait_s", "B heavy user", COLORS[0]),
            ("C-burst", "heavy_user_mean_wait_s", "C hog user", COLORS[2]),
        ],
    )
    chart(
        args.out / "quota_normal_user_wait.svg",
        "Other-user mean waiting time under dominant-user workloads",
        "Mean per-user wait (seconds)",
        data,
        [
            ("B-heavy", "normal_users_mean_wait_s", "B normal users", COLORS[1]),
            ("C-burst", "normal_users_mean_wait_s", "C light users", COLORS[3]),
        ],
    )
    chart(
        args.out / "quota_gpu_utilisation.svg",
        "Quota and GPU utilisation by workload",
        "GPU utilisation (percent)",
        data,
        [
            ("A-balanced", "gpu_utilisation_percent", "A balanced", COLORS[0]),
            ("B-heavy", "gpu_utilisation_percent", "B heavy user", COLORS[1]),
            ("C-burst", "gpu_utilisation_percent", "C burst", COLORS[2]),
            ("D-single-user", "gpu_utilisation_percent", "D single user", COLORS[3]),
        ],
    )
    chart(
        args.out / "quota_wait_ratio.svg",
        "Heavy-to-normal mean waiting-time ratio",
        "Heavy / normal mean wait (ratio)",
        data,
        [
            ("B-heavy", "heavy_to_normal_wait_ratio", "B heavy user", COLORS[0]),
            ("C-burst", "heavy_to_normal_wait_ratio", "C burst user", COLORS[2]),
        ],
    )
    chart(
        args.out / "quota_throughput.svg",
        "Quota and completed-job throughput",
        "Completed jobs per hour",
        data,
        [
            ("A-balanced", "throughput_per_hour", "A balanced", COLORS[0]),
            ("B-heavy", "throughput_per_hour", "B heavy user", COLORS[1]),
            ("C-burst", "throughput_per_hour", "C burst", COLORS[2]),
            ("D-single-user", "throughput_per_hour", "D single user", COLORS[3]),
        ],
    )
    for path in sorted(args.out.glob("*.svg")):
        print(path)


if __name__ == "__main__":
    main()
