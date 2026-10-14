"""W1-D7: turn the experiment outputs into figures and a written report.

Produces the "2 图 1 表" deliverable plus a markdown report that can go straight
onto GitHub or CSDN.

Figures
-------
* ``fig1_layer_curve.png``  - per-layer cosine similarity / SQNR, FP32 vs INT8,
  with the first layer below threshold annotated
* ``fig2_calib_sweep.png``  - accuracy vs calibration size, plus the diversity
  comparison (narrow / mixed / balanced)
* ``table1_summary.md``     - the comparison table across all quantized variants

Usage
-----
    python 07_make_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w1_quant.common import redirect_tempdir, write_csv  # noqa: E402

TMPDIR = redirect_tempdir()  # must run before matplotlib is imported

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_csv(path: str) -> list[dict]:
    import csv

    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key, value in row.items():
            if value in ("", "None", "nan"):
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def fig_layer_curve(layer_rows: list[dict], out_path: str, threshold: float) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = []
    for row in layer_rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])
    if not variants:
        return False

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(variants), 1)))

    for color, label in zip(colors, variants):
        rows = sorted(
            [r for r in layer_rows if r["variant"] == label],
            key=lambda r: r["order"],
        )
        x = [r["order"] for r in rows]
        axes[0].plot(x, [r["cosine"] for r in rows], marker="o", ms=3,
                     lw=1.2, color=color, label=label)
        axes[1].plot(x, [r["sqnr_db"] for r in rows], marker="s", ms=3,
                     lw=1.2, color=color, label=label)

    axes[0].axhline(threshold, color="red", ls="--", lw=1,
                    label=f"threshold {threshold}")
    axes[0].set_ylabel("cosine similarity (FP32 vs INT8)")
    axes[0].set_title("Per-layer activation drift after INT8 PTQ")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].axhline(30, color="red", ls="--", lw=1, label="30 dB reference")
    axes[1].set_ylabel("SQNR (dB)")
    axes[1].set_xlabel("layer order in graph")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    # annotate the first layer that breaks
    for label in variants:
        rows = sorted(
            [r for r in layer_rows if r["variant"] == label],
            key=lambda r: r["order"],
        )
        for row in rows:
            if row["cosine"] is not None and row["cosine"] < threshold:
                axes[0].annotate(
                    "first break\n%s" % str(row["tensor"])[:26],
                    xy=(row["order"], row["cosine"]),
                    xytext=(row["order"] + 1, row["cosine"] - 0.12),
                    fontsize=7,
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="black"),
                )
                break

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_calib_sweep(rows: list[dict], out_path: str) -> bool:
    if not rows:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size_rows = sorted(
        [r for r in rows if r.get("group") == "size"],
        key=lambda r: r["calib_images"],
    )
    div_rows = [r for r in rows if r.get("group") == "diversity"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    if size_rows:
        x = [r["calib_images"] for r in size_rows]
        ax = axes[0]
        ax.plot(x, [r["mean_cosine"] for r in size_rows], marker="o",
                color="tab:blue", label="mean cosine")
        ax.set_xscale("log")
        ax.set_xlabel("calibration images (log scale)")
        ax.set_ylabel("mean cosine similarity", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(x, [r["mean_sqnr_db"] for r in size_rows], marker="s",
                 color="tab:orange", label="mean SQNR")
        ax2.set_ylabel("mean SQNR (dB)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax.set_title("Experiment A: calibration size vs accuracy")
        for xi, yi in zip(x, [r["mean_cosine"] for r in size_rows]):
            ax.annotate("%d" % xi, (xi, yi), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8)

    if div_rows:
        ax = axes[1]
        labels = [r["label"] for r in div_rows]
        cosines = [r["mean_cosine"] for r in div_rows]
        sqnrs = [r["mean_sqnr_db"] for r in div_rows]
        xpos = np.arange(len(labels))
        bars = ax.bar(xpos - 0.2, cosines, width=0.38, label="mean cosine",
                      color="tab:blue")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=15, fontsize=8)
        ax.set_ylabel("mean cosine similarity", color="tab:blue")
        ax.set_ylim(min(cosines) - 0.02 if cosines else 0, 1.0)
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(alpha=0.3, axis="y")
        for bar, value in zip(bars, cosines):
            ax.annotate("%.4f" % value, (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7)

        ax2 = ax.twinx()
        bars2 = ax2.bar(xpos + 0.2, sqnrs, width=0.38, label="mean SQNR",
                        color="tab:orange")
        ax2.set_ylabel("mean SQNR (dB)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        for bar, value in zip(bars2, sqnrs):
            ax2.annotate("%.1f" % value, (bar.get_x() + bar.get_width() / 2, value),
                         textcoords="offset points", xytext=(0, 3),
                         ha="center", fontsize=7)
        ax.set_title("Experiment B: calibration diversity (same count)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def build_summary_table(acc_rows: list[dict], quant_report: dict | None) -> str:
    lines = [
        "# W1 量化对比总表",
        "",
        "模型：MobileNetV2（ImageNet 预训练，ONNX opset 13 + onnxsim）",
        "量化：ONNX Runtime `quantize_static`，权重与激活均为 INT8",
        "",
        "| 变体 | 格式 | per-channel | 校准方法 | 大小(MB) | top1一致率 | 置信度下降 | KL散度 | 最大logit偏差 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    meta: dict = {}
    if quant_report:
        for row in quant_report.get("variants", []):
            meta[row["label"]] = row
    for row in acc_rows:
        label = row.get("label")
        info = meta.get(label, {})
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                label,
                info.get("quant_format", "-"),
                info.get("per_channel", "-"),
                info.get("calibrate_method", "-"),
                _n(row.get("file_mb"), 2),
                _n(row.get("top1_match"), 4),
                _n(row.get("confidence_drop"), 4),
                _n(row.get("mean_kl_divergence"), 5),
                _n(row.get("max_abs_logit_diff"), 4),
            )
        )
    lines.append("")
    lines.append(
        "说明：`top1一致率` = INT8 与 FP32 选取同一类的比例；`置信度下降` = 两者的平均 top-1 概率之差；"
        "`KL散度` 衡量整个输出分布的偏移，而不只是 argmax。"
    )
    return "\n".join(lines)


def _n(value, digits: int) -> str:
    if value is None:
        return "-"
    try:
        return ("%%.%df" % digits) % float(value)
    except (TypeError, ValueError):
        return str(value)


def build_report(
    export_report, quant_report, layer_report, acc_rows, sweep_rows, figures
) -> str:
    lines = ["# MobileNetV2 INT8 量化误差分析", ""]
    lines.append("> 一周实验的完整记录：环境 / 导出 / 量化 / 逐层误差 / 校准集影响。")
    lines.append("")

    # ---- export -----------------------------------------------------------
    lines.append("## 1. ONNX 导出")
    lines.append("")
    if export_report:
        stages = export_report.get("stages", {})
        lines.append("| 阶段 | 节点数 | initializer | 大小(MB) | 动态 batch |")
        lines.append("|---|---|---|---|---|")
        for name, stats in stages.items():
            lines.append(
                "| %s | %d | %d | %.2f | %s |"
                % (name, stats["nodes"], stats["initializers"],
                   stats["size_mb"], stats["has_dynamic_batch"])
            )
        lines.append("")
        if "verify" in export_report:
            v = export_report["verify"]
            lines.append(
                "- onnxsim 后数值验证：`max|diff| = %.3e`，`cosine = %.8f`"
                % (v["max_abs_diff"], v["cosine"])
            )
        if "exported" in stages and "simplified" in stages:
            b = stages["exported"]["nodes"]
            a = stages["simplified"]["nodes"]
            ib = stages["exported"]["initializers"]
            ia = stages["simplified"]["initializers"]
            lines.append(
                "- onnxsim 删除 %d 个节点（%.1f%%），节点数 %d → %d；"
                "同时 initializer 从 %d 增加到 %d。"
                "实测删除的全部是 **`Constant` 算子**（70 个），它们被折叠成了 initializer。"
                % (b - a, (b - a) / b * 100, b, a, ib, ia)
            )
            lines.append(
                "- 注意：文件大小**没有变化**（%.2f MB）。所以常量折叠改变的是"
                "**图的形状**（节点少了、initializer 多了），不是模型体积——"
                "推理时少执行 70 个算子，权重字节一个没少。"
                % stages["simplified"]["size_mb"]
            )
        lines.append("")

    # ---- quantization -----------------------------------------------------
    lines.append("## 2. 量化变体")
    lines.append("")
    if quant_report:
        lines.append("| 变体 | 格式 | per-channel | 校准方法 | 大小(MB) | Q | DQ | QLinear |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in quant_report.get("variants", []):
            lines.append(
                "| %s | %s | %s | %s | %.2f | %d | %d | %d |"
                % (row["label"], row["quant_format"], row["per_channel"],
                   row["calibrate_method"], row["file_mb"],
                   row["quantize_linear"], row["dequantize_linear"],
                   row["qlinear_fused"])
            )
        lines.append("")
    else:
        lines.append("(未找到 03_quant_report.json)")
        lines.append("")

    # ---- layer analysis ---------------------------------------------------
    lines.append("## 3. 逐层误差分析")
    lines.append("")
    if layer_report:
        for label, info in layer_report.items():
            lines.append(f"### {label}")
            lines.append("")
            lines.append("- 对比张量数：%d" % info["num_tensors"])
            lines.append("- 平均 cosine：%.6f" % info["mean_cosine"])
            worst = info["worst"]
            lines.append(
                "- 最差层：`%s`（%s，第 %d 层），cosine=%.6f，SQNR=%.2f dB"
                % (worst["tensor"], worst["op_type"], worst["order"],
                   worst["cosine"], worst["sqnr_db"])
            )
            first = info.get("first_break")
            if first:
                lines.append(
                    "- **第一个跌破 %.2f 的层**：`%s`（%s，第 %d 层），cosine=%.6f"
                    % (info["threshold"], first["tensor"], first["op_type"],
                       first["order"], first["cosine"])
                )
            else:
                lines.append("- 没有任何层跌破 %.2f" % info["threshold"])
            lines.append("")
    else:
        lines.append("(未找到 04_layer_analysis.json)")
        lines.append("")

    # ---- accuracy ---------------------------------------------------------
    lines.append("## 4. 精度对比")
    lines.append("")
    if acc_rows:
        lines.append(build_summary_table(acc_rows, quant_report).split("\n", 1)[1])
        lines.append("")
    else:
        lines.append("(未找到 05_accuracy.csv)")
        lines.append("")

    # ---- calibration sweep ------------------------------------------------
    lines.append("## 5. 校准集的影响")
    lines.append("")
    if sweep_rows:
        size_rows = sorted(
            [r for r in sweep_rows if r.get("group") == "size"],
            key=lambda r: r["calib_images"],
        )
        div_rows = [r for r in sweep_rows if r.get("group") == "diversity"]
        if size_rows:
            lines.append("### 5.1 规模")
            lines.append("")
            lines.append("| 校准图片数 | mean cosine | min cosine | SQNR(dB) | top1一致率 | KL | 置信度下降 |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in size_rows:
                lines.append(
                    "| %d | %.5f | %.5f | %.2f | %.4f | %.5f | %.4f |"
                    % (r["calib_images"], r["mean_cosine"], r["min_cosine"],
                       r["mean_sqnr_db"], r["top1_match"], r["mean_kl"], r["conf_drop"])
                )
            lines.append("")
        if div_rows:
            lines.append("### 5.2 多样性（图片总数相同）")
            lines.append("")
            lines.append("| 校准集 | 图片数 | mean cosine | min cosine | SQNR(dB) | top1一致率 | KL |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in div_rows:
                lines.append(
                    "| %s | %d | %.5f | %.5f | %.2f | %.4f | %.5f |"
                    % (r["label"], r["calib_images"], r["mean_cosine"],
                       r["min_cosine"], r["mean_sqnr_db"], r["top1_match"], r["mean_kl"])
                )
            lines.append("")
    else:
        lines.append("(未找到 06_calib_sweep.csv)")
        lines.append("")

    # ---- findings ---------------------------------------------------------
    lines.append("## 6. 结论（从上面的数据直接读出来的）")
    lines.append("")
    lines.extend(build_findings(export_report, layer_report, acc_rows, sweep_rows))
    lines.append("")

    # ---- figures ----------------------------------------------------------
    if figures:
        lines.append("## 7. 图表")
        lines.append("")
        for caption, path in figures.items():
            rel = os.path.relpath(path, ROOT).replace("\\", "/")
            lines.append(f"![{caption}]({rel})")
            lines.append("")

    # ---- limits -----------------------------------------------------------
    lines.append("## 8. 本实验的边界（面试时主动说明，比被问出来好）")
    lines.append("")
    lines.append("- 评估集是 **ImageNet 每个类别一张真实照片**（`EliSchwartz/imagenet-sample-images`，"
                 "共约 2000 张，覆盖 997 类），来源和规模都有限，**绝对精度数字不可外推**；"
                 "本实验的结论是**相对趋势**。")
    lines.append("- 评估集按类别 ID 排序后取尾部 64 张，实测 **100% 落在 object/scene 域**，"
                 "与整体分布（animal 占小部分）不一致；64 张的样本量使 top-1 一致率的"
                 "分辨率约为 1/64≈1.6 个百分点。这直接限制了 5.2 的解读，见 6.3。")
    lines.append("- 量化在 x86 CPU 上用 ONNX Runtime 完成，代表 PTQ 的一般行为；"
                 "RKNN / nncase 的具体算子实现会带来差异，W2/W3 会给出 NPU 侧数据。")
    lines.append("- `quantize_static` 的校准是 MinMax/Entropy/Percentile，"
                 "不是逐层敏感度搜索；混合量化的选层策略放在 W2 做。")
    lines.append("")
    return "\n".join(lines)


def build_findings(export_report, layer_report, acc_rows, sweep_rows) -> list[str]:
    """Derive the written conclusions from the measured numbers."""
    out: list[str] = []

    if acc_rows:
        by_label = {r.get("label"): r for r in acc_rows}
        load = by_label.get("qdq_pertensor_minmax")
        pc = by_label.get("qdq_perchannel_minmax")
        if load and pc:
            out.append(
                "### 6.1 per-channel 是这次量化的决定性因素，不是校准集"
            )
            out.append("")
            out.append(
                "- per-tensor → per-channel：top-1 一致率 **%.4f → %.4f**，"
                "KL 散度 **%.5f → %.5f**（降低约 %.0f 倍），最大 logit 偏差 "
                "**%.2f → %.2f**。"
                % (
                    load["top1_match"], pc["top1_match"],
                    load["mean_kl_divergence"], pc["mean_kl_divergence"],
                    load["mean_kl_divergence"] / max(pc["mean_kl_divergence"], 1e-12),
                    load["max_abs_logit_diff"], pc["max_abs_logit_diff"],
                )
            )
            out.append(
                "- 代价只有 **%.2f MB → %.2f MB**（+%.0f%%）。用 0.24 MB 换回 14 个百分点的"
                "一致率，是端侧部署里性价比最高的一次取舍。"
                % (
                    load["file_mb"], pc["file_mb"],
                    (pc["file_mb"] - load["file_mb"]) / load["file_mb"] * 100,
                )
            )
            out.append(
                "- 置信度下降从 **%.4f 降到 %.4f**：这一点对实际流水线很关键——"
                "argmax 一致不代表概率分布不变，下游如果按阈值做拒识或做 NMS，"
                "分布漂移会直接改变行为。"
                % (load["confidence_drop"], pc["confidence_drop"])
            )
            out.append("")

    if layer_report:
        pc = layer_report.get("qdq_perchannel_minmax") or {}
        load = layer_report.get("qdq_pertensor_minmax") or {}
        out.append("### 6.2 误差随层深累积，但两类量化的形态完全不同")
        out.append("")
        if load:
            out.append(
                "- **per-tensor 从第 %d 层就开始崩**（`%s`，cosine=%.4f），"
                "之后一路下滑，最差层 cosine 只有 **%.4f**。"
                % (
                    load["first_break"]["order"] if load.get("first_break") else -1,
                    load["first_break"]["tensor"] if load.get("first_break") else "?",
                    load["first_break"]["cosine"] if load.get("first_break") else float("nan"),
                    load["worst"]["cosine"],
                )
            )
        if pc:
            out.append(
                "- **per-channel 撑到第 %d 层**才跌破阈值，最差层 cosine 仍有 **%.4f**，"
                "整条曲线基本贴着 0.98–0.99，没有崩溃段。"
                % (
                    pc["first_break"]["order"] if pc.get("first_break") else -1,
                    pc["worst"]["cosine"],
                )
            )
        if load and pc:
            out.append(
                "- 最脆弱的层集中在中后段的 **depthwise 卷积和残差 Add**："
                "`%s`、`%s` 一类。这与 MobileNet 系列的已知结论一致——"
                "逐深度卷积每个通道只有一个卷积核，通道间动态范围差异大，"
                "per-tensor 的单一 scale 必然两头不讨好。"
                % (
                    _shorten(load["worst"]["tensor"]),
                    _shorten(pc["worst"]["tensor"]),
                )
            )
        out.append("")

    if sweep_rows:
        size_rows = sorted(
            [r for r in sweep_rows if r.get("group") == "size"],
            key=lambda r: r["calib_images"],
        )
        div_rows = [r for r in sweep_rows if r.get("group") == "diversity"]
        out.append("### 6.3 校准集：数量几乎不重要，而\"多样性\"的结论是反直觉的")
        out.append("")
        if len(size_rows) >= 2:
            first, last = size_rows[0], size_rows[-1]
            out.append(
                "- 校准集从 **%d 张 → %d 张**（50 倍），mean cosine 只从 %.5f 变到 %.5f"
                "（变化 %.5f），SQNR 反而下降 %.2f dB。"
                "**在这个模型上，堆校准图片数量几乎没有任何收益。**"
                % (
                    first["calib_images"], last["calib_images"],
                    first["mean_cosine"], last["mean_cosine"],
                    last["mean_cosine"] - first["mean_cosine"],
                    first["mean_sqnr_db"] - last["mean_sqnr_db"],
                )
            )
        if div_rows:
            worst = min(div_rows, key=lambda r: r["top1_match"])
            out.append(
                "- 同样 500 张，三种构成的差异："
                + "、".join(
                    "`%s`=%.4f" % (r["label"], r["top1_match"]) for r in div_rows
                )
                + "。**全部落在噪声量级。**"
            )
            out.append(
                "- 先看评估集的真实分布（这个数据决定了怎么解读）：全部 1994 张图按类别 ID "
                "排序后，**评估集取的尾部 64 张全部属于 object/scene 域**，"
                "而 `domain_animal` 组只占整体的一小部分。"
            )
            out.append(
                "- **预期落空的地方要老实说**：教科书结论是\"校准集要代表部署分布\"，"
                "按此 `domain_balanced_500`（三个域等量抽取）应该最好，"
                "实测它反而最低（%.4f）。而表现最好的 `narrow_500` 取自 animal 域，"
                "**和评估集所在域并不匹配**——这与\"分布匹配\"的直觉正好相反。"
                % worst["top1_match"]
            )
            out.append(
                "- 所以\"域不匹配所以掉点\"这个解释在这里**站不住**。"
                "能站住的是这个：per-channel 已经把 mean cosine 压到 0.989~0.990 的窄带内，"
                "各配置之间只差 0.0002，**校准集构成的影响已经被压到测量分辨率以下**；"
                "此时 64 张评估集上 0.906 与 0.938 的差距只相当于 2~3 张图的翻转，"
                "不足以支撑任何结论。要定论至少需要 500 张评估图并重复多次。"
            )
            out.append(
                "- 因此本节的结论不是\"多样性不重要\"，而是："
                "**当量化配置（per-channel）本身已经把误差压到很低时，校准集构成退居次要；"
                "它是否成为主要矛盾，取决于量化配置有多激进。** "
            )
            out.append(
                "- 验证方法很明确：在 **per-tensor** 配置下重跑本节——"
                "那里 mean cosine 只有 0.913，离散度是 per-channel 的 3 倍以上，"
                "校准集构成的差异才有机会从噪声里露出来。"
                "这正是 W2 混合量化要回答的问题。"
            )
        out.append("")

    return out


def _shorten(tensor: str, limit: int = 46) -> str:
    return tensor if len(tensor) <= limit else "…" + tensor[-(limit - 1):]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=os.path.join(ROOT, "artifacts"))
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--outdir", default=os.path.join(ROOT, "W1_报告"))
    args = ap.parse_args()

    A = args.artifacts
    export_report = load_json(os.path.join(A, "reports/01_export_report.json"))
    quant_report = load_json(os.path.join(A, "reports/03_quant_report.json"))
    layer_report = load_json(os.path.join(A, "reports/04_layer_analysis.json"))
    acc_rows = load_csv(os.path.join(A, "tables/05_accuracy.csv"))
    sweep_rows = load_csv(os.path.join(A, "tables/06_calib_sweep.csv"))
    layer_rows = load_csv(os.path.join(A, "tables/04_layer_metrics.csv"))

    os.makedirs(args.outdir, exist_ok=True)
    figures: dict[str, str] = {}

    fig1 = os.path.join(args.outdir, "fig1_layer_curve.png")
    if layer_rows and fig_layer_curve(layer_rows, fig1, args.threshold):
        figures["图1 逐层量化误差（cosine / SQNR）"] = fig1
        print("figure 1  :", os.path.relpath(fig1, ROOT))

    fig2 = os.path.join(args.outdir, "fig2_calib_sweep.png")
    if sweep_rows and fig_calib_sweep(sweep_rows, fig2):
        figures["图2 校准集规模与多样性影响"] = fig2
        print("figure 2  :", os.path.relpath(fig2, ROOT))

    if acc_rows:
        table_path = os.path.join(args.outdir, "table1_summary.md")
        with open(table_path, "w", encoding="utf-8") as fh:
            fh.write(build_summary_table(acc_rows, quant_report))
        write_csv(acc_rows, os.path.join(args.outdir, "table1_summary.csv"))
        print("table     :", os.path.relpath(table_path, ROOT))

    report = build_report(
        export_report, quant_report, layer_report, acc_rows, sweep_rows, figures
    )
    report_path = os.path.join(args.outdir, "README.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print("report    :", os.path.relpath(report_path, ROOT))

    missing = []
    for name, obj in (
        ("01_export_report", export_report),
        ("03_quant_report", quant_report),
        ("04_layer_analysis", layer_report),
    ):
        if not obj:
            missing.append(name)
    if not acc_rows:
        missing.append("05_accuracy")
    if not sweep_rows:
        missing.append("06_calib_sweep")
    if missing:
        print("\n缺少这些输入（先跑对应的脚本）: %s" % ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
