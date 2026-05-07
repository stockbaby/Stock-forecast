from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("docs/code_review_meeting_2026_05_07_previews")
W, H = 1600, 900


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


INK = "#172033"
MUTED = "#64748B"
BLUE = "#2563EB"
CYAN = "#0891B2"
GREEN = "#059669"
ORANGE = "#EA580C"
RED = "#DC2626"
LINE = "#D7DEE8"
PALE = "#F6F8FB"


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        current = ""
        for ch in para:
            candidate = current + ch
            if draw.textbbox((0, 0), candidate, font=fnt)[2] <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = ch
        if current:
            lines.append(current)
    return lines


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, color: str = INK, bold: bool = False, width: int | None = None, line_gap: int = 8) -> int:
    fnt = font(size, bold)
    x, y = xy
    lines = wrap(draw, text, fnt, width) if width else text.split("\n")
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=color)
        y += size + line_gap
    return y


def base(title: str, eyebrow: str = "CODE REVIEW") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((70, 35), eyebrow, font=font(14, True), fill=BLUE)
    d.text((70, 72), title, font=font(34, True), fill=INK)
    d.line((70, 138, 1530, 138), fill=LINE, width=2)
    d.text((1110, 845), "2026-05-07 · CSI300 Competition Review", font=font(14), fill="#94A3B8")
    return img, d


def metric(d: ImageDraw.ImageDraw, x: int, y: int, value: str, label: str, color: str) -> None:
    d.text((x, y), value, font=font(42, True), fill=color)
    d.text((x, y + 58), label, font=font(15), fill=MUTED)


def bullets(d: ImageDraw.ImageDraw, x: int, y: int, items: list[str], width: int = 720) -> None:
    yy = y
    for item in items:
        d.ellipse((x, yy + 8, x + 8, yy + 16), fill=BLUE)
        yy = draw_text(d, (x + 22, yy), item, 21, INK, width=width, line_gap=7) + 8


def table(d: ImageDraw.ImageDraw, x: int, y: int, rows: list[list[str]], colw: list[int], rowh: int = 42) -> None:
    yy = y
    for r, row in enumerate(rows):
        xx = x
        fill = "#E2E8F0" if r == 0 else "white"
        for c, cell in enumerate(row):
            d.rectangle((xx, yy, xx + colw[c], yy + rowh), outline=LINE, fill=fill)
            draw_text(d, (xx + 8, yy + 10), cell, 15 if r else 14, INK, bold=(r == 0), width=colw[c] - 16, line_gap=4)
            xx += colw[c]
        yy += rowh


def save(img: Image.Image, idx: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / f"slide_{idx:02d}.png")


slides = []

img = Image.new("RGB", (W, H), "#0B1220")
d = ImageDraw.Draw(img)
d.rectangle((0, 665, W, H), fill="#111827")
d.line((96, 120, 470, 120), fill=CYAN, width=6)
d.text((96, 158), "CSI300 Stock Forecast", font=font(26), fill="#93C5FD")
d.text((96, 235), "Code Review & Model Review", font=font(54, True), fill="white")
d.text((100, 342), "从数据形态、训练目标、组合策略到未来改进方向", font=font(24), fill="#CBD5E1")
metric(d, 100, 710, "0.106689", "A-stage replay after fixing latest inference", GREEN)
metric(d, 470, 710, "0.012834", "Original public score", ORANGE)
metric(d, 810, 710, "002493 · 1.0", "Current aggressive result", CYAN)
d.text((1180, 820), "PastoralBabyBoom · 2026-05-07", font=font(16), fill="#94A3B8")
save(img, 1)

img, d = base("这次复盘的核心结论", "EXECUTIVE SUMMARY")
draw_text(d, (90, 185), "我们不是在做全市场排序研究，而是在做 T 时点的收益最大化组合选择。", 31, INK, True, 1300)
bullets(d, 110, 320, [
    "最大改进来自修复预测日期错位：训练用有 label，推理必须用无 label 的 T 日截面。",
    "Top1 进攻版在 A-stage replay 命中，但单票满仓方差大，不能作为唯一长期策略。",
    "MASTER official 仍是当前最可靠的主模型；portfolio-return 直接 loss 初版过拟合。",
    "下一阶段围绕 latest inference、multi-seed、top2/confidence 兜底推进。",
], 840)
metric(d, 1080, 300, "0.106689", "best replay score", GREEN)
metric(d, 1080, 455, "6.52%", "Top1 validation std", RED)
save(img, 2)

img, d = base("数据长什么样", "DATA SHAPE")
table(d, 90, 180, [
    ["Dataset", "Rows", "Cols", "Coverage"],
    ["raw stock_data.csv", "723,269", "12", "300 stocks · 2,747 dates"],
    ["processed master", "723,269", "301", "2015-01-05 → 2026-04-24"],
    ["hs300 list", "300", "3", "component universe"],
], [300, 150, 140, 420], 54)
bullets(d, 105, 455, [
    "OHLCV、短/中窗口收益和波动率",
    "每日截面 rank / z-score",
    "行业、风格、beta、liquidity、correlation relation",
    "短期动量、放量突破、接近新高、行业集体动量",
], 720)
d.rounded_rectangle((1040, 190, 1460, 455), radius=18, outline="#A5F3FC", fill="#ECFEFF", width=2)
d.text((1080, 235), "关键 caveat", font=font(27, True), fill=CYAN)
draw_text(d, (1080, 305), "当前本地 data/raw/hs300_index.csv 缺失；市场指数特征需明确 fallback。", 20, INK, width=330)
save(img, 3)

img, d = base("从数据到提交：现在的闭环", "PIPELINE")
xs = [110, 395, 680, 965, 1250]
labels = ["Raw data", "Features", "Sequences", "Model scores", "Portfolio"]
for i, (x, lab) in enumerate(zip(xs, labels)):
    d.rounded_rectangle((x, 220, x + 215, 300), radius=14, fill="#E0F2FE" if i == 4 else PALE, outline=LINE)
    d.text((x + 25, 248), lab, font=font(18, True), fill=INK)
    if i < 4:
        d.line((x + 230, 260, x + 270, 260), fill=MUTED, width=2)
bullets(d, 115, 385, [
    "训练阶段只使用有 label 样本。",
    "推理阶段对 T 日无 label 截面单独构造序列。",
    "组合阶段显式比较 Top1 / Top2 / confidence。",
    "walk-forward 用于降低全验证集搜索过拟合。",
], 760)
d.rounded_rectangle((980, 385, 1460, 590), radius=16, outline="#FDBA74", fill="#FFF7ED", width=2)
d.text((1025, 430), "之前的关键 bug", font=font(25, True), fill=ORANGE)
draw_text(d, (1025, 490), "submission 取了 2026-04-20；真正线上需要 T=2026-04-24。", 20, INK, width=370)
save(img, 4)

img, d = base("模型方法：MASTER 是当前主线", "METHODS")
d.text((100, 185), "MASTER-style", font=font(26, True), fill=BLUE)
bullets(d, 110, 235, ["特征分组归一化", "stock + market projection", "Transformer encoder", "market gate + cross-attention", "T 日截面 score"], 560)
d.text((850, 185), "StockMixer", font=font(26, True), fill=GREEN)
bullets(d, 860, 235, ["轻量快速，可作弹性候选", "feature-group projector", "channel/token mixer", "recent weighted pooling", "当前 fast 版弱于 MASTER"], 520)
d.text((100, 640), "损失函数演进：MSE → pairwise rank → Top-K listwise → portfolio loss → walk-forward", font=font(24, True), fill=INK)
save(img, 5)

img, d = base("多方法验证：谁更稳，谁更冲", "RESULTS")
table(d, 85, 175, [
    ["Method", "Strategy", "Mean", "Std", "Latest A"],
    ["master_multiseed", "top2", "0.02450", "0.04912", "n/a"],
    ["master_official", "top2", "0.02145", "0.04913", "0.04964"],
    ["master_official", "top1", "0.02130", "0.06522", "0.10669"],
    ["master_official", "confidence", "0.01982", "0.05221", "0.06279"],
], [300, 230, 150, 150, 170], 54)
draw_text(d, (1030, 210), "历史均值不等于线上安全：Top1 这次赢，但风险最高。", 28, INK, True, 420)
draw_text(d, (1030, 360), "最佳当前解释：official MASTER 保留主线；multi-seed latest inference 是下一步。", 21, MUTED, width=420)
save(img, 6)

img, d = base("Top1 满仓：收益最大，但不是默认安全策略", "RISK REVIEW")
metric(d, 110, 190, "2.13%", "Top1 mean", GREEN)
metric(d, 360, 190, "6.52%", "Top1 std", RED)
metric(d, 610, 190, "37.36%", "negative days", ORANGE)
metric(d, 900, 190, "-15.97%", "worst day", RED)
bullets(d, 110, 410, ["冲榜版：top1_weight", "稳健默认：top2_softmax", "折中版：confidence_topk", "未来加入 regime-aware concentration gate"], 780)
table(d, 1010, 410, [["Policy", "Latest A", "Role"], ["Top1", "0.10669", "aggressive"], ["Confidence", "0.06279", "balanced"], ["Top2", "0.04964", "fallback"]], [150, 150, 170], 48)
save(img, 7)

img, d = base("训练目标对齐：方向正确，第一版实现失败", "OBJECTIVE REVIEW")
draw_text(d, (105, 180), "目标函数应该更像组合收益最大化，而不是只看全局 RankIC。", 30, INK, True, 1250)
bullets(d, 110, 300, ["可微 softmax 权重 × 真实未来收益", "理论上最贴近比赛目标", "当前实现对头部噪声非常敏感"], 650)
table(d, 840, 300, [["Method", "Strategy", "Mean", "Latest A"], ["official", "top2", "0.02145", "0.04964"], ["portfolio-loss", "top2", "0.00451", "-0.00955"], ["portfolio-loss", "top1", "0.00271", "-0.05155"]], [220, 160, 130, 150], 52)
draw_text(d, (110, 650), "结论：保留 portfolio-return 作为实验方向；下一版需要 capped labels、downside penalty、multi-seed 和 walk-forward loss-weight selection。", 23, BLUE, True, 1250)
save(img, 8)

img, d = base("局限性：现在最需要 code review 的地方", "LIMITATIONS")
items = [("Date alignment", "每个阶段必须断言 latest prediction date == configured T"), ("Feature availability", "hs300_index.csv 缺失，市场特征需明确 fallback"), ("Top1 risk", "满仓单票高波动，需生产级 concentration gate"), ("Validation overfit", "全验证集调参会乐观，必须 walk-forward"), ("Objective noise", "直接 portfolio loss 容易追噪声头部")]
for i, (a, b) in enumerate(items):
    y = 185 + i * 95
    d.rounded_rectangle((110, y, 390, y + 45), radius=12, fill=[RED, ORANGE, BLUE, CYAN, GREEN][i])
    d.text((130, y + 12), a, font=font(18, True), fill="white")
    d.text((430, y + 12), b, font=font(22), fill=INK)
save(img, 9)

img, d = base("代码审查地图：先看这些文件", "CODE REVIEW MAP")
table(d, 85, 175, [["Area", "Files", "Review question"], ["Data", "dataset_builder.py\ndeep_sequence.py", "T 日无 label 推理是否始终正确？"], ["Models", "master.py\nstockmixer.py", "目标函数和归一化是否与线上一致？"], ["Portfolio", "construct.py", "Top1 / Top2 / confidence 是否可控？"], ["Validation", "simulate_online_windows.py\nvalidate_multiple_methods.py", "是否足够防止参数过拟合？"]], [180, 520, 650], 70)
draw_text(d, (110, 760), "Meeting decision: 生产默认到底选 aggressive Top1，还是 top2/confidence fallback？", 24, BLUE, True, 1200)
save(img, 10)

img, d = base("未来推进方向：先稳住闭环，再提高上限", "ROADMAP")
d.text((110, 190), "Immediate", font=font(25, True), fill=BLUE)
bullets(d, 120, 245, ["phase T 显式配置并强制校验", "生产策略分 aggressive / fallback 两档", "完成 multi-seed MASTER latest inference"], 360)
d.text((590, 190), "Next experiments", font=font(25, True), fill=GREEN)
bullets(d, 600, 245, ["master_multiseed + top2/confidence", "downside-penalized surrogate", "regime-aware concentration gate"], 370)
d.text((1070, 190), "Longer term", font=font(25, True), fill=ORANGE)
bullets(d, 1080, 245, ["验证主题/动量特征", "relation 稳定后再考虑 GNN/HIST", "提交前自动 checklist"], 360)
d.line((120, 680, 1460, 680), fill=LINE, width=2)
draw_text(d, (130, 730), "North star: 每个新阶段：确认 T 日推理正确 → 多方法 walk-forward → 根据风险档位生成提交。", 27, INK, True, 1250)
save(img, 11)

print(f"wrote {len(list(OUT_DIR.glob('slide_*.png')))} previews to {OUT_DIR}")
