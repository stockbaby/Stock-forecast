const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const OUT = path.join("docs", "code_review_meeting_2026_05_07.pptx");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "PastoralBabyBoom / Codex";
pptx.subject = "CSI300 stock forecast code review";
pptx.title = "Code Review and Model Review";
pptx.company = "PastoralBabyBoom";
pptx.lang = "zh-CN";
pptx.theme = {
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  lang: "zh-CN",
};
pptx.defineLayout({ name: "WIDE", width: 13.333, height: 7.5 });
pptx.layout = "WIDE";

const C = {
  ink: "172033",
  muted: "64748B",
  pale: "F6F8FB",
  blue: "2563EB",
  cyan: "0891B2",
  green: "059669",
  orange: "EA580C",
  red: "DC2626",
  line: "D7DEE8",
  white: "FFFFFF",
  dark: "0B1220",
};

function addSlide(title, eyebrow) {
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  slide.addText(eyebrow || "CODE REVIEW", {
    x: 0.55,
    y: 0.28,
    w: 2.4,
    h: 0.22,
    fontFace: "Microsoft YaHei",
    fontSize: 8,
    bold: true,
    color: C.blue,
    charSpace: 1.2,
    margin: 0,
  });
  slide.addText(title, {
    x: 0.55,
    y: 0.58,
    w: 10.8,
    h: 0.42,
    fontFace: "Microsoft YaHei",
    fontSize: 24,
    bold: true,
    color: C.ink,
    margin: 0,
    breakLine: false,
    fit: "shrink",
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55,
    y: 1.15,
    w: 12.2,
    h: 0,
    line: { color: C.line, width: 1 },
  });
  slide.addText("2026-05-07 · CSI300 Competition Review", {
    x: 9.0,
    y: 7.05,
    w: 3.7,
    h: 0.22,
    fontSize: 7.5,
    color: "94A3B8",
    align: "right",
    margin: 0,
  });
  return slide;
}

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x,
    y,
    w,
    h,
    fontFace: "Microsoft YaHei",
    fontSize: opts.size || 12,
    bold: !!opts.bold,
    color: opts.color || C.ink,
    valign: opts.valign || "mid",
    margin: opts.margin ?? 0.04,
    fit: opts.fit || "shrink",
    breakLine: false,
  });
}

function metric(slide, label, value, x, y, w, color = C.blue, note = "") {
  addText(slide, value, x, y, w, 0.48, { size: 25, bold: true, color, margin: 0 });
  addText(slide, label, x, y + 0.56, w, 0.22, { size: 8.5, color: C.muted, margin: 0 });
  if (note) addText(slide, note, x, y + 0.82, w, 0.28, { size: 8, color: C.muted, margin: 0 });
}

function tag(slide, text, x, y, w, color) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h: 0.34,
    rectRadius: 0.08,
    fill: { color },
    line: { color },
  });
  addText(slide, text, x + 0.08, y + 0.06, w - 0.16, 0.18, {
    size: 8,
    color: C.white,
    bold: true,
    margin: 0,
  });
}

function lineFlow(slide, items, y) {
  const xs = [0.75, 3.2, 5.65, 8.1, 10.55];
  items.forEach((item, i) => {
    slide.addShape(pptx.ShapeType.roundRect, {
      x: xs[i],
      y,
      w: 1.85,
      h: 0.72,
      rectRadius: 0.06,
      fill: { color: i === items.length - 1 ? "E0F2FE" : C.pale },
      line: { color: i === items.length - 1 ? "7DD3FC" : C.line, width: 1 },
    });
    addText(slide, item, xs[i] + 0.12, y + 0.16, 1.6, 0.32, { size: 9.5, bold: true, margin: 0 });
    if (i < items.length - 1) {
      slide.addShape(pptx.ShapeType.line, {
        x: xs[i] + 1.95,
        y: y + 0.36,
        w: 0.95,
        h: 0,
        line: { color: C.muted, width: 1.2, beginArrowType: "none", endArrowType: "triangle" },
      });
    }
  });
}

function addTable(slide, rows, x, y, w, h, colW) {
  slide.addTable(rows, {
    x,
    y,
    w,
    h,
    colW,
    border: { type: "solid", color: C.line, pt: 0.5 },
    margin: 0.06,
    fontFace: "Microsoft YaHei",
    fontSize: 8.2,
    color: C.ink,
    valign: "mid",
    fit: "shrink",
    fill: { color: C.white },
  });
}

function bulletList(slide, bullets, x, y, w, h) {
  slide.addText(bullets.map(b => ({ text: b, options: { bullet: { type: "ul" } } })), {
    x,
    y,
    w,
    h,
    fontFace: "Microsoft YaHei",
    fontSize: 11,
    color: C.ink,
    breakLine: false,
    fit: "shrink",
    margin: 0.05,
    paraSpaceAfterPt: 6,
  });
}

function bar(slide, label, value, max, x, y, w, color) {
  addText(slide, label, x, y, 2.5, 0.22, { size: 8.5, color: C.ink, margin: 0 });
  slide.addShape(pptx.ShapeType.rect, { x: x + 2.65, y: y + 0.04, w, h: 0.11, fill: { color: "E5E7EB" }, line: { color: "E5E7EB" } });
  slide.addShape(pptx.ShapeType.rect, { x: x + 2.65, y: y + 0.04, w: Math.max(0.03, (value / max) * w), h: 0.11, fill: { color }, line: { color } });
  addText(slide, value.toFixed(3), x + 2.65 + w + 0.12, y - 0.01, 0.6, 0.18, { size: 7.5, color: C.muted, margin: 0 });
}

// 1 Cover
{
  const slide = pptx.addSlide();
  slide.background = { color: C.dark };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 5.55, w: 13.333, h: 1.95, fill: { color: "111827" }, line: { color: "111827" } });
  slide.addShape(pptx.ShapeType.line, { x: 0.8, y: 1.0, w: 3.1, h: 0, line: { color: C.cyan, width: 5 } });
  slide.addText("CSI300 Stock Forecast", { x: 0.78, y: 1.25, w: 10.5, h: 0.55, fontFace: "Microsoft YaHei", fontSize: 16, color: "93C5FD", margin: 0 });
  slide.addText("Code Review & Model Review", { x: 0.75, y: 1.95, w: 11.4, h: 0.95, fontFace: "Microsoft YaHei", fontSize: 34, bold: true, color: C.white, margin: 0, fit: "shrink" });
  slide.addText("从数据形态、训练目标、组合策略到未来改进方向", { x: 0.78, y: 3.05, w: 8.4, h: 0.36, fontFace: "Microsoft YaHei", fontSize: 14, color: "CBD5E1", margin: 0 });
  metric(slide, "A-stage replay after fixing latest inference", "0.106689", 0.82, 5.95, 2.35, C.green, "T=2026-04-24");
  metric(slide, "Original public score", "0.012834", 3.65, 5.95, 2.0, C.orange, "date-aligned replay gap");
  metric(slide, "Current production result", "002493 · 1.0", 6.25, 5.95, 2.6, C.cyan, "aggressive Top1");
  slide.addText("PastoralBabyBoom · 2026-05-07", { x: 9.1, y: 6.65, w: 3.3, h: 0.25, fontFace: "Microsoft YaHei", fontSize: 9, color: "94A3B8", align: "right", margin: 0 });
}

// 2 Review thesis
{
  const slide = addSlide("这次复盘的核心结论", "EXECUTIVE SUMMARY");
  slide.addText("我们不是在做全市场排序研究，而是在做 T 时点的收益最大化组合选择。", {
    x: 0.75, y: 1.55, w: 11.2, h: 0.7, fontFace: "Microsoft YaHei", fontSize: 23, bold: true, color: C.ink, margin: 0, fit: "shrink",
  });
  bulletList(slide, [
    "最大改进来自修复预测日期错位：训练用有 label，推理必须用无 label 的 T 日截面。",
    "Top1 进攻版在 A-stage replay 命中，但单票满仓方差大，不能作为唯一长期策略。",
    "MASTER official 仍是当前最可靠的主模型；portfolio-return 直接 loss 初版过拟合。",
    "下一阶段应围绕 latest inference、multi-seed、top2/confidence 兜底和稳健目标函数推进。",
  ], 0.82, 2.55, 7.0, 2.4);
  metric(slide, "best replay score", "0.106689", 8.55, 2.3, 2.0, C.green);
  metric(slide, "Top1 validation std", "6.52%", 10.45, 2.3, 1.7, C.red);
  metric(slide, "processed features", "301", 8.55, 4.3, 1.8, C.blue);
  metric(slide, "stocks", "300", 10.45, 4.3, 1.3, C.cyan);
}

// 3 Data
{
  const slide = addSlide("数据长什么样", "DATA SHAPE");
  addTable(slide, [
    [
      { text: "Dataset", options: { bold: true, fill: { color: "E2E8F0" } } },
      { text: "Rows", options: { bold: true, fill: { color: "E2E8F0" } } },
      { text: "Columns", options: { bold: true, fill: { color: "E2E8F0" } } },
      { text: "Coverage", options: { bold: true, fill: { color: "E2E8F0" } } },
    ],
    ["raw stock_data.csv", "723,269", "12", "300 stocks · 2,747 dates"],
    ["processed master dataset", "723,269", "301", "2015-01-05 → 2026-04-24"],
    ["hs300 stock list", "300", "3", "component universe"],
  ], 0.78, 1.55, 6.7, 1.65, [2.1, 1.1, 1.1, 2.4]);
  addText(slide, "核心字段", 0.85, 3.55, 1.3, 0.25, { size: 11, bold: true, color: C.blue, margin: 0 });
  bulletList(slide, [
    "OHLCV: 开盘、收盘、最高、最低、成交量、成交额",
    "派生收益: ret_1 / ret_3 / ret_5 / ret_20 等",
    "截面特征: 每日 rank、z-score、行业相对强弱",
    "主题特征: 短期动量、放量突破、接近新高、行业集体动量",
  ], 0.85, 3.95, 5.8, 1.9);
  slide.addShape(pptx.ShapeType.roundRect, { x: 8.25, y: 1.55, w: 3.7, h: 2.1, rectRadius: 0.08, fill: { color: "ECFEFF" }, line: { color: "A5F3FC" } });
  addText(slide, "关键 caveat", 8.55, 1.85, 1.9, 0.3, { size: 14, bold: true, color: C.cyan, margin: 0 });
  addText(slide, "当前本地 data/raw/hs300_index.csv 缺失；市场指数特征应被视为可选或来自既有 processed artifact。", 8.55, 2.35, 3.05, 0.8, { size: 11, color: C.ink, margin: 0.02 });
  tag(slide, "训练 label: T+1 open → T+5 open", 8.4, 4.25, 2.75, C.blue);
  tag(slide, "线上 T 日无 label，必须单独推理", 8.4, 4.8, 3.05, C.red);
}

// 4 Pipeline
{
  const slide = addSlide("从数据到提交：现在的闭环", "PIPELINE");
  lineFlow(slide, ["Raw data", "Features", "Sequences", "Model scores", "Portfolio"], 1.75);
  bulletList(slide, [
    "训练阶段只使用有 label 样本，保持监督目标干净。",
    "推理阶段对 T 日无 label 截面单独构造序列。",
    "组合阶段不再默认 Top5 分散，而是显式比较 Top1 / Top2 / confidence。",
    "walk-forward 模拟用于降低全验证集搜索带来的过拟合。",
  ], 0.85, 3.0, 6.2, 2.2);
  slide.addShape(pptx.ShapeType.roundRect, { x: 7.65, y: 3.0, w: 4.65, h: 1.85, rectRadius: 0.08, fill: { color: "FFF7ED" }, line: { color: "FDBA74" } });
  addText(slide, "之前的关键 bug", 7.95, 3.28, 2.1, 0.3, { size: 13, bold: true, color: C.orange, margin: 0 });
  addText(slide, "submission 取了 valid_pred_df.max(date)=2026-04-20；真正线上需要 T=2026-04-24 无 label 截面。", 7.95, 3.8, 3.95, 0.62, { size: 10.5, color: C.ink, margin: 0.02 });
  metric(slide, "after fixing", "latest_inference", 8.0, 5.25, 2.6, C.green, "metrics now record inference_date");
}

// 5 Model principles
{
  const slide = addSlide("模型方法：MASTER 是当前主线", "METHODS");
  addText(slide, "MASTER-style", 0.8, 1.55, 2.0, 0.3, { size: 15, bold: true, color: C.blue, margin: 0 });
  bulletList(slide, [
    "特征分组归一化：price / volume / volatility / market / style",
    "stock projection + market projection",
    "Transformer encoder 建模时间序列",
    "market gate 与 cross-attention 融合市场状态",
    "输出每个股票在 T 日截面的 score",
  ], 0.85, 1.95, 5.3, 2.35);
  addText(slide, "StockMixer", 7.0, 1.55, 2.0, 0.3, { size: 15, bold: true, color: C.green, margin: 0 });
  bulletList(slide, [
    "更轻量，训练快，可作弹性候选",
    "feature-group projector + channel/token mixer",
    "recent weighted pooling 和多 patch 窗口",
    "当前 fast 版验证不如 MASTER，但适合候选切换",
  ], 7.05, 1.95, 5.1, 2.0);
  addText(slide, "损失函数演进", 0.85, 4.7, 1.6, 0.25, { size: 12, bold: true, color: C.ink, margin: 0 });
  lineFlow(slide, ["MSE", "Pairwise rank", "Top-K listwise", "Portfolio loss", "Walk-forward"], 5.35);
}

// 6 Results
{
  const slide = addSlide("多方法验证：谁更稳，谁更冲", "RESULTS");
  addTable(slide, [
    [{ text: "Method", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Strategy", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Mean", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Std", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Latest A", options: { bold: true, fill: { color: "E2E8F0" } } }],
    ["master_multiseed", "top2_softmax", "0.02450", "0.04912", "n/a"],
    ["master_official", "top2_softmax", "0.02145", "0.04913", "0.04964"],
    ["master_official", "top1_weight", "0.02130", "0.06522", "0.10669"],
    ["master_official", "confidence", "0.01982", "0.05221", "0.06279"],
    ["stockmixer_fast", "top1_weight", "0.01626", "0.04426", "0.05014"],
  ], 0.75, 1.45, 7.2, 2.55, [1.8, 1.65, 1.05, 1.05, 1.15]);
  bar(slide, "master_multiseed top2", 0.0245, 0.026, 8.3, 1.65, 2.3, C.green);
  bar(slide, "master_official top2", 0.02145, 0.026, 8.3, 2.1, 2.3, C.blue);
  bar(slide, "master_official top1", 0.02130, 0.026, 8.3, 2.55, 2.3, C.orange);
  bar(slide, "confidence topk", 0.01982, 0.026, 8.3, 3.0, 2.3, C.cyan);
  addText(slide, "历史均值不等于线上安全：Top1 这次赢，但风险最高。", 8.3, 4.05, 3.8, 0.5, { size: 14, bold: true, color: C.ink, margin: 0 });
  addText(slide, "最佳当前解释：official MASTER 保留为生产主线；multi-seed + latest inference 是下一步最值得补齐的方向。", 8.3, 4.72, 3.8, 0.7, { size: 10.5, color: C.muted, margin: 0.02 });
}

// 7 Risk
{
  const slide = addSlide("Top1 满仓：收益最大，但不是默认安全策略", "RISK REVIEW");
  metric(slide, "Top1 mean", "2.13%", 0.9, 1.55, 1.5, C.green);
  metric(slide, "Top1 std", "6.52%", 2.75, 1.55, 1.5, C.red);
  metric(slide, "negative days", "37.36%", 4.6, 1.55, 1.7, C.orange);
  metric(slide, "worst day", "-15.97%", 6.65, 1.55, 1.7, C.red);
  addText(slide, "兜底策略建议", 0.9, 3.2, 1.7, 0.28, { size: 14, bold: true, color: C.blue, margin: 0 });
  bulletList(slide, [
    "冲榜版：top1_weight，只在强信号或明确冒险时使用。",
    "稳健默认：top2_softmax，历史均值接近 Top1、方差低很多。",
    "折中版：confidence_topk，根据分数边际和不确定性动态降仓。",
    "未来应加入 regime-aware concentration gate。",
  ], 0.95, 3.65, 6.3, 2.1);
  addTable(slide, [
    [{ text: "Policy", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Latest A", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Role", options: { bold: true, fill: { color: "E2E8F0" } } }],
    ["Top1", "0.10669", "aggressive"],
    ["Confidence", "0.06279", "balanced"],
    ["Top2", "0.04964", "fallback"],
  ], 8.0, 3.55, 3.8, 1.4, [1.25, 1.0, 1.3]);
}

// 8 Portfolio objective
{
  const slide = addSlide("训练目标对齐：方向正确，第一版实现失败", "OBJECTIVE REVIEW");
  slide.addText("目标函数应该更像组合收益最大化，而不是只看全局 RankIC。", {
    x: 0.85, y: 1.48, w: 9.5, h: 0.45, fontFace: "Microsoft YaHei", fontSize: 20, bold: true, color: C.ink, margin: 0,
  });
  addText(slide, "Direct portfolio-return loss", 0.9, 2.35, 2.7, 0.28, { size: 12, bold: true, color: C.orange, margin: 0 });
  bulletList(slide, [
    "可微 softmax 权重 × 真实未来收益",
    "理论上最贴近比赛目标",
    "但当前实现对头部噪声非常敏感",
  ], 0.95, 2.75, 4.8, 1.45);
  addTable(slide, [
    [{ text: "Method", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Strategy", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Mean", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Latest A", options: { bold: true, fill: { color: "E2E8F0" } } }],
    ["official", "top2", "0.02145", "0.04964"],
    ["portfolio-loss", "top2", "0.00451", "-0.00955"],
    ["portfolio-loss", "top1", "0.00271", "-0.05155"],
  ], 6.55, 2.45, 5.2, 1.5, [1.65, 1.25, 1.0, 1.0]);
  addText(slide, "结论", 0.95, 5.15, 0.7, 0.22, { size: 12, bold: true, color: C.blue, margin: 0 });
  addText(slide, "保留 portfolio-return 作为实验方向，但下一版需要 capped labels、downside penalty、multi-seed 和 walk-forward loss-weight selection。", 1.75, 5.07, 9.7, 0.45, { size: 12, color: C.ink, margin: 0 });
}

// 9 Limitations
{
  const slide = addSlide("局限性：现在最需要 code review 的地方", "LIMITATIONS");
  const items = [
    ["Date alignment", "每个阶段必须断言 latest prediction date == configured T"],
    ["Feature availability", "hs300_index.csv 当前缺失，市场特征需明确 fallback"],
    ["Top1 risk", "满仓单票高波动，需生产级 concentration gate"],
    ["Validation overfit", "全验证集调参会乐观，必须 walk-forward"],
    ["Objective noise", "直接 portfolio loss 容易追噪声头部"],
  ];
  items.forEach((it, i) => {
    const y = 1.45 + i * 0.9;
    tag(slide, it[0], 0.85, y, 2.2, [C.red, C.orange, C.blue, C.cyan, C.green][i]);
    addText(slide, it[1], 3.35, y + 0.06, 7.8, 0.24, { size: 11, color: C.ink, margin: 0 });
  });
  addText(slide, "Review stance", 0.95, 6.08, 1.25, 0.2, { size: 9.5, bold: true, color: C.muted, margin: 0 });
  addText(slide, "先保护线上正确性，再追模型复杂度。", 2.25, 6.05, 3.8, 0.25, { size: 10.5, color: C.ink, margin: 0 });
}

// 10 Code map
{
  const slide = addSlide("代码审查地图：先看这些文件", "CODE REVIEW MAP");
  addTable(slide, [
    [{ text: "Area", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Files", options: { bold: true, fill: { color: "E2E8F0" } } }, { text: "Review question", options: { bold: true, fill: { color: "E2E8F0" } } }],
    ["Data", "src/training/dataset_builder.py\nsrc/models/deep_sequence.py", "T 日无 label 推理是否始终正确？"],
    ["Models", "src/models/master.py\nsrc/models/stockmixer.py", "目标函数和归一化是否与线上一致？"],
    ["Portfolio", "src/portfolio/construct.py", "Top1 / Top2 / confidence 策略是否可控？"],
    ["Validation", "scripts/simulate_online_windows.py\nscripts/validate_multiple_methods.py", "是否足够防止参数过拟合？"],
  ], 0.75, 1.45, 11.7, 3.9, [1.45, 4.2, 5.65]);
  addText(slide, "Meeting decision", 0.85, 5.85, 1.6, 0.22, { size: 10.5, bold: true, color: C.blue, margin: 0 });
  addText(slide, "生产默认到底选 aggressive Top1，还是 top2/confidence fallback？需要明确策略档位。", 2.55, 5.82, 7.3, 0.28, { size: 11, color: C.ink, margin: 0 });
}

// 11 Roadmap
{
  const slide = addSlide("未来推进方向：先稳住闭环，再提高上限", "ROADMAP");
  addText(slide, "Immediate", 0.85, 1.55, 1.4, 0.25, { size: 13, bold: true, color: C.blue, margin: 0 });
  bulletList(slide, [
    "把 phase T 作为显式配置并强制校验",
    "生产策略分 aggressive / fallback 两档",
    "完成 multi-seed MASTER latest inference",
  ], 0.95, 1.95, 3.35, 1.5);
  addText(slide, "Next experiments", 4.95, 1.55, 1.8, 0.25, { size: 13, bold: true, color: C.green, margin: 0 });
  bulletList(slide, [
    "master_multiseed + top2/confidence",
    "downside-penalized portfolio surrogate",
    "regime-aware concentration gate",
  ], 5.05, 1.95, 3.45, 1.5);
  addText(slide, "Longer term", 9.0, 1.55, 1.5, 0.25, { size: 13, bold: true, color: C.orange, margin: 0 });
  bulletList(slide, [
    "验证主题/动量特征在新阶段是否稳定",
    "relation gains 经 walk-forward 稳定后再考虑 GNN/HIST",
    "形成提交前自动 replay/validation checklist",
  ], 9.1, 1.95, 3.4, 1.7);
  slide.addShape(pptx.ShapeType.line, { x: 1.0, y: 5.45, w: 10.9, h: 0, line: { color: C.line, width: 1.5 } });
  addText(slide, "North star", 0.95, 5.9, 1.3, 0.25, { size: 10.5, bold: true, color: C.muted, margin: 0 });
  addText(slide, "每个新阶段：确认 T 日推理正确 → 多方法 walk-forward → 根据风险档位生成提交。", 2.3, 5.83, 8.6, 0.35, { size: 15, bold: true, color: C.ink, margin: 0 });
}

fs.mkdirSync(path.dirname(OUT), { recursive: true });
pptx.writeFile({ fileName: OUT });
