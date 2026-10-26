// Builds PCAP_NBody_Adaptive_Scheduling.pptx from data.json (all numbers come from the
// project's result CSVs via extract.py).  Run:  node build.js
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const SKILL = process.env.PPTX_SKILL_DIR;
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const IMG = (f) => path.join(__dirname, "img", f);
const OUTFILE = path.join(__dirname, "PCAP_NBody_Adaptive_Scheduling.pptx");

// ------------------------------------------------------------------ palette & type
const INK = "111111", PAPER = "F5F3EE", MUTED = "6B6862", RULE = "D9D5CC", ACC = "D9472B";
const GREY = "8E8A82", LIGHT = "B9B5AC", DARKG = "3A3936", ONDARK = "B9B5AC";
const SANS = "IBM Plex Sans", MONO = "IBM Plex Mono";
const THEME = {
  name: "PCAP Editorial",
  headFontFace: SANS, bodyFontFace: SANS,
  colors: { dk1: INK, lt1: PAPER, dk2: MUTED, lt2: "E4E1D9", accent1: ACC, accent2: GREY, accent3: DARKG,
            accent4: LIGHT, accent5: MUTED, accent6: INK, hlink: ACC, folHlink: MUTED },
};
const L = 0.75, W = 11.833, R = L + W, TOP = 2.75;   // grid

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";                          // 13.333 x 7.5 in
pres.theme = { headFontFace: SANS, bodyFontFace: SANS };
pres.title = "Adaptive Scheduling for Load-Balanced Parallel N-Body Simulation";
pres.author = "Tanish Hemant Kuvar, M. Vishal Kumar, Samyak Jain";
pres.subject = "PCAP course project";
const C = pres.SchemeColor;

// ------------------------------------------------------------------ layouts
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: PAPER },
  objects: [
    { placeholder: { options: { name: "label", type: "body", x: L, y: 0.5, w: 8, h: 0.3, margin: 0, fontFace: MONO,
                                fontSize: 10, color: MUTED, charSpacing: 1.5, valign: "top", align: "left" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: L, y: 0.86, w: 11.4, h: 1.1, margin: 0, fontFace: SANS,
                                fontSize: 30, bold: true, color: INK, valign: "top", align: "left", lineSpacingMultiple: 0.95 }, text: "" } },
    { placeholder: { options: { name: "takeaway", type: "body", x: L, y: 2.02, w: 11.0, h: 0.5, margin: 0, fontFace: SANS,
                                fontSize: 15, color: MUTED, valign: "top", align: "left" }, text: "" } },
    { text: { text: "Adaptive scheduling for parallel N-body simulation  ·  PCAP [COURSE CODE]",
              options: { x: L, y: 6.98, w: 9, h: 0.25, margin: 0, fontFace: MONO, fontSize: 9, color: MUTED } } },
  ],
  slideNumber: { x: R - 0.6, y: 6.98, w: 0.6, h: 0.25, fontFace: MONO, fontSize: 9, color: MUTED, align: "right" },
});
pres.defineSlideMaster({ title: "DARK", background: { color: INK }, objects: [] });

// ------------------------------------------------------------------ helpers
function contentSlide(section, num, label, title, takeaway) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText([{ text: num, options: { color: ACC } }, { text: "  —  " + label }], { placeholder: "label" });
  s.addText(title, { placeholder: "title" });
  s.addText(takeaway, { placeholder: "takeaway" });
  return s;
}
function rule(s, x, y, w, color = RULE, width = 0.75) {
  s.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width } });
}
function text(s, t, o) {
  s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontFace: SANS, fontSize: 14, color: INK, valign: "top" }, o));
}
function mono(s, t, o) { text(s, t, Object.assign({ fontFace: MONO, fontSize: 11, color: MUTED }, o)); }
function box(s, t, o) {   // hairline rectangle with centred mono text
  s.addText(t, Object.assign({ shape: pres.shapes.RECTANGLE, fill: { color: PAPER }, line: { color: INK, width: 0.75 },
    margin: 4, fontFace: MONO, fontSize: 11, color: INK, align: "center", valign: "middle" }, o));
}
function arrow(s, x1, y1, x2, y2, color = INK) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, line: { color, width: 1, endArrowType: "triangle" } });
}
const pct = (v, d = 1) => (100 * v).toFixed(d);
const SUP = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻" };
const sci = (v) => { const [m, e] = v.toExponential(1).split("e"); return m + " × 10" + String(parseInt(e, 10)).split("").map((c) => SUP[c]).join(""); };
const f2 = (v) => v.toFixed(2);
const CHART_BASE = () => ({
  catAxisLabelFontFace: MONO, valAxisLabelFontFace: MONO, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLineShow: true, catAxisLineColor: LIGHT,
  valAxisLineShow: false, valGridLine: { color: "E4E1D9", size: 0.5 }, catGridLine: { style: "none" },
  legendFontFace: MONO, legendFontSize: 10, legendColor: DARKG,
  catAxisTitleFontFace: MONO, valAxisTitleFontFace: MONO, catAxisTitleFontSize: 10, valAxisTitleFontSize: 10,
  catAxisTitleColor: MUTED, valAxisTitleColor: MUTED,
  plotArea: { fill: { color: PAPER } }, chartArea: { fill: { color: PAPER } },
});

// numbers used in several places
const sn = D.snap, tr = D.trace, gainMin = D.adaptive_gain_clustered.min, gainMax = D.adaptive_gain_clustered.max;
const gpu65 = D.gpu_direct.find((g) => g.n === 65536);
const gpuX = gpu65.cpu_ms / gpu65.gpu32_ms;
const spd8 = []; for (const ic in D.speedup65536) for (const s in D.speedup65536[ic]) spd8.push(D.speedup65536[ic][s][3]);
const spdMin = Math.min(...spd8), spdMax = Math.max(...spd8);

// ================================================================== COVER
pres.addSection({ title: "Opening" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
  s.addImage({ path: IMG("stars_static_dark.png"), x: 5.833, y: 0, w: 7.5, h: 7.5,
               altText: "Simulated star cloud after collapse; bodies computed by the busiest thread shown in vermilion" });
  mono(s, "PCAP  ·  COURSE PROJECT", { x: L, y: 0.75, w: 4.9, h: 0.3, color: ONDARK, fontSize: 10, charSpacing: 1.5 });
  text(s, "Adaptive Scheduling for Load-Balanced Parallel N-Body Simulation",
       { x: L, y: 1.2, w: 4.9, h: 2.2, fontSize: 32, bold: true, color: PAPER, lineSpacingMultiple: 0.95 });
  text(s, "A hybrid MPI–OpenMP–CUDA Barnes–Hut simulator that chooses its OpenMP loop schedule at run time from measured thread imbalance",
       { x: L, y: 3.5, w: 4.7, h: 1.0, fontSize: 14, color: ONDARK });
  rule(s, L, 4.75, 4.7, "3A3936");
  mono(s, [
    { text: "Tanish Hemant Kuvar", options: { color: PAPER, breakLine: false } }, { text: "   240962540", options: { breakLine: true } },
    { text: "M. Vishal Kumar", options: { color: PAPER } }, { text: "       240962546", options: { breakLine: true } },
    { text: "Samyak Jain", options: { color: PAPER } }, { text: "           240962638", options: { breakLine: true } },
    { text: "AIML-A" },
  ], { x: L, y: 4.9, w: 4.7, h: 1.05, fontSize: 11, color: ONDARK, lineSpacingMultiple: 1.15 });
  mono(s, [
    { text: "Guide        [GUIDE / FACULTY NAME]", options: { breakLine: true } },
    { text: "Institution  [INSTITUTION]", options: { breakLine: true } },
    { text: "Parallel Computing Architecture and Programming  ·  [COURSE CODE]", options: { breakLine: true } },
    { text: "27 October 2026" },
  ], { x: L, y: 6.0, w: 5.0, h: 1.05, fontSize: 9, color: ONDARK, lineSpacingMultiple: 1.15 });
  mono(s, `Cold collapse, N = ${sn.n.toLocaleString("en-US")}, t = ${sn.t.toFixed(2)}  ·  vermilion: bodies of the busiest thread`,
       { x: 7.1, y: 7.08, w: 5.48, h: 0.25, fontSize: 8, color: "8E8A82", align: "right" });
  s.addNotes("Introduce the team and the one-line idea: a gravity simulation parallelised at three levels, plus a controller that "
    + "changes how OpenMP shares work when the work becomes uneven.\nThe image is our own simulation after a cloud collapsed; vermilion "
    + "stars are the ones computed by the single busiest thread.");
}

// ================================================================== 01 PROBLEM
pres.addSection({ title: "Problem" });
{
  const s = contentSlide("Problem", "01", "PROBLEM", "Clustering turns equal shares of bodies into unequal shares of work",
    "With a static schedule, the thread that owns the dense core sets the pace for all eight.");
  s.addImage({ path: IMG("stars_static.png"), x: 0.75, y: TOP, w: 3.7, h: 3.7,
               altText: "Collapsed star cloud; the busiest thread's bodies form one block of the core" });
  mono(s, `Cold collapse, N = ${sn.n.toLocaleString("en-US")}, 8 threads, static, t = ${sn.t.toFixed(2)}. Vermilion: bodies computed by thread T${sn.busiest}.`,
       { x: 0.75, y: 6.5, w: 3.95, h: 0.38, fontSize: 8 });

  const rows = [
    ["COST", "Every step needs the force on every body: N(N−1)/2 ≈ 2.1 × 10⁹ pairs at N = 65,536."],
    ["SHORTCUT", "Barnes–Hut octree: a distant cell acts as one mass (θ = 0.5), giving O(N log N)."],
    ["IMBALANCE", "A body in a dense region opens more cells, so its force costs more."],
    ["SCHEDULE", "Static gives each thread an equal count of Morton-ordered bodies, i.e. one region of space."],
  ];
  rows.forEach(([k, v], i) => {
    const y = TOP + i * 0.98;
    rule(s, 5.15, y, 3.9);
    mono(s, k, { x: 5.15, y: y + 0.1, w: 3.9, h: 0.22, fontSize: 9, color: ACC, charSpacing: 1 });
    text(s, v, { x: 5.15, y: y + 0.36, w: 3.9, h: 0.6, fontSize: 13 });
  });

  // per-thread loop time at the same frame
  const labels = sn.static_wt_ms.map((_, i) => "T" + i);
  const hot = sn.static_wt_ms.map((v, i) => (i === sn.busiest ? v : 0));
  const cold = sn.static_wt_ms.map((v, i) => (i === sn.busiest ? 0 : v));
  mono(s, "FORCE-LOOP TIME PER THREAD [ms]", { x: 9.55, y: TOP, w: 3.03, h: 0.22, fontSize: 9, color: MUTED, charSpacing: 1 });
  s.addChart(pres.charts.BAR, [{ name: "busiest thread", labels, values: hot }, { name: "other threads", labels, values: cold }],
    Object.assign(CHART_BASE(), {
      x: 9.45, y: TOP + 0.25, w: 3.13, h: 2.45, barDir: "bar", barGrouping: "stacked", chartColors: [ACC, LIGHT],
      catAxisOrientation: "maxMin", valAxisMinVal: 0, valAxisMaxVal: 50, valAxisMajorUnit: 25, showLegend: false,
      barGapWidthPct: 45, valAxisLabelFormatCode: "0",
    }));
  text(s, (sn.ratio_max_min).toFixed(1) + "×", { x: 9.55, y: 5.55, w: 3.0, h: 0.75, fontFace: MONO, fontSize: 40, bold: true, color: ACC });
  mono(s, `busiest vs least-loaded thread; λ = ${f2(sn.static_lam)} at this step`, { x: 9.55, y: 6.32, w: 3.03, h: 0.4, fontSize: 9 });
  s.addNotes("An N-body step needs the gravitational force on every body; done exactly that is N squared pairs, which is why we use the "
    + "Barnes-Hut tree. Its cost per body depends on how dense the neighbourhood is.\nBecause we sort bodies along a Morton curve, a static "
    + "schedule hands each thread one region of space. In this frame thread T2 owns part of the dense core and takes 3.7 times as long as the "
    + "lightest thread, so seven threads wait.\nDemo option: open results/presentation.html (section 2) and play the animation here.");
}

// ================================================================== 02 OBJECTIVES
pres.addSection({ title: "Objectives" });
{
  const s = contentSlide("Objectives", "02", "OBJECTIVES", "All four synopsis objectives are implemented and measured",
    "Each objective maps to code in the repository and to a measured result.");
  const cols = [{ x: L, w: 0.7 }, { x: 1.55, w: 4.55 }, { x: 6.3, w: 3.0 }, { x: 9.5, w: 3.08 }];
  const head = ["", "OBJECTIVE", "IMPLEMENTED AS", "EVIDENCE"];
  head.forEach((h, i) => mono(s, h, { x: cols[i].x, y: TOP, w: cols[i].w, h: 0.22, fontSize: 9, charSpacing: 1 }));
  const rows = [
    ["O1", "Sequential direct-sum and Barnes–Hut baselines with energy and momentum checks",
     "--force direct | bh, 1 thread", "max ΔE/E 1.1 × 10⁻⁷ (direct)"],
    ["O2", "OpenMP force loop; characterise static, dynamic and guided schedules",
     "schedule(runtime)", "4 inputs × 3 sizes × 1–8 threads"],
    ["O3", "MPI distribution across ranks and a CUDA force kernel",
     "MPI_Allgatherv; tiled kernel via NVRTC", "1–4 ranks bit-identical; GPU fp32 error 4.6 × 10⁻⁶"],
    ["O4", "Adaptive controller, evaluated on uniform and clustered inputs",
     "omp_set_schedule, k = 5", "faster than static in 9/9 clustered cases"],
    ["+", "Beyond the synopsis: corrected imbalance metric under dynamic, thread pinning, thermal gate",
     "slides 04 and 05", "8/8 correctness checks"],
  ];
  rows.forEach((r, i) => {
    const y = TOP + 0.32 + i * 0.74;
    rule(s, L, y, W);
    if (r[0] === "+") text(s, "+", { x: cols[0].x, y: y + 0.11, w: cols[0].w, h: 0.5, fontFace: SANS, fontSize: 17, bold: true, color: ACC });
    else text(s, [{ text: "OBJ", options: { fontFace: MONO, fontSize: 8, bold: false, charSpacing: 1, breakLine: true } },
                  { text: r[0].slice(1), options: { fontFace: MONO, fontSize: 17, bold: true } }],
              { x: cols[0].x, y: y + 0.08, w: cols[0].w, h: 0.6, color: ACC, lineSpacingMultiple: 0.9 });
    text(s, r[1], { x: cols[1].x, y: y + 0.12, w: cols[1].w, h: 0.58, fontSize: 13 });
    mono(s, r[2], { x: cols[2].x, y: y + 0.14, w: cols[2].w, h: 0.55, fontSize: 10.5, color: DARKG });
    mono(s, r[3], { x: cols[3].x, y: y + 0.14, w: cols[3].w, h: 0.55, fontSize: 10.5, color: INK });
  });
  rule(s, L, TOP + 0.32 + 5 * 0.74, W);
  s.addNotes("These are the four objectives from our synopsis. O1 gave correct sequential baselines, O2 the OpenMP loop and the three "
    + "built-in schedules, O3 the MPI decomposition and the CUDA kernels, O4 the adaptive controller.\nThe last row is what we added "
    + "because the measurements required it; slide 04 explains the controller correction.");
}

// ================================================================== 03 ARCHITECTURE
pres.addSection({ title: "Design" });
{
  const s = contentSlide("Design", "03", "ARCHITECTURE", "Every rank replicates the tree; threads and the GPU share the force work",
    "One timestep crosses three levels of parallelism. Bodies are re-sorted along a Morton curve every step.");
  const steps = ["kick + drift", "MPI_Allgatherv", "Morton sort", "octree build", "force loop", "kick", "controller"];
  const bw = 1.45, gap = (W - 7 * bw) / 6;
  steps.forEach((t, i) => {
    const x = L + i * (bw + gap), hotStep = (t === "force loop" || t === "controller");
    box(s, t, { x, y: TOP, w: bw, h: 0.48, line: { color: hotStep ? ACC : INK, width: hotStep ? 1.25 : 0.75 },
                color: hotStep ? ACC : INK, fontSize: 10.5 });
    if (i < 6) arrow(s, x + bw + 0.04, TOP + 0.24, x + bw + gap - 0.04, TOP + 0.24, MUTED);
  });
  const bands = [
    { y: 3.55, label: "MPI\nACROSS RANKS" }, { y: 4.65, label: "OPENMP\nACROSS CORES" }, { y: 5.75, label: "CUDA\nON THE GPU" }];
  bands.forEach((b) => {
    rule(s, L, b.y, W);
    mono(s, b.label, { x: L, y: b.y + 0.14, w: 2.0, h: 0.5, fontSize: 9.5, color: INK, charSpacing: 1 });
  });
  const X0 = 3.0;
  for (let r = 0; r < 4; r++) box(s, "rank " + r, { x: X0 + r * 1.3, y: 3.72, w: 1.15, h: 0.42, fontSize: 10 });
  mono(s, "each rank owns one contiguous Morton block;\npositions and velocities exchanged every step",
       { x: 8.35, y: 3.7, w: 4.23, h: 0.6, fontSize: 9.5, color: MUTED });
  for (let t = 0; t < 8; t++) box(s, "T" + t, { x: X0 + t * 0.6, y: 4.83, w: 0.5, h: 0.42, fontSize: 10 });
  mono(s, "#pragma omp for schedule(runtime)", { x: X0, y: 5.33, w: 4.8, h: 0.25, fontSize: 9.5, color: MUTED });
  box(s, "controller → omp_set_schedule()", { x: 8.35, y: 4.83, w: 4.23, h: 0.42, line: { color: ACC, width: 1.25 },
       color: ACC, fontSize: 10.5 });
  box(s, "tiled direct kernel  ·  256-body shared-memory tiles", { x: X0, y: 5.93, w: 4.65, h: 0.42, fontSize: 10 });
  box(s, "near-field kernel  ·  leaf ranges from the CPU walk", { x: 7.93, y: 5.93, w: 4.65, h: 0.42, fontSize: 10 });
  mono(s, "compiled at run time with NVRTC for the detected GPU (no nvcc / MSVC needed)",
       { x: X0, y: 6.45, w: 9.58, h: 0.25, fontSize: 9.5, color: MUTED });
  s.addNotes("Read the top row left to right: that is one timestep. Every rank holds the full state, sorts it and builds the same tree, "
    + "then computes forces only for its own block.\nInside a rank the force loop is an OpenMP loop with schedule(runtime), so the "
    + "controller can change the schedule between steps without recompiling.\nThe GPU either computes all pairs with a tiled kernel, or "
    + "takes the near-field pairs that the CPU tree walk collected.\nLive option (needs internet): results/web/viewer.html shows the "
    + "thread ownership in 3D.");
}

// ================================================================== 04 CONTROLLER
{
  const s = contentSlide("Design", "04", "ADAPTIVE CONTROLLER", "The controller switches schedule when measured imbalance crosses a threshold",
    "λ = max(tᵢ) / mean(tᵢ) − 1 over a k = 5 step window, with hysteresis between 0.05 and 0.15.");
  const steps = [
    ["01", "time each thread's share of the force loop"],
    ["02", "accumulate over a window of k = 5 steps"],
    ["03", "compute λ for the window"],
    ["04", "compare with 0.15 (switch up) and 0.05 (switch back)"],
    ["05", "omp_set_schedule(policy, chunk = 8)"],
  ];
  steps.forEach(([n, t], i) => {
    const y = TOP + i * 0.78;
    rule(s, L, y, 4.4);
    text(s, n, { x: L, y: y + 0.17, w: 0.55, h: 0.4, fontFace: MONO, fontSize: 13, bold: true, color: ACC });
    text(s, t, { x: L + 0.6, y: y + 0.17, w: 3.8, h: 0.55, fontSize: 13 });
  });
  rule(s, L, TOP + 5 * 0.78, 4.4);
  // state diagram
  const sx = 5.9, dx = 10.18, sy = TOP + 0.15, bw = 2.4, bh = 1.0;
  box(s, [{ text: "STATIC", options: { bold: true, fontSize: 14, breakLine: true } },
          { text: "equal contiguous blocks", options: { fontSize: 9.5, color: MUTED } }], { x: sx, y: sy, w: bw, h: bh });
  box(s, [{ text: "DYNAMIC", options: { bold: true, fontSize: 14, breakLine: true, color: ACC } },
          { text: "chunks of 8 on demand", options: { fontSize: 9.5, color: MUTED } }],
      { x: dx, y: sy, w: bw, h: bh, line: { color: ACC, width: 1.25 } });
  arrow(s, sx + bw + 0.1, sy + 0.3, dx - 0.1, sy + 0.3, INK);
  arrow(s, dx - 0.1, sy + 0.72, sx + bw + 0.1, sy + 0.72, INK);
  mono(s, "λ > 0.15", { x: sx + bw, y: sy - 0.02, w: dx - sx - bw, h: 0.25, align: "center", color: INK, fontSize: 11 });
  mono(s, "predicted λ < 0.05", { x: sx + bw, y: sy + 0.8, w: dx - sx - bw, h: 0.25, align: "center", color: INK, fontSize: 11 });

  rule(s, 5.9, 4.3, 6.68);
  mono(s, "CORRECTION TO ALGORITHM 1", { x: 5.9, y: 4.42, w: 6.68, h: 0.22, fontSize: 9, color: ACC, charSpacing: 1 });
  text(s, "Under dynamic the measured times are balanced by construction, so λ would always fall below 0.05 and the policy would oscillate. The controller instead predicts the times static would give:",
       { x: 5.9, y: 4.72, w: 6.68, h: 0.85, fontSize: 13 });
  const sub = (t) => ({ text: t, options: { subscript: true } });
  text(s, [{ text: "T" }, sub("i"), { text: " = B" }, sub("i"), { text: " / v" }, sub("i"), { text: "        v" }, sub("i"),
           { text: " = W" }, sub("i"), { text: " / t" }, sub("i")],
       { x: 5.9, y: 5.6, w: 6.68, h: 0.45, fontFace: MONO, fontSize: 18, color: INK });
  mono(s, "Tᵢ predicted static time  ·  Bᵢ work in thread i's static block  ·  Wᵢ work it executed. Under static, Tᵢ = tᵢ.",
       { x: 5.9, y: 6.12, w: 6.68, h: 0.45, fontSize: 9.5 });
  s.addNotes("Every five steps the controller looks at how long each thread spent in the force loop and computes lambda, the slowest "
    + "thread relative to the average. Above 0.15 it switches to dynamic; below 0.05 it switches back. Two thresholds stop it flipping "
    + "on noise.\nThe synopsis algorithm measured lambda directly in both states. Under dynamic that is always near zero, so it would "
    + "oscillate. We predict what static would cost from each thread's measured speed and the work in its static block.\nLive demo: "
    + "python demo.py --part 2 shows lambda rising and the switch.");
}

// ================================================================== 05 SETUP
pres.addSection({ title: "Evaluation" });
{
  const s = contentSlide("Evaluation", "05", "EXPERIMENTAL SETUP", "Experiments cover four inputs, three sizes, 1–8 threads and 1–4 ranks",
    "Medians of three shuffled repetitions on one hybrid laptop, with threads pinned to the performance cores.");
  const rows = [
    ["HARDWARE", "Intel i7-14700HX (8 P + 12 E cores)  ·  RTX 4060 Laptop GPU"],
    ["SOFTWARE", "GCC 15.1 -O3 -march=native  ·  MS-MPI 10.1  ·  NVRTC 12.9"],
    ["INPUTS", "uniform (balanced control)  ·  plummer  ·  collide  ·  collapse"],
    ["SIZES", "N = 4,096 · 16,384 · 65,536  ·  150 steps, dt = 0.01, θ = 0.5"],
    ["PARALLEL", "1, 2, 4, 8 threads per rank  ·  1, 2, 4 MPI ranks"],
    ["METRICS", "S(p) = T(1) / T(p)  ·  E(p) = S(p) / p  ·  λ per step"],
  ];
  rows.forEach(([k, v], i) => {
    const y = TOP + i * 0.64;
    rule(s, L, y, 7.4);
    mono(s, k, { x: L, y: y + 0.2, w: 1.5, h: 0.25, fontSize: 9, color: ACC, charSpacing: 1 });
    text(s, v, { x: 2.35, y: y + 0.16, w: 5.8, h: 0.45, fontSize: 13 });
  });
  rule(s, L, TOP + 6 * 0.64, 7.4);
  // correctness hero
  mono(s, "CORRECTNESS", { x: 8.95, y: TOP, w: 3.63, h: 0.22, fontSize: 9, color: MUTED, charSpacing: 1 });
  text(s, "8/8", { x: 8.95, y: TOP + 0.3, w: 3.63, h: 1.0, fontFace: MONO, fontSize: 54, bold: true, color: ACC });
  mono(s, "automated validation checks passed", { x: 8.95, y: TOP + 1.32, w: 3.63, h: 0.25, fontSize: 10, color: INK });
  const facts = [["0.25 %", "rms Barnes–Hut force error, θ = 0.5"],
                 ["1.1 × 10⁻⁷", "max energy drift, direct sum"],
                 ["1 checksum", "across 13 thread / rank / schedule configurations"]];
  facts.forEach(([a, b], i) => {
    const y = TOP + 1.85 + i * 0.68;
    rule(s, 8.95, y, 3.63);
    text(s, a, { x: 8.95, y: y + 0.08, w: 3.63, h: 0.3, fontFace: MONO, fontSize: 14, bold: true });
    mono(s, b, { x: 8.95, y: y + 0.38, w: 3.63, h: 0.25, fontSize: 9.5 });
  });
  s.addNotes("All measurements are medians of three repetitions run in shuffled order, so slow drift affects every schedule equally. "
    + "Threads are pinned to the eight performance cores because the slower efficiency cores alone produced lambda of about 0.3 on a "
    + "balanced job.\nLater experiments ran behind a thermal gate after we saw the laptop throttle; throttled data is kept in "
    + "results/archive.\nCorrectness: the final state is bit-identical for every configuration. Live option: python demo.py --part 1.");
}

// ================================================================== 06 RESULT: IMBALANCE OVER TIME
{
  const s = contentSlide("Evaluation", "06", "RESULTS · IMBALANCE", "Imbalance appears only once the core forms, and the controller follows it",
    `Cold collapse, N = 65,536, 8 threads. Work imbalance stays below 0.05 until t = 1.00, then rises to ${f2(tr.work_lam_max)}.`);
  const labels = tr.time.map((t) => t.toFixed(2));
  s.addChart(pres.charts.LINE, [
    { name: "static run, measured λ", labels, values: tr.static_lam },
    { name: "work imbalance of static blocks", labels, values: tr.work_lam },
    { name: "adaptive run, measured λ", labels, values: tr.adaptive_lam },
  ], Object.assign(CHART_BASE(), {
    x: L - 0.1, y: TOP, w: 8.75, h: 4.0, chartColors: [LIGHT, INK, ACC], lineSize: 2, lineDataSymbol: "none",
    catAxisLabelFrequency: 25, valAxisMinVal: 0, valAxisMaxVal: 0.7, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
    showLegend: true, legendPos: "t", showCatAxisTitle: true, catAxisTitle: "simulation time",
    showValAxisTitle: true, valAxisTitle: "λ (5-step mean)",
  }));
  const ix = 9.75;
  const notes = [["t = " + tr.first_switch.toFixed(2), "brief switch on a timing spike, reverted by the 0.05 threshold"],
                 ["t = " + tr.sustained_switch.toFixed(2), "sustained switch as the core forms"],
                 ["λ " + f2(tr.static_lam[tr.static_lam.length - 1]) + " → " + f2(tr.adaptive_lam[tr.adaptive_lam.length - 1]),
                  "measured λ at the last step, static vs adaptive"]];
  mono(s, "ADAPTIVE RUN", { x: ix, y: TOP, w: 2.83, h: 0.22, fontSize: 9, charSpacing: 1 });
  notes.forEach(([a, b], i) => {
    const y = TOP + 0.45 + i * 1.12;
    rule(s, ix, y, 2.83);
    text(s, a, { x: ix, y: y + 0.12, w: 2.83, h: 0.4, fontFace: MONO, fontSize: 18, bold: true, color: i === 1 ? ACC : INK });
    text(s, b, { x: ix, y: y + 0.55, w: 2.83, h: 0.5, fontSize: 12, color: DARKG });
  });
  rule(s, ix, TOP + 0.45 + 3 * 1.12, 2.83);
  s.addNotes("This is one benchmark run of the collapsing cloud. The dark line is the imbalance implied by counted work: almost zero "
    + "while the cloud is uniform, then up to 0.64 once the core forms after t = 1.\nGrey is what static actually measured; vermilion is "
    + "the adaptive run. Its controller reacted briefly to a noise spike at t = 0.41 and the 0.05 threshold sent it back, then switched "
    + "for good at t = 1.16, after which its measured imbalance stays near zero.\nLive option: results/presentation.html, section 3.");
}

// ================================================================== 07 RESULT: SCHEDULES
{
  const s = contentSlide("Evaluation", "07", "RESULTS · SCHEDULES", "Adaptive beats static in all nine clustered cases; guided gains more",
    `Time saved vs static at 8 threads. Speed-up at 8 threads (N = 65,536): ${spdMin.toFixed(1)}–${spdMax.toFixed(1)}×.`);
  const cases = [];
  for (const n of [4096, 16384, 65536]) for (const ic of ["plummer", "collide", "collapse"]) cases.push([ic, n]);
  const labels = cases.map(([ic, n]) => ic + "\n" + n.toLocaleString("en-US"));
  const sav = (ic, n, sch) => 100 * (1 - D.rel8[ic + "_" + n][sch]);
  s.addChart(pres.charts.BAR, [
    { name: "guided", labels, values: cases.map(([ic, n]) => sav(ic, n, "guided")) },
    { name: "adaptive (ours)", labels, values: cases.map(([ic, n]) => sav(ic, n, "adaptive")) },
  ], Object.assign(CHART_BASE(), {
    x: L - 0.1, y: TOP, w: 8.15, h: 4.0, barDir: "col", barGrouping: "clustered", chartColors: [LIGHT, ACC],
    barGapWidthPct: 55, valAxisMinVal: 0, valAxisMaxVal: 20, valAxisMajorUnit: 5, valAxisLabelFormatCode: "0",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontFace: MONO, dataLabelFontSize: 8.5,
    dataLabelColor: DARKG, showLegend: true, legendPos: "t", showValAxisTitle: true, valAxisTitle: "time saved vs static [%]",
    catAxisLabelFontSize: 9,
  }));
  const hx = 9.3, hw = 3.28;
  mono(s, "CLUSTERED CASES FASTER THAN STATIC", { x: hx, y: TOP, w: hw, h: 0.22, fontSize: 9, charSpacing: 1 });
  text(s, D.adaptive_gain_clustered.count_faster + "/" + D.adaptive_gain_clustered.count,
       { x: hx, y: TOP + 0.28, w: hw, h: 1.0, fontFace: MONO, fontSize: 54, bold: true, color: ACC });
  mono(s, `adaptive saves ${pct(gainMin)}–${pct(gainMax)} % of run time`, { x: hx, y: TOP + 1.3, w: hw, h: 0.25, fontSize: 10, color: INK });
  const u = D.adaptive_uniform, uDev = Math.max(...Object.values(u).map((v) => Math.abs(1 - v)));
  const facts = [["uniform input", `within ${pct(uDev)} % of static, all sizes`],
                 ["small balanced job", `dynamic +${pct(D.dynamic_small_uniform - 1, 0)} % time; adaptive stays static`],
                 ["best fixed schedule", `guided in ${D.best_counts.guided} of 12 cases, adaptive in ${D.best_counts.adaptive}`]];
  facts.forEach(([a, b], i) => {
    const y = TOP + 1.8 + i * 0.75;
    rule(s, hx, y, hw);
    mono(s, a.toUpperCase(), { x: hx, y: y + 0.08, w: hw, h: 0.22, fontSize: 8.5, color: ACC, charSpacing: 1 });
    text(s, b, { x: hx, y: y + 0.32, w: hw, h: 0.3, fontSize: 12 });
  });
  s.addNotes("Each pair of bars is one clustered case at 8 threads: how much run time each schedule saves compared with static. "
    + "Adaptive is faster than static in all nine cases, between 1.5 and 16.6 percent.\nOn the uniform input it stays within 2.3 percent of "
    + "static, which is what the synopsis predicted, and it avoids the 46 percent penalty that dynamic pays on a small balanced job.\n"
    + "Guided was the best fixed schedule in nine of twelve cases; we say so plainly and discuss why on the next slide.");
}

// ================================================================== 08 GPU, LIMITATIONS, FUTURE
pres.addSection({ title: "Analysis" });
{
  const s = contentSlide("Analysis", "08", "GPU · LIMITATIONS · NEXT STEPS", "The GPU removes the O(N²) cost; imbalance between ranks remains open",
    "Tiled CUDA direct sum vs 8 CPU threads, and what the measurements say we should do next.");
  const labels = D.gpu_direct.map((g) => g.n.toLocaleString("en-US"));
  s.addChart(pres.charts.LINE, [
    { name: "CPU direct, 8 threads", labels, values: D.gpu_direct.map((g) => g.cpu_ms) },
    { name: "GPU fp64", labels, values: D.gpu_direct.map((g) => g.gpu64_ms) },
    { name: "GPU fp32", labels, values: D.gpu_direct.map((g) => g.gpu32_ms) },
  ], Object.assign(CHART_BASE(), {
    x: L - 0.1, y: TOP + 0.95, w: 5.55, h: 3.05, chartColors: [GREY, LIGHT, ACC], lineSize: 2, lineDataSymbol: "circle",
    lineDataSymbolSize: 6, valAxisLogScaleBase: 10, valAxisMinVal: 0.1, valAxisMaxVal: 10000, valAxisLabelFormatCode: "General",
    showLegend: true, legendPos: "t", showValAxisTitle: true, valAxisTitle: "ms per step (log)", showCatAxisTitle: true, catAxisTitle: "N",
  }));
  text(s, gpuX.toFixed(0) + "×", { x: L, y: TOP - 0.05, w: 2.3, h: 0.85, fontFace: MONO, fontSize: 48, bold: true, color: ACC });
  mono(s, `fp32 kernel vs 8 CPU threads, N = 65,536
${gpu65.gpu32_ms.toFixed(1)} vs ${gpu65.cpu_ms.toFixed(0)} ms per step
rms force error ${sci(gpu65.err32)}`,
       { x: 2.85, y: TOP + 0.08, w: 3.4, h: 0.75, fontSize: 9.5 });

  const cx = 6.9, cw = 5.68;
  const blocks = [
    ["LIMITATIONS", [
      "Guided matched or beat adaptive in 9 of 12 cases",
      "A timing spike can trigger a false switch (uniform, 1 × 8: +17 % time)",
      `Imbalance between ranks reached λ = ${D.mpi_rank_lam_max.toFixed(2)} (threads only fixed)`]],
    ["NEXT STEPS", [
      "Cost-weighted rank partitioning: up to 17.2 % faster",
      "Controller that also picks guided; multi-node runs"]],
  ];
  let y = TOP;
  blocks.forEach(([h, items]) => {
    mono(s, h, { x: cx, y, w: cw, h: 0.22, fontSize: 9, color: ACC, charSpacing: 1 });
    y += 0.3;
    items.forEach((it) => {
      rule(s, cx, y, cw);
      text(s, it, { x: cx, y: y + 0.12, w: cw, h: 0.35, fontSize: 13 });
      y += 0.55;
    });
    rule(s, cx, y, cw);
    y += 0.35;
  });
  s.addNotes("On the GPU, the tiled kernel computes the exact all-pairs force 88 times faster than eight CPU threads in single precision; "
    + "double precision is slow on this laptop GPU. The Barnes-Hut hybrid gains 1.3 to 1.4 times.\nLimitations: guided is the strongest "
    + "fixed schedule here, a time-based trigger can react to noise, and the OpenMP controller cannot fix imbalance between MPI ranks.\n"
    + "Vishal implemented two next steps already: cost-weighted rank partitioning and an auto mode that tries dynamic and guided.\n"
    + "Live option: python demo.py --part 4 runs the GPU comparison.");
}

// ================================================================== CLOSING
pres.addSection({ title: "Closing" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Closing" });
  text(s, "Thank you", { x: L, y: 2.35, w: 8, h: 1.2, fontSize: 60, bold: true, color: PAPER });
  text(s, "Questions?", { x: L, y: 3.6, w: 8, h: 0.6, fontSize: 24, color: ONDARK });
  rule(s, L, 4.75, 5.5, "3A3936");
  mono(s, "Tanish Hemant Kuvar  ·  M. Vishal Kumar  ·  Samyak Jain", { x: L, y: 4.95, w: 8, h: 0.3, fontSize: 11, color: PAPER });
  mono(s, "github.com/tanishkuvar11/N-Body-Simulation", { x: L, y: 5.35, w: 8, h: 0.3, fontSize: 11, color: ONDARK });
  s.addNotes("Invite questions. Have results/presentation.html open in a browser tab and the demo terminal ready in case the professor "
    + "wants to see a run again.");
}

(async () => {
  await pres.writeFile({ fileName: OUTFILE });
  await applyTheme(OUTFILE, THEME);
  console.log("wrote", OUTFILE);
})();
