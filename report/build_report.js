// IEEE-conference-format report (A4, two columns, Times New Roman) -> PCAP_Report_IEEE.docx
// All numbers come from ../presentation/data.json and data_extra.json (extracted from results/*.csv).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  BorderStyle, SectionType, TabStopType, LevelFormat, Tab, VerticalAlign,
} = require("docx");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "presentation", "data.json"), "utf8"));
const X = JSON.parse(fs.readFileSync(path.join(__dirname, "data_extra.json"), "utf8"));
const FIG = (f) => path.join(__dirname, "fig", f);

// ------------------------------------------------------------------ geometry (IEEE A4 template)
const PAGE_W = 11906, PAGE_H = 16838;
const MARGIN = { top: 1077, bottom: 2438, left: 812, right: 812 };      // 19 / 43 / 14.32 / 14.32 mm
const COL_GAP = 239;                                                    // 4.22 mm
const TEXT_W = PAGE_W - MARGIN.left - MARGIN.right;                     // 10282
const COL_W = (TEXT_W - COL_GAP) / 2;                                   // 5021.5 dxa = 3.49 in
const FONT = "Times New Roman";

// ------------------------------------------------------------------ inline mini-markup
// *italic*  **bold**  _{sub}  ^{sup}  `mono`
function runs(str, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\}|`[^`]+`)/g;
  let last = 0, m;
  const push = (text, o) => text && out.push(new TextRun(Object.assign({ text, font: FONT }, base, o)));
  while ((m = re.exec(str))) {
    push(str.slice(last, m.index), {});
    const t = m[0];
    if (t.startsWith("**")) push(t.slice(2, -2), { bold: true });
    else if (t.startsWith("*")) push(t.slice(1, -1), { italics: true });
    else if (t.startsWith("_{")) out.push(...runs(t.slice(2, -1), Object.assign({}, base, { subScript: true })));
    else if (t.startsWith("^{")) out.push(...runs(t.slice(2, -1), Object.assign({}, base, { superScript: true })));
    else push(t.slice(1, -1), { font: "Courier New", size: (base.size || 20) - 2 });
    last = m.index + t.length;
  }
  push(str.slice(last), {});
  return out;
}

const body = (s, o = {}) => new Paragraph(Object.assign({
  children: runs(s), alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 202 }, spacing: { after: 0, line: 240 },
}, o));
const bodyNoIndent = (s, o = {}) => body(s, Object.assign({ indent: { firstLine: 0 } }, o));

let secNo = 0, subNo = 0;
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
function H1(text, numbered = true) {
  subNo = 0;
  const label = numbered ? ROMAN[secNo++] + ". " : "";
  return new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: label + text, smallCaps: true, font: FONT, size: 20 })] });
}
function H2(text) {
  const label = String.fromCharCode(65 + subNo++) + ". ";
  return new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 },
    children: [new TextRun({ text: label + text, italics: true, font: FONT, size: 20 })] });
}
let eqNo = 0;
function EQ(s) {
  eqNo++;
  return new Paragraph({ spacing: { before: 80, after: 80 },
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(COL_W / 2) }, { type: TabStopType.RIGHT, position: Math.floor(COL_W) }],
    children: [new TextRun({ children: [new Tab()], font: FONT }), ...runs(s), new TextRun({ children: [new Tab(), `(${eqNo})`], font: FONT })] });
}
function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
let figNo = 0;
function FIGURE(file, caption, widthIn = 3.45) {
  figNo++;
  const { w, h } = pngSize(file);
  const pw = Math.round(widthIn * 96), ph = Math.round(pw * h / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: pw, height: ph },
        altText: { title: `Fig. ${figNo}`, description: caption, name: `fig${figNo}` } })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 140 },
      children: runs(`Fig. ${figNo}. ` + caption, { size: 16 }) }),
  ];
}
let tabNo = 0;
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const RULE = { style: BorderStyle.SINGLE, size: 6, color: "000000" };
function TABLE(title, header, rows, widthsRel, align) {
  tabNo++;
  const total = Math.floor(COL_W) - 20;
  const sum = widthsRel.reduce((a, b) => a + b, 0);
  const widths = widthsRel.map((x) => Math.floor(total * x / sum));
  widths[widths.length - 1] += total - widths.reduce((a, b) => a + b, 0);
  const cell = (txt, i, isHead, isLastRow) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 20, bottom: 20, left: 50, right: 50 }, verticalAlign: VerticalAlign.CENTER,
    borders: { top: NONE, left: NONE, right: NONE, bottom: isHead ? RULE : NONE },
    children: [new Paragraph({ alignment: (align && align[i]) || (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER),
      children: runs(txt, { size: 16, bold: isHead ? false : undefined }) })],
  });
  const trs = [new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true)) })]
    .concat(rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, ri === rows.length - 1)) })));
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 140, after: 0 },
      children: [new TextRun({ text: "TABLE " + ROMAN[tabNo - 1], font: FONT, size: 16 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 60 },
      children: [new TextRun({ text: title, smallCaps: true, font: FONT, size: 16 })] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      borders: { top: RULE, bottom: RULE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
      rows: trs }),
    new Paragraph({ spacing: { after: 100 }, children: [] }),
  ];
}
const f2 = (v) => v.toFixed(2), pct = (v, d = 1) => (100 * v).toFixed(d);

// ------------------------------------------------------------------ numbers
const tr = D.trace, sn = D.snap, sp = D.speedup65536;
const gMin = pct(D.adaptive_gain_clustered.min), gMax = pct(D.adaptive_gain_clustered.max);
const uDev = pct(Math.max(...Object.values(D.adaptive_uniform).map((v) => Math.abs(1 - v))));
const spd8 = []; for (const ic in sp) for (const s in sp[ic]) spd8.push(sp[ic][s][3]);
const spdMin = Math.min(...spd8).toFixed(1), spdMax = Math.max(...spd8).toFixed(1);
const g65 = D.gpu_direct.find((g) => g.n === 65536), gpuX = (g65.cpu_ms / g65.gpu32_ms).toFixed(0);
const dynSmall = pct(D.dynamic_small_uniform - 1, 0);
const mpi = Object.fromEntries(X.mpi.map((r) => [r.cfg, r]));
const rankLamMax = Math.max(...X.mpi.map((r) => r.rank_lam)).toFixed(2);
const ctl = X.ctl;

// ------------------------------------------------------------------ title block (single column)
const title = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "Adaptive Scheduling for Load-Balanced Parallel N‑Body Simulation: A Hybrid OpenMP–MPI–CUDA Approach", font: FONT, size: 48 })] }),
];
const authors = [["Tanish Hemant Kuvar", "240962540"], ["M. Vishal Kumar", "240962546"], ["Samyak Jain", "240962638"]];
const aw = Math.floor(TEXT_W / 3);
title.push(new Table({
  width: { size: aw * 3, type: WidthType.DXA }, columnWidths: [aw, aw, aw], alignment: AlignmentType.CENTER,
  borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE },
  rows: [new TableRow({ children: authors.map(([n, r]) => new TableCell({
    width: { size: aw, type: WidthType.DXA }, borders: { top: NONE, bottom: NONE, left: NONE, right: NONE },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: n, font: FONT, size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Reg. No. " + r, font: FONT, size: 20 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "AIML-A", font: FONT, size: 20, italics: true })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Manipal Institute of Technology", font: FONT, size: 20, italics: true })] }),
    ] })) })],
}));
title.push(new Paragraph({ spacing: { after: 200 }, children: [] }));

// ------------------------------------------------------------------ body (two columns)
const B = [];
const abs = (label, text) => new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100 }, indent: { firstLine: 202 },
  children: [new TextRun({ text: label, bold: true, italics: true, font: FONT, size: 18 }), ...runs(text, { size: 18, bold: true })] });

B.push(abs("Abstract—", `Gravitational N-body systems cluster as they evolve, so an equal division of bodies among threads stops being an equal division of work. We present a hybrid MPI–OpenMP–CUDA Barnes–Hut simulator in which bodies are Morton-ordered, so that every rank and every thread owns a compact region of space, and in which a lightweight run-time controller selects the OpenMP loop schedule from measured per-thread loop times. The controller switches between static and dynamic scheduling with hysteresis through the standard omp_set_schedule() interface; to avoid oscillation it predicts the imbalance a static schedule would have while dynamic scheduling is active. On a laptop processor with eight performance cores the simulator reaches a speed-up of ${spdMin}–${spdMax}× at 8 threads for N = 65,536 and produces bit-identical results across all thread counts, rank counts and schedules. The controller stays within ${uDev} % of static scheduling on a balanced input and is faster than static in all nine clustered configurations, by ${gMin}–${gMax} %; the built-in guided schedule, however, matched or exceeded it in ${D.best_counts.guided} of 12 cases. A tiled CUDA direct-summation kernel is ${gpuX}× faster than 8 CPU threads in single precision.`));
B.push(abs("Index Terms—", "N-body simulation, Barnes–Hut, MPI, OpenMP, CUDA, load balancing, adaptive loop scheduling."));

// I. Introduction
B.push(H1("Introduction"));
B.push(body("An N-body simulation advances N mutually interacting bodies one timestep at a time. Under softened Newtonian gravity the acceleration of body *i* is"));
B.push(EQ("*a*_{*i*} = *G* ∑_{*j*≠*i*} *m*_{*j*} (*x*_{*j*} − *x*_{*i*}) / (|*x*_{*j*} − *x*_{*i*}|^{2} + ε^{2})^{3/2}"));
B.push(bodyNoIndent("where ε is a softening length. Direct summation needs N(N−1)/2 interactions per step, about 2.1 × 10^{9} at N = 65,536. The Barnes–Hut method [1] reduces the cost to O(N log N) by treating distant groups of bodies as single masses, but large systems still need parallel hardware: MPI across nodes, OpenMP across cores and CUDA on GPUs."));
B.push(body("The cost of a body in Barnes–Hut grows with the density around it, because a body in a dense region opens more tree cells. As a system clusters, the threads that own dense regions do far more work than those that own sparse ones. A fixed static schedule then leaves threads idle, while a fixed dynamic schedule pays scheduling overhead even when the load is uniform. Because clustering develops during a run, the best schedule changes within a single run, which makes run-time switching attractive."));
B.push(body("This report makes four contributions:"));
const contributions = [
  "a hybrid MPI–OpenMP–CUDA Barnes–Hut simulator with a Morton-ordered domain decomposition whose final state is bit-identical across schedules, thread counts and rank counts;",
  "a run-time schedule controller built only on omp_set_schedule(), including a correction to the synopsis algorithm that predicts static thread times while dynamic scheduling is active;",
  "an evaluation on uniform and clustered inputs (four inputs, N up to 65,536, 1–8 threads, 1–4 ranks) with measurement controls for a hybrid, thermally limited laptop;",
  "CUDA kernels compiled at run time with NVRTC: a tiled direct-summation kernel and a Barnes–Hut near-field hybrid.",
];
contributions.forEach((c) => B.push(new Paragraph({ numbering: { reference: "contrib", level: 0 }, alignment: AlignmentType.JUSTIFIED, children: runs(c) })));
B.push(body("Section II reviews related work, Section III states the problem, Section IV describes the design, Section V the methodology, Section VI the results, and Sections VII–IX discuss limitations, future work and conclusions."));

// II. Related work
B.push(H1("Related Work"));
B.push(H2("Parallel N-Body Methods"));
B.push(body("Singh et al. [2] first showed that tree codes need cost-aware partitioning, and the tiled CUDA kernel of Nyland et al. [3] and the all-GPU Barnes–Hut of Burtscher and Pingali [4] remain the standard GPU baselines. Keller et al. [5] built Cornerstone, which constructs distributed octrees entirely on GPUs and ran a Barnes–Hut code with up to 8 trillion particles. Cazalbou et al. [6] proposed hybrid MPI–GPU algorithms for distributed octree construction, and Nagarajan et al. [7] reformulated Barnes–Hut as a ray-tracing problem for GPU ray-tracing cores."));
B.push(H2("Automatic OpenMP Loop Scheduling"));
B.push(body("LB4OMP [8] added over a dozen dynamic and adaptive loop scheduling techniques to the LLVM OpenMP runtime. Auto4OMP [9] selects the algorithm and chunk size during execution, partly using a load-imbalance threshold, and Ciorba et al. [10] summarised this automated selection. Müller Korndörfer et al. [11] compared expert-based and reinforcement-learning selection, found run-time selection beneficial, and named CPU–GPU and cross-node selection as future work. Da Silva et al. [12] auto-tuned the chunk size of dynamic scheduling for full waveform inversion."));
B.push(H2("Algorithm Selection in Particle Simulations"));
B.push(body("AutoPas [13] selects among short-range particle algorithms and parallelisations at run time; Newcome et al. extended it to multi-site molecular dynamics [14] and compared selection strategies [15]."));
B.push(body("Automated schedule selection is mature at the node level, but [8]–[11] rely on custom runtimes, and recent N-body work [5]–[7] targets force computation rather than thread scheduling. We test a portable controller that uses only the standard OpenMP interface, on a clustering gravitational workload inside a full MPI–OpenMP–CUDA pipeline."));

// III. Problem statement
B.push(H1("Problem Statement and Objectives"));
B.push(body("In parallel N-body simulation an equal division of bodies does not give an equal division of work once bodies cluster, and no single OpenMP schedule is optimal for a whole run. We ask whether a run-time controller that measures per-thread load and switches the OpenMP schedule mid-simulation can reduce total execution time compared with fixed static, dynamic and guided schedules, in a hybrid MPI–OpenMP–CUDA implementation. The objectives were: (O1) sequential direct-sum and Barnes–Hut baselines with energy and momentum checks; (O2) OpenMP force computation and a characterisation of static, dynamic and guided schedules; (O3) MPI distribution across ranks and a CUDA force kernel; (O4) the adaptive controller, evaluated on uniform and clustered inputs. All four were implemented and measured."));

// IV. Design
B.push(H1("Design and Implementation"));
B.push(H2("Simulation Core"));
B.push(body("Bodies are advanced with the kick–drift–kick leapfrog integrator (G = 1, ε = 0.05, Δt = 0.01). Forces use Barnes–Hut with opening angle θ = 0.5: a cell of size *s* at distance *d* from its centre of mass is used as a single mass if *s*/*d* < θ and the cell does not contain the body itself; leaves hold at most 8 bodies. Every step all bodies are sorted along a Morton (Z-order) curve with a stable radix sort of 63-bit keys (21 bits per axis). Because the order is sorted, every octree cell owns a contiguous index range, so the tree is built top-down by binary search on the next three key bits; single-child chains are collapsed, giving at most 2N−1 nodes."));
B.push(H2("Hybrid Decomposition"));
B.push(body("Fig. 1 shows one timestep. Each MPI rank owns a contiguous block of the Morton order. After the first half-kick and drift of its own block, every rank exchanges positions and velocities with MPI_Allgatherv (block sizes may differ), so all ranks hold the full state; each rank then sorts all bodies and builds the full tree, and computes forces only for its own block. Inside a rank the force loop is an OpenMP loop with schedule(runtime), so the schedule can be changed between steps without recompilation. Each thread writes only the accelerations of its own bodies, so the loop has no data races. A per-body work count (tree nodes visited plus interactions) is recorded for the controller."));
B.push(...FIGURE(FIG("fig1_pipeline.png"), "One timestep on every MPI rank. Highlighted: the OpenMP force loop whose schedule is chosen at run time, and the controller that chooses it."));
B.push(body("Morton ordering makes the tree walk cache-friendly, but it also means a static block is a compact region of space. When that region contains a cluster, one thread receives most of the expensive bodies (Fig. 2)."));
B.push(...FIGURE(FIG("fig2_stars.png"), `Cold collapse, N = ${sn.n.toLocaleString("en-US")}, 8 threads, t = ${sn.t.toFixed(2)}. Highlighted: bodies computed by the busiest thread. (a) Under static scheduling it owns one block of the dense core and takes ${sn.ratio_max_min.toFixed(1)}× as long as the least-loaded thread (λ = ${f2(sn.static_lam)}). (b) After the controller switches to dynamic its bodies are spread over the cloud.`));
B.push(H2("CUDA Kernels"));
B.push(body("The usual CUDA compiler requires the MSVC host compiler on Windows, which was not available, so the kernels are compiled at run time with NVRTC for the detected device (a native cubin, with a PTX fallback) and launched through the CUDA Driver API; both libraries are loaded dynamically, so the program also runs without a GPU. The direct-summation kernel follows Nyland et al. [3]: each thread computes one body, and a block of 256 threads stages 256 bodies at a time in shared memory. In the hybrid Barnes–Hut mode, CPU threads walk the tree and accumulate the far field; near-field leaves are recorded as index ranges, contiguous thanks to the Morton order and merged when adjacent, and a second kernel evaluates those pairs. Both kernels support single and double precision."));
B.push(H2("Adaptive Controller"));
B.push(body("Every *k* steps the controller computes the imbalance of the per-thread loop times *t*_{*i*}, accumulated over the window:"));
B.push(EQ("λ = max_{*i*} *t*_{*i*} / mean_{*i*} *t*_{*i*} − 1"));
B.push(bodyNoIndent("Two thresholds give hysteresis (Algorithm 1). Under dynamic scheduling, however, the measured *t*_{*i*} are balanced by construction: λ would always fall below the lower threshold, and the controller of the synopsis would switch back to static at every check and oscillate. While dynamic is active, the controller therefore evaluates the thread times that static would produce,"));
B.push(EQ("*T*_{*i*} = *B*_{*i*} / *v*_{*i*},   *v*_{*i*} = *W*_{*i*} / *t*_{*i*}"));
B.push(bodyNoIndent("where *B*_{*i*} is the work in thread *i*’s static block and *W*_{*i*} the work it actually executed. Under static, *W*_{*i*} = *B*_{*i*} and *T*_{*i*} = *t*_{*i*}, which is exactly the synopsis metric. The prediction includes both workload clustering and differences in core speed."));
const ALG = [
  "**Algorithm 1:** Adaptive schedule controller (per MPI rank)",
  "1: **input:** λ_{low} = 0.05, λ_{high} = 0.15, *k* = 5, chunk *c* = 8; policy ← STATIC",
  "2: **for** each timestep *s* **do**",
  "3:   force loop with schedule(runtime); add *t*_{*i*}, *W*_{*i*}, *B*_{*i*} to the window",
  "4:   **if** *s* mod *k* = 0 **then**",
  "5:     λ ← imbalance of *t*_{*i*} (STATIC) or of *T*_{*i*} (DYNAMIC)",
  "6:     **if** STATIC **and** λ > λ_{high} **then** policy ← DYNAMIC(*c*)",
  "7:     **else if** DYNAMIC **and** λ < λ_{low} **then** policy ← STATIC",
  "8:     omp_set_schedule(policy, *c*); reset window",
  "9:   second half-kick",
];
ALG.forEach((l, i) => B.push(new Paragraph({ spacing: { before: i === 0 ? 120 : 0, after: i === ALG.length - 1 ? 140 : 0 },
  border: i === 0 ? { top: RULE, bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000" } } : (i === ALG.length - 1 ? { bottom: RULE } : undefined),
  indent: i === 0 ? undefined : { left: 120 }, children: runs(l, { size: 17 }) })));
B.push(H2("Determinism and Measurement"));
B.push(body("Each body’s force is computed with the same operation sequence whichever thread or rank computes it, and every rank sorts and builds the same tree, so the final state is bit-identical for every schedule, thread count, rank count and controller decision. Per-thread times use QueryPerformanceCounter, because omp_get_wtime() in the MinGW runtime has a resolution of 1 ms. The test CPU mixes performance and efficiency cores; unpinned threads produced λ ≈ 0.3 on a perfectly balanced input, so threads are pinned one per performance core, detected through the Windows EfficiencyClass."));

// V. Methodology
B.push(H1("Experimental Methodology"));
B.push(...TABLE("Experimental Setup", ["Item", "Configuration"], [
  ["CPU", "Intel Core i7-14700HX, 8 P + 12 E cores"],
  ["GPU", "NVIDIA RTX 4060 Laptop GPU, 8 GB"],
  ["Software", "GCC 15.1 -O3 -march=native, MS-MPI 10.1, NVRTC 12.9"],
  ["Inputs", "uniform, Plummer, colliding clusters, cold collapse"],
  ["Sizes", "N = 4,096, 16,384, 65,536; 150 steps"],
  ["Parallel", "1, 2, 4, 8 threads per rank; 1, 2, 4 ranks"],
  ["Protocol", "3 repetitions, shuffled order, median"],
], [1, 3.2], [AlignmentType.LEFT, AlignmentType.LEFT]));
B.push(body("The uniform input is a sphere in homologous expansion whose density stays uniform; its work imbalance stays between 0.006 and 0.012 throughout, making it a balanced control. The Plummer sphere starts in equilibrium (virial ratio 0.986), the colliding clusters are imbalanced from the start, and the cold collapse is uniform at first and forms a dense core after t ≈ 1. Speed-up and efficiency are"));
B.push(EQ("*S*(*p*) = *T*(1) / *T*(*p*),   *E*(*p*) = *S*(*p*) / *p*"));
B.push(bodyNoIndent("with *T*(1) the single-thread static run. Long sessions throttled the laptop by up to about 40 %, so later experiments ran behind a thermal gate: before every eight runs a reference configuration had to be within 8 % of its normal time. Throttled measurements were discarded. Because single-core turbo differed between sessions, MPI speed-ups are reported against the single-thread time of the schedule experiment, the conservative baseline."));

// VI. Results
B.push(H1("Results"));
B.push(H2("Correctness"));
B.push(...TABLE("Validation Checks (All Passed)", ["Check", "Result"], [
  ["Bit-identical state, 13 configurations", "1 checksum"],
  ["GPU direct vs CPU direct (rms)", "1.8×10^{−16} fp64, 1.6×10^{−6} fp32"],
  ["BH+GPU hybrid vs CPU Barnes–Hut", "identical to 7 digits"],
  ["Barnes–Hut force error, θ = 0.3/0.5/0.7/0.9", "0.05/0.25/0.62/1.16 %"],
  ["Max energy drift, direct / Barnes–Hut", "1.1×10^{−7} / 1.4×10^{−5}"],
  ["Leapfrog order (error ratio, halved Δt)", "4.05, 4.02 (ideal 4)"],
  ["Parallel sort and tree build", "identical for 1, 3, 8 threads"],
], [2.4, 1.6]));
B.push(H2("Imbalance Dynamics"));
B.push(body(`Fig. 3 follows one cold-collapse run at N = 65,536 and 8 threads. The work imbalance of the static blocks stays below 0.05 until t = 1.00 and then rises to ${f2(tr.work_lam_max)} as the core forms; the imbalance measured under static follows it. The adaptive run switched briefly at t = ${f2(tr.first_switch)} on a timing spike, which the lower threshold reverted after one window, and permanently at t = ${f2(tr.sustained_switch)}. At the last step the measured imbalance was ${f2(tr.static_lam[tr.static_lam.length - 1])} under static and ${f2(tr.adaptive_lam[tr.adaptive_lam.length - 1])} under adaptive scheduling.`));
B.push(...FIGURE(FIG("fig3_imbalance.png"), "Imbalance during a cold collapse (N = 65,536, 8 threads, 5-step mean). Dashed lines: the controller thresholds 0.05 and 0.15; dotted lines: the adaptive run's switches to dynamic."));
B.push(H2("Schedule Comparison"));
B.push(body(`Table III lists the speed-up at 8 threads and N = 65,536; efficiency ranges from 70 % to 81 %. Fig. 4 shows the run time saved relative to static for all nine clustered configurations at 8 threads. Every balancing schedule beats static in all of them. The adaptive controller is faster than static in 9 of 9 cases, by ${gMin}–${gMax} %, and stays within ${uDev} % of static on the uniform input, as the synopsis predicted. On the small uniform input (N = 4,096), dynamic scheduling with chunk 8 was ${dynSmall} % slower than static, while the controller stayed on static. Guided scheduling was the best fixed schedule in ${D.best_counts.guided} of 12 cases and adaptive in ${D.best_counts.adaptive}. Fig. 5 shows that static falls behind only on the clustered input.`));
const row = (ic, name) => [name, ...["static", "dynamic", "guided", "adaptive"].map((s) => sp[ic][s][3].toFixed(2))];
B.push(...TABLE("Speed-up at 8 Threads, N = 65,536", ["Input", "Static", "Dynamic", "Guided", "Adaptive"],
  [row("uniform", "Uniform"), row("plummer", "Plummer"), row("collide", "Collide"), row("collapse", "Collapse")], [1.4, 1, 1, 1, 1]));
B.push(...FIGURE(FIG("fig4_saved.png"), "Run time saved relative to static at 8 threads for the clustered inputs at N = 4,096, 16,384 and 65,536 (4k, 16k, 64k). P: Plummer sphere; CC: colliding clusters; CL: cold collapse."));
B.push(...FIGURE(FIG("fig5_speedup.png"), "Speed-up versus threads at N = 65,536 for the balanced uniform input and the clustered colliding-cluster input. Dashed: ideal."));
B.push(H2("Controller Sensitivity and Overhead"));
B.push(body(`Table IV varies the controller on the cold collapse (N = 65,536, 8 threads). Checking every 20 steps reacts late to the collapse, and a high threshold of 0.10 switches on noise five times; k = 1–5 with a high threshold of 0.15–0.25 works best. For dynamic scheduling on the colliding clusters, chunk 1 (${ctl.collide_dynamic_c1.t.toFixed(2)} s) was slower than static (${ctl.collide_static.t.toFixed(2)} s), while chunks of 8, 64 and 512 took ${ctl.collide_dynamic_c8.t.toFixed(2)}, ${ctl.collide_dynamic_c64.t.toFixed(2)} and ${ctl.collide_dynamic_c512.t.toFixed(2)} s. On the smallest uniform problem (N = 1,024, 400 steps), the controller took ${X.small.n1024_adaptive.toFixed(3)} s against ${X.small.n1024_static.toFixed(3)} s for static and ${X.small.n1024_dynamic_c1.toFixed(3)} s for dynamic with chunk 1.`));
B.push(...TABLE("Controller Sensitivity, Cold Collapse", ["Setting", "Time [s]", "Switches"], [
  ["k = 1", ctl.k1.t.toFixed(2), String(ctl.k1.sw)], ["k = 5 (default)", ctl.k5.t.toFixed(2), String(ctl.k5.sw)],
  ["k = 10", ctl.k10.t.toFixed(2), String(ctl.k10.sw)], ["k = 20", ctl.k20.t.toFixed(2), String(ctl.k20.sw)],
  ["λ_{high} = 0.10", ctl["high0.1"].t.toFixed(2), String(ctl["high0.1"].sw)],
  ["λ_{high} = 0.25", ctl["high0.25"].t.toFixed(2), String(ctl["high0.25"].sw)],
  ["λ_{high} = 0.40", ctl["high0.4"].t.toFixed(2), String(ctl["high0.4"].sw)],
], [1.6, 1, 1]));
B.push(H2("Hybrid MPI × OpenMP"));
const mrow = (c) => [c.replace(" x ", " × "), mpi[c].uniform_best.toFixed(2), mpi[c].collide_static.toFixed(2),
  mpi[c].collide_adaptive ? mpi[c].collide_adaptive.toFixed(2) : "–", mpi[c].rank_lam.toFixed(3)];
B.push(...TABLE("MPI × OpenMP Speed-up, N = 65,536", ["Ranks × threads", "Uniform (best)", "Collide static", "Collide adaptive", "Rank λ"],
  ["2 x 1", "4 x 1", "1 x 8", "2 x 4", "4 x 2", "2 x 8", "4 x 4"].map(mrow), [1.3, 1, 1, 1, 0.8]));
B.push(body(`Table V reports speed-ups against the single-thread baseline. All configurations produce the same final state. With the same eight cores, 1 rank × 8 threads is the fastest split: every additional rank repeats the sort and tree build and exchanges data every step. The controller improves on static in every configuration on the clustered input, saving 5–18 %. The imbalance between ranks reached λ = ${rankLamMax}, which the OpenMP controller cannot correct. With both hardware threads of the eight performance cores (16 workers) the speed-up reaches ${mpi["4 x 4"].uniform_best.toFixed(1)}×.`));
B.push(H2("GPU Acceleration"));
B.push(body(`At N = 65,536 the single-precision tiled kernel takes ${g65.gpu32_ms.toFixed(1)} ms per step against ${g65.cpu_ms.toFixed(0)} ms for 8 CPU threads, a factor of ${gpuX}, with an rms error of 4.6 × 10^{−6} (Fig. 6); double precision is limited by the low FP64 throughput of the laptop GPU. The Barnes–Hut hybrid with 256-body leaves is ${D.gpu_bh.map((g) => (g.cpu_ms / g.hyb_ms).toFixed(2)).join(", ")}× faster than the best CPU-only configuration at N = 4,096, 16,384 and 65,536; with small leaves the CPU tree walk dominates and the GPU part does not pay off.`));
B.push(...FIGURE(FIG("fig6_gpu.png"), "Direct summation, time per step versus N: CPU with 8 threads and the tiled CUDA kernel in double and single precision."));

// VII. Discussion
B.push(H1("Discussion and Limitations"));
B.push(body("The controller removes the penalty of static scheduling on clustered inputs without requiring a schedule to be chosen by hand, but on this platform guided scheduling was equally good or better. The controller spends the balanced early phase of a run in static before it detects imbalance, and guided scheduling keeps large contiguous Morton blocks for its first chunks, preserving cache locality, while its small final chunks absorb the imbalance."));
B.push(body("A time-based trigger can also react to noise: on the uniform input with 1 rank × 8 threads, the controller switched to dynamic on a timing spike and was 17 % slower than static. Imbalance between MPI ranks is outside the controller’s reach, and all measurements come from a single laptop, where ranks cost more than threads and timings depend on temperature and turbo state. The replicated tree also limits memory scaling for very large N."));

// VIII. Future work
B.push(H1("Future Work"));
B.push(body("Three extensions have been implemented and measured on a second machine (AMD Ryzen 7 7730U): cost-weighted rank partitioning, which cuts the Morton order into pieces of equal measured work and was up to 17.2 % faster with 4 ranks on colliding clusters; a parallel Morton sort and tree build, 2.07–2.53× faster at 8 threads; and an automatic mode in which the controller tries dynamic(8), dynamic(64) and guided(8) for one window each and keeps the fastest. Remaining work is to let the controller tune the chunk size, to distribute the tree instead of replicating it, and to evaluate on a multi-node cluster with GPUs on every node."));

// IX. Conclusion
B.push(H1("Conclusion"));
B.push(body(`We built and validated a hybrid MPI–OpenMP–CUDA Barnes–Hut simulator whose results are bit-identical across all parallel configurations, and a run-time schedule controller that uses only the standard OpenMP interface. As bodies cluster, static scheduling leaves the thread that owns the dense region with up to ${sn.ratio_max_min.toFixed(1)}× the work of the lightest thread. The controller matches static scheduling on balanced input and beats it in all nine clustered configurations, by up to ${gMax} %, but does not outperform the best fixed schedule, guided, on this hardware. The single-precision CUDA kernel is ${gpuX}× faster than 8 CPU threads for direct summation.`));

// Acknowledgment + references
B.push(H1("References", false));
const REFS = [
  "J. Barnes and P. Hut, “A hierarchical O(N log N) force-calculation algorithm,” *Nature*, vol. 324, pp. 446–449, 1986.",
  "J. P. Singh, C. Holt, T. Totsuka, A. Gupta, and J. L. Hennessy, “Load balancing and data locality in adaptive hierarchical N-body methods,” *J. Parallel Distrib. Comput.*, vol. 27, no. 2, pp. 118–141, 1995.",
  "L. Nyland, M. Harris, and J. Prins, “Fast N-body simulation with CUDA,” in *GPU Gems 3*, Addison-Wesley, 2007, ch. 31.",
  "M. Burtscher and K. Pingali, “An efficient CUDA implementation of the tree-based Barnes Hut n-body algorithm,” in *GPU Computing Gems Emerald Ed.*, Morgan Kaufmann, 2011, pp. 75–92.",
  "S. Keller, A. Cavelan, R. Cabezón, L. Mayer, and F. M. Ciorba, “Cornerstone: Octree construction algorithms for scalable particle simulations,” in *Proc. Platform for Advanced Scientific Computing Conf. (PASC)*, 2023.",
  "R. Cazalbou, F. Duchaine, E. Quémerais, B. Andrieu, G. Staffelbach, and B. Maugars, “Hybrid multi-GPU distributed octrees construction for massively parallel code coupling applications,” in *Proc. PASC*, 2024, pp. 14:1–14:11.",
  "V. Nagarajan, R. Gangaraju, K. Sundararajah, A. Pelenitsyn, and M. Kulkarni, “RT–BarnesHut: Accelerating Barnes–Hut using ray-tracing hardware,” in *Proc. ACM SIGPLAN Symp. Principles and Practice of Parallel Programming (PPoPP)*, 2025.",
  "J. H. Müller Korndörfer, A. Eleliemy, A. Mohammed, and F. M. Ciorba, “LB4OMP: A dynamic load balancing library for multithreaded applications,” *IEEE Trans. Parallel Distrib. Syst.*, vol. 33, no. 4, pp. 830–841, 2022.",
  "A. Mohammed, J. H. Müller Korndörfer, A. Eleliemy, and F. M. Ciorba, “Automated scheduling algorithm selection and chunk parameter calculation in OpenMP,” *IEEE Trans. Parallel Distrib. Syst.*, vol. 33, pp. 4383–4394, 2022.",
  "F. M. Ciorba, A. Mohammed, J. H. Müller Korndörfer, and A. Eleliemy, “Automated scheduling algorithm selection in OpenMP,” in *Proc. 22nd Int. Symp. Parallel Distrib. Comput. (ISPDC)*, 2023, pp. 106–109.",
  "J. H. Müller Korndörfer, A. Mohammed, A. Eleliemy, Q. Guilloteau, R. Krummenacher, and F. M. Ciorba, “A comparative study of OpenMP scheduling algorithm selection strategies,” arXiv:2507.20312, 2025.",
  "F. H. S. da Silva, J. B. Fernandes, I. M. Sardina, T. Barros, S. X. de Souza, and I. A. Assis, “Auto-tuning for OpenMP dynamic scheduling applied to full waveform inversion,” *Comput. Geosci.*, vol. 202, art. 105932, 2025.",
  "F. A. Gratl, S. Seckler, H.-J. Bungartz, and P. Neumann, “N ways to simulate short-range particle systems: Automated algorithm selection with the node-level library AutoPas,” *Comput. Phys. Commun.*, vol. 273, art. 108262, 2022.",
  "S. J. Newcome, F. A. Gratl, P. Neumann, and H.-J. Bungartz, “Towards auto-tuning multi-site molecular dynamics simulations with AutoPas,” *J. Comput. Appl. Math.*, vol. 433, art. 115278, 2023.",
  "S. J. Newcome, F. A. Gratl, M. Lerchner, A. Pazar, M. K. Mishra, and H.-J. Bungartz, “Algorithm selection in short-range molecular dynamics simulations,” in *Proc. Int. Conf. Computational Science (ICCS)*, 2025.",
];
REFS.forEach((r, i) => B.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 400, hanging: 400 },
  tabStops: [{ type: TabStopType.LEFT, position: 400 }], spacing: { after: 20 },
  children: [new TextRun({ text: `[${i + 1}]`, font: FONT, size: 16 }), new TextRun({ children: [new Tab()], font: FONT, size: 16 }), ...runs(r, { size: 16 })] })));

// ------------------------------------------------------------------ document
const page = { size: { width: PAGE_W, height: PAGE_H }, margin: MARGIN };
const doc = new Document({
  creator: "Tanish Hemant Kuvar, M. Vishal Kumar, Samyak Jain",
  title: "Adaptive Scheduling for Load-Balanced Parallel N-Body Simulation",
  styles: { default: { document: { run: { font: FONT, size: 20 }, paragraph: { spacing: { line: 240 } } } } },
  numbering: { config: [{ reference: "contrib", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 400, hanging: 250 } } } }] }] },
  sections: [
    { properties: { page }, children: title },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: COL_GAP, equalWidth: true } }, children: B },
  ],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "PCAP_Report_IEEE.docx"), buf);
  console.log("wrote PCAP_Report_IEEE.docx");
});
