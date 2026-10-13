"""Generate the PDF appendix from the JSON records used by dashboard.py.

From submission/report_source: python3 generate_dashboard_appendix.py
Override paths with --results PATH and --output PATH when run elsewhere.
"""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results')
parser.add_argument('--output',type=Path,default=Path(__file__).with_name('dashboard_appendix.tex'))
a=parser.parse_args()
read=lambda name: json.loads((a.results/(name+'.json')).read_text())
runs=[('official_baseline_onnx_fp32','FP32 ONNX'),('official_baseline_pytorch','FP32 PyTorch'),('q0_pertensor_minmax_u8s8','Q0'),('q1_perchannel_minmax_s8s8','Q1'),('q1b_perchannel_minmax_u8s8','Q1b'),('q2_perchannel_entropy_s8s8','Q2'),('q3a_perchannel_pct9999_s8s8','Q3a (selected)'),('q3b_perchannel_pct99999_s8s8','Q3b'),('q4_q3a_firstlast_fp32','Q4'),('q6a_q3a_calib100','Q6a: 100 images'),('q6b_q3a_calib1000','Q6b: 1,000 images'),('q7_fp16','Q7: FP16'),('b3_blocks6-9-16_kd_fp32','B3: pruned FP32'),('b3_blocks6-9-16_kd_int8','B3: pruned INT8')]
lines=[r'''\clearpage
\section*{Appendix A. Dashboard data at a glance}
\noindent\small Recorded experiments from 13 October 2026. Values below come directly from the JSON files behind the dashboard; archived and smoke-test runs are excluded. Teal highlights the submitted model. The first two pages remain the main technical report.
\vspace{5pt}

\noindent\textbf{A1. Full-dataset experiments: all 27,000 EuroSAT images}
\begin{center}
\fontsize{8.5}{10}\selectfont
\setlength{\tabcolsep}{4.3pt}\renewcommand{\arraystretch}{1.2}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrrrrrc@{}}
\toprule
\rowcolor{slate}\textbf{Run} & \textbf{Acc.} & \textbf{Worst} & \textbf{Params} & \textbf{MACs} & \textbf{Size} & \textbf{Latency} & \textbf{RAM} & \textbf{Energy} & \textbf{Time}\\
\rowcolor{slate} & \textbf{\%} & \textbf{\%} & \textbf{M} & \textbf{G} & \textbf{MiB} & \textbf{ms} & \textbf{MiB} & \textbf{J/image} & \textbf{basis}\\
\midrule''']
for name,label in runs:
 d=read(name);t=d.get('milestone_timing',d);basis='M' if 'milestone_timing' in d else 'P' if d.get('timing_status')=='provisional' else 'S'
 row=[label,f"{d['accuracy']:.2f}",f"{d['worst_class_accuracy']:.2f}",f"{d['params_million']:.3f}",f"{d['macs_giga']:.4f}",f"{d['size_mb']:.3f}",f"{t['latency_median_ms']:.3f}",f"{t['peak_ram_mb']:.2f}",f"{t['joules_per_inference']:.5f}",basis]
 prefix=''
 if name.startswith('q3a_'):prefix=r'\rowcolor{pale}';row=[r'\textbf{'+v+'}' for v in row]
 lines.append(prefix+' & '.join(row)+r'\\')
lines.append(r'''\bottomrule
\end{tabular*}
\end{center}
\vspace{-4pt}
\noindent\footnotesize\textbf{Reading the table.} M = median of five milestone rounds; P = provisional single-session measurement; S = single-session reference. Latency is the median within a round. All energy values are CodeCarbon estimates; unusually high provisional values should not be treated as settled comparisons. Worst class is Forest for Q0, Pasture for Q1/Q1b/Q2, and PermanentCrop otherwise. ONNX and PyTorch counts/runtime overhead differ; compare within the same runtime. Q0--Q7 settings are defined in Table 1; B3 removes blocks 6, 9 and 16 and adds one recovery epoch.
\vspace{8pt}

\noindent\small\textbf{A2. Pruning exploration: 2,700-image quick evaluation, before recovery training}

\noindent\footnotesize All candidates use adaptive BN before evaluation. Counts are measured with the same PyTorch/thop method as the reference. These subset accuracies are not directly comparable with A1; latency, RAM and energy were not measured for these Stage A runs.
\vspace{4pt}
''')
left=[('a0_reference_adaptive_bn','Reference + BN')]+[(f'a1_block{i}',f'Drop block {i}') for i in [3,5,6,8,9,10,12,13,15,16]]+[(f'a2_blocks{i}',f'Stack {i} block'+('s' if i>1 else '')) for i in range(1,6)]
right=[(f'a3_global_r{i}',f'Global {i}\\%') for i in [10,20,30,40]]+[(f'a4_uniform_r{i}',f'Uniform {i}\\%') for i in [10,20,30,40]]+[('a5_global_r30_bnscale','30\\% BN-scale'),('a5_global_r30_taylor','30\\% Taylor')]
for idx,group in enumerate([left,right]):
 lines.append(r'\noindent\begin{minipage}[t]{0.49\textwidth}\vspace{0pt}\centering\fontsize{8.5}{10}\selectfont\setlength{\tabcolsep}{2.3pt}\renewcommand{\arraystretch}{1.16}')
 lines.append(r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}\toprule')
 lines.append(r'\rowcolor{slate}\textbf{Candidate} & \textbf{Acc. \%} & \textbf{Worst \%} & \textbf{Params M} & \textbf{MACs G} & \textbf{MiB}\\\midrule')
 for name,label in group:
  d=read('stageA/'+name)
  lines.append(' & '.join([label,f"{d['accuracy']:.2f}",f"{d['worst_class_accuracy']:.2f}",f"{d['params_million']:.3f}",f"{d['macs_giga']:.4f}",f"{d['size_mb']:.3f}"])+r'\\')
 lines.append(r'\bottomrule\end{tabular*}')
 if idx==1:
  lines.append(r'''\par\vspace{8pt}\raggedright\footnotesize
\textbf{Depth-stack key.} Blocks are added in this order: 6, 9, 16, 8, 5. Thus ``Stack 3'' removes 6+9+16; ``Stack 4'' also removes block 8.

\vspace{5pt}\textbf{Width settings.} Global and uniform sweeps use magnitude importance. The requested percentage is a channel-pruning setting, not a percentage of total weights. BN-scale and Taylor are alternative importance criteria at 30\%.

\vspace{5pt}\textbf{Why Q3a was selected.} It satisfies the accuracy and storage targets without recovery training. The smaller B3 INT8 candidate loses 2.40 accuracy points and misses the worst-class target.''')
 lines.append(r'\end{minipage}'+(r'\hfill' if idx==0 else ''))
lines.append(r'''\par\vspace{8pt}
\noindent\footnotesize\textbf{Source and precision.} A1 uses \code{results/*.json}, replacing timing fields with \code{milestone\_timing} when present, as the dashboard does. A2 uses \code{results/stageA/*.json}. Figures are rounded only for display; the saved JSON records retain the measurement precision. M = million parameters; G = billion multiply--accumulate operations; MiB = $2^{20}$ bytes.
''')
a.output.write_text('\n'.join(lines)+'\n')
print(f'Generated appendix: {len(runs)} full-dataset and {len(left)+len(right)} pruning rows.')
