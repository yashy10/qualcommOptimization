"""Regenerate the report's vector figure from recorded results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results')
a=p.parse_args()
read=lambda n:json.loads((a.results/(n+'.json')).read_text())
base=read('official_baseline_onnx_fp32'); final=read('b3_kd_lr001_e3_int8')
plt.rcParams.update({'font.size':8,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,(ax,bx)=plt.subplots(1,2,figsize=(10,2.65),gridspec_kw={'width_ratios':[1,1.25]})
points=[('FP32','official_baseline_onnx_fp32',(0,-15)),('FP16','q7_fp16',(5,-16)),('Q3a','q3a_perchannel_pct9999_s8s8',(10,6)),('B3 selected','b3_kd_lr001_e3_int8',(6,-18)),('B3 one epoch','b3_blocks6-9-16_kd_int8',(8,0))]
for label,n,offset in points:
 d=read(n);selected=n==final['name'];color='#006B68' if selected else '#8294A0'
 ax.scatter(d['size_mb'],d['accuracy'],s=85 if selected else 35,c=color,marker='*' if selected else 'o',zorder=3)
 ax.annotate(label,(d['size_mb'],d['accuracy']),xytext=offset,textcoords='offset points',color='#17384A',fontsize=8)
ax.set(xlabel='Model size (MiB)',ylabel='Full-dataset accuracy (%)',xlim=(1.5,9.4),ylim=(96.1,99.5))
ax.grid(alpha=.18);ax.set_title('(a) Storage and accuracy',loc='left',fontweight='bold',color='#17384A')
classes=list(base['per_class_accuracy']); labels=['Annual crop','Forest','Herbaceous veg.','Highway','Industrial','Pasture','Permanent crop','Residential','River','Sea / lake']
x=list(range(len(classes)))
bx.plot(x,[base['per_class_accuracy'][c] for c in classes],'o-',color='#8294A0',label='FP32 baseline',markersize=3,lw=1)
bx.plot(x,[final['per_class_accuracy'][c] for c in classes],'o-',color='#006B68',label='Selected B3 INT8',markersize=3,lw=1)
bx.set_xticks(x,labels,rotation=45,ha='right',fontsize=7)
bx.set(ylabel='Class accuracy (%)',ylim=(95,100.3));bx.grid(axis='y',alpha=.18)
bx.legend(loc='lower right',fontsize=7,frameon=False)
bx.set_title('(b) Accuracy across classes',loc='left',fontweight='bold',color='#17384A')
fig.tight_layout(pad=.8,w_pad=2)
fig.savefig(Path(__file__).with_name('tradeoff.pdf'),bbox_inches='tight')
