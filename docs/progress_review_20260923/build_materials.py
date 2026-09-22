"""Build the report, numerical tables and vector figures from the frozen evidence."""
import csv,gzip,hashlib,json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
D=Path(__file__).resolve().parent
read=lambda p:json.loads((D/p).read_text())
pt=read('evidence/plant_timing.json')['table'];pw=read('evidence/plant_widths.json')['cases'];nn=read('evidence/nncs_timing_audit.json')['rows'];ca=read('evidence/cartpole_audit.json');cr=read('evidence/cartpole_campaign.json')['rows'];pc=read('configs/plant_contracts.json')
impl=['ours','huan','xiangru','native'];labels={'ours':'我们','huan':'Huan','xiangru':'Xiangru','native':'Flow*'}
armnames={'ours_strict':'我们 strict v2','huan_strict':'Huan strict','xiangru_strict':'Xiangru strict','native':'Flow*','huan_parity':'Huan parity','xiangru_parity':'Xiangru parity'}
plants=['van_der_pol','brusselator'];pnames={'van_der_pol':'Van der Pol','brusselator':'Brusselator'}
cases=[(p,b) for p in plants for b in [1,32]]
def tr(p,b,i,mode='strict'):return next(r for r in pt if r['plant']==p and r['batch']==b and r['implementation']==i and (i=='native' or r['mode']==mode))
def wr(p,b,i,mode='strict'):return next(r for r in pw if r['plant']==p and r['batch']==b and r['backend']==i and r['mode']==mode)
def num(x):return f'{x:.6f}'
def wratio(r):return num(r['worst_channel_p95'])+' / '+num(r['worst_channel_max'])
def export_csv(name,head,rows):
 with (D/'tables'/name).open('w',newline='') as f:w=csv.writer(f);w.writerow(head);w.writerows(rows)
core_rows=[[pnames[p]+f' B{b}']+[num(tr(p,b,i)['core_median']) for i in impl] for p,b in cases]
width_rows=[[pnames[p]+f' B{b}']+[wratio(wr(p,b,i)) for i in impl[:3]] for p,b in cases]
process_rows=[[pnames[p]+f' B{b}']+[num(tr(p,b,i)['process_median']) for i in impl] for p,b in cases]
export_csv('plant_core_medians.csv',['task']+impl,core_rows)
export_csv('plant_width_summary.csv',['task']+impl[:3],width_rows)
# Recompute every displayed plant median from the 120 original samples.
raw=read('evidence/plant_raw_timing.json');assert len(raw)==120 and all(r['eligible'] and r['completed'] for r in raw)
for row in pt:
 s=[r for r in raw if all(r['row'][k]==row[k] for k in ['plant','batch','implementation','mode'])]
 assert len(s)==5 and statistics.median(r['core_seconds'] for r in s)==row['core_median']
 assert statistics.median(r['process_wall_seconds'] for r in s)==row['process_median']
# Recompute all 80 channel p95/max values from the delivered full width CSV.
with gzip.open(D/'tables/plant_widths.csv.gz','rt') as f:width_data=list(csv.DictReader(f))
assert len(width_data)==65600
for r in pw:
 for ch,summary in r['channels'].items():
  view,coord=ch.split('_');a=[float(x['width_ratio']) for x in width_data if x['backend']==r['backend'] and x['mode']==r['mode'] and x['plant']==r['plant'] and int(x['batch'])==r['batch'] and x['view']==view and x['coordinate']==coord]
  assert len(a)==summary['rows'] and max(a)==summary['max']
  assert abs(float(np.quantile(a,.95))-summary['p95'])<1e-12
records=read('evidence/nncs_timing_campaign.json')['records']
assert len(records)==72 and sum(bool(r['include']) for r in records)==60
for r in nn:
 chosen=[x['process_wall_s'] for x in records if x['include'] and x['case']==r['case'] and x['arm']==r['arm']]
 assert len(chosen)==5 and statistics.median(chosen)==r['process_median_s']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42})
colors=['#147D92','#8193A8','#C5A56A','#293D56']
fig,axs=plt.subplots(1,2,figsize=(11.5,4.1))
for ax,b in zip(axs,[1,32]):
 x=np.arange(2);w=.18
 for j,i in enumerate(impl):
  vals=[tr(p,b,i)['core_median'] for p in plants];bars=ax.bar(x+(j-1.5)*w,vals,w,color=colors[j],label={'ours':'Ours strict', 'huan':'Huan strict','xiangru':'Xiangru strict','native':'Flow*'}[i])
  ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
 ax.set_xticks(x,[pnames[p] for p in plants]);ax.set_ylabel('Core solve time (s)');ax.set_title(f'B{b} × '+('1000 steps' if b==1 else '20 steps'));ax.set_ylim(0,ax.get_ylim()[1]*1.17);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False);fig.tight_layout(rect=[0,.12,1,1]);fig.savefig(D/'figures/plant_core.pdf');plt.close(fig)
for plant in plants:
 fig,axs=plt.subplots(2,2,figsize=(11.5,5.7),sharex=True)
 for ax,(view,coord) in zip(axs.flat,[('endpoint','x'),('endpoint','y'),('tube','x'),('tube','y')]):
  for i,col,label in [('ours',colors[0],'Ours strict'),('huan',colors[1],'Huan = Xiangru strict')]:
   vals=[r for r in width_data if r['plant']==plant and r['batch']=='1' and r['backend']==i and r['mode']=='strict' and r['view']==view and r['coordinate']==coord]
   ax.plot([float(r['time']) for r in vals],[float(r['width_ratio']) for r in vals],label=label,color=col,lw=1.4)
  ax.axhline(1.25,color='#B24B35',ls='--',lw=.8,label='Max target 1.25');ax.axhline(1,color='#293D56',lw=.7);ax.set_title(f'{view} / {coord}',fontsize=11);ax.grid(alpha=.2);ax.set_ylabel('Width / Flow*')
 for ax in axs[1]:ax.set_xlabel('Time (s)')
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(D/f'figures/{plant}_widths.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11.5,4.1))
for ax,case,title in zip(axs,['tora','single_pendulum'],['TORA: B12 × 200','Single Pendulum: B1 × 100']):
 rows=[next(r for r in nn if r['case']==case and r['arm']==a) for a in ['ours_strict','huan_strict','xiangru_strict','native']]
 bars=ax.bar(range(4),[r['process_median_s'] for r in rows],color=colors)
 for bar,r in zip(bars,rows):
  if not r['complete']:bar.set_hatch('///');bar.set_alpha(.65)
  ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.12,f"{r['process_median_s']:.2f}"+(' *' if not r['complete'] else ''),ha='center',fontsize=10)
 ax.set_xticks(range(4),['Ours v2','Huan','Xiangru','Flow*']);ax.set_ylabel('Whole runner process (s)');ax.set_title(title);ax.set_ylim(0,max(r['process_median_s'] for r in rows)*1.22);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.text(.5,.015,'* Incomplete TORA runs: 2357/2400 accepted lane-steps; not a successful speed comparison.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(D/'figures/nncs_time.pdf');plt.close(fig)
# Derive time-dependent full NNCS width curves directly from the available models' ranges.
fig,axs=plt.subplots(1,2,figsize=(11.5,3.8))
for ax,case,title in zip(axs,['tora','single_pendulum'],['TORA (all 12 partitions)','Single Pendulum']):
 with gzip.open(D/f'tables/{case}_linear_v2.csv.gz','rt') as f:rows=list(csv.DictReader(f))
 vals={}
 for r in rows:
  if r['accepted']=='1':vals.setdefault(int(r['step']),[]).append(float(r['ratio']))
 ax.plot(sorted(vals),[max(vals[s]) for s in sorted(vals)],color=colors[0],lw=1.7,label='Ours v2: max across channels/lanes')
 ax.axhline(1,color=colors[3],lw=1,label='Flow* reference');ax.set_title(title);ax.set_xlabel('ODE step');ax.set_ylabel('Width / Flow*');ax.grid(alpha=.2)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False);fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(D/'figures/nncs_width.pdf');plt.close(fig)
# Structured prose is emitted both as readable Markdown and as a typeset report.
content=[]
def h(t):content.append(('h',t))
def sub(t):content.append(('sub',t))
def para(t):content.append(('p',t))
def table(head,rows,widths=None):content.append(('table',(head,rows,widths)))
def figure(file,caption):content.append(('figure',(file,caption)))
h('1. 当前结论')
para('目标是把验证连续系统的完整可达域计算迁移到 PyTorch/CUDA，既减少耗时，又保持范围紧度及数值误差覆盖。当前已经实现可运行的完整 GPU 引擎、同任务多方对照和全轨迹范围重算；总目标尚未完成。本次按用户要求整理后暂停，不继续启动优化实验。')
table(['问题','已有结果','仍需解决'],[
['固定多项式任务是否更快','四项任务核心时间均优于 Huan/Xiangru strict；三项相对 Flow* 达到 ≤1.20 倍目标','原式 VDP B1 仍为 Flow* 的约 2.015 倍'],
['范围是否接近 Flow*','Brusselator 和 B32 短轨迹接近；原式 VDP 已明显优于作者 strict','VDP B1 p95/max=1.123238/1.271071，略超 1.10/1.25'],
['作者闭环控制实验是否复现','TORA、单摆完成六组相同配置实验；CartPole 六组完整尝试','TORA v2 全成功但偏宽；CartPole 六组都未完成全程'],
['是否有实际加速','TORA 完整进程相对 Flow* 配对速度比约 1.802；完整范围输出减少 15.3%–43.6%','单摆完整进程比 Flow* 慢约 5.5%；输出仍有明显差距'],
['能否宣称完整严格验证器','已保留 plant 的严格误差路径、SR 和拒绝回滚检查','控制器舍入误差资格、候选统一入口与整体验收未完成']],[.19,.40,.41])
para('最重要的区别：跑完不等于验证为安全，宽度接近不等于证明正确，核心求解快不等于包含启动和输出的全过程快。后文给出具体数字和相应计量边界。')
h('2. 项目在解决什么问题')
para('给定微分方程、一个初始状态盒和时间范围，程序要包住从盒内任意状态出发的所有可能轨迹，而非只模拟一条轨迹。状态范围越宽，下游安全判断就越容易变成 Unknown；因此速度与范围质量必须同时衡量。')
para('Taylor model（泰勒模型）把状态表示为多项式加区间余项。多项式保留变量之间的依赖，余项承担截断及浮点误差。程序逐步组合、积分并验证候选包络；成功才提交新状态。SR（symbolic remainder，符号余项）保留跨步误差关系，减轻反复装箱造成的范围膨胀；历史队列满时按原规则重置。')
para('原路线中，GPU 局部算子很快，但大量 CPU 组织、传输和逐小算子调度仍限制完整任务。本阶段改为沿完整 GPU 状态连续推进，并优化调度和范围观察。这里的 PyTorch 加速包含必要的定向舍入 CUDA 扩展，不是禁止一切扩展的“纯 PyTorch”改写。')
table(['模块','主要工作','如何体现进展'],[
['完整推进引擎','稀疏多项式、Horner 组合、CUDA 图、验证和 SR 状态处理','以完整1000步及真实32分区任务衡量，不用局部算子倍率代替'],
['范围输出','从完整 Taylor model 重建物理范围；定向舍入；原生 CPU 多项式乘法','输出工作量相同的五次配对，完整模型保持一致'],
['闭环 NNCS','CROWN 控制器每个周期提供控制边界，再推进连续 plant','复用相同作者配置与模型，核对完成状态和全轨迹宽度'],
['复现与证据','固定版本、配置、实际库、完整日志、失败行与审计','同机同任务比较，并保存可重算的CSV/JSON']],[.20,.39,.41])
h('3. 三个仓库、四类实现')
para('主仓库 https://github.com/lsnnnnnnnn/torch_tm_flowpipe 负责已有工程、适配入口与实验。Huan 仓库 https://github.com/huanzhang12/flowstar-gpu 提供被复用和对照的 GPU 引擎。Xiangru 仓库 https://github.com/xiangruzh/CROWN-Reach_Development 提供 NNCS 集成、控制器及作者 benchmark。Native Flow* 是 C++ 可达域对照，不是第四个要重写的项目，也不是无条件数学真值。')
table(['材料中的名称','固定版本','用途'],[
['我们 plant 正式比较版','engine f53696c8；adapter 4beba860','四方固定原式多项式对照'],
['我们 NNCS 线性路径 v2','engine fff9d0f5','TORA/单摆/CartPole 新候选；尚未统一推广'],
['GPU 乘法候选','engine d8bdc4d2','独立性能候选，与 fff9 分属不同实验线'],
['范围输出候选','observer 1bc9b40f','CPU 原生乘法后端；输出语义不变'],
['Huan 原版','d5f0b68f','保持源码原样；strict 和 parity 分开'],
['Xiangru 原版','1c16d4ef，2026_experiment','保持源码原样；共享作者 NNCS 驱动'],
['匹配 Flow*','722a5611 / 已冻结匹配构建','固定 plant 与匹配 NNCS 对照；早期 registry 使用另一历史构建，单列']],[.27,.35,.38])
para('“我们”不是从零独立发明的全新算法：它建立在已有工程、Huan 引擎和保存的严格修复之上。本报告比较的是具体版本的完整实现。Huan/Xiangru 的 strict 标签与修复版 strict 的误差覆盖范围不完全相同；parity 是作者的另一数值模式，仅作复现参考。')
para('NNCS 表中的 Huan 与 Xiangru 分别表示各自原引擎接入同一个 Xiangru 控制器驱动，不是两套独立 NNCS 产品。所有模型和驱动版本均固定；不声称这是各仓库当前最新提交的性能。')
h('4. 实验口径：哪些数可以直接比较')
table(['指标','定义与方向','边界'],[
['核心求解时间','完整求解秒数，越小越快','plant 包含初始化/图捕获/推进/SR/同步；排除进程启动、扩展预载、记录和范围观察'],
['完整进程时间','启动 runner 到退出，越小越快','包含 Python/库/控制器启动和最终判定；time-only 实验关闭轨迹记录；不含独立 outward 导出'],
['独立完整输出时间','展开多项式、求范围、gzip模型及CSV写出','不含求解；不能与另一场实验中位数相加冒充实测全过程'],
['宽度比','(上界−下界) / 对应 Flow* 宽度','1 表示相同；大于1更宽；需同时看成功范围与绝对边界差'],
['p95 / max','先逐通道计算，再报告最差通道','plant 通道为 endpoint-x/y 与 tube-x/y；全部有效步/分区参与，无删最差步'],
['完成率','接受的分区步 / 请求分区步','B12×200 表示2400分区步；部分失败的耗时不作为成功加速分母']],[.22,.38,.40])
para('Endpoint 是每一步末端的范围；tube 是该步整个局部时间区间的范围。两者分别给出，不能用较紧的 endpoint 冒充整段包络。当前固定 plant 对照使用完整匹配的1000/20步和各步 [0,h]，不沿用早期 native 最后一步截短的观察域。')
para('固定四方 plant 实验：4任务×6实现/模式×5次，共120个有效 time-only 样本。完整宽度另来自20条 GPU 记录任务，统一观察后65,600行/80通道本地复核。NNCS 正式计时：2任务×6组×(1次排除预跑+5次计入)，共72次调用，60次计入。CartPole 只有六组单次带记录尝试，不能称五次正式计时。')
para('硬件为同一 huan-c4140-server-3，Intel Xeon Gold 6138、Tesla V100-SXM2 16GB；固定 plant 和 NNCS 主比较均使用单 CPU 核6、PyTorch/OMP/MKL/OpenBLAS单线程。PyTorch 2.5.1+cu121，扩展构建工具链 GCC13/CUDA12.6，实际加载库按SHA锁定。固定四方计时GPU2，后续NNCS及GPU乘法候选GPU3；输出CPU五配对固定核10。扩展缓存已经存在，不宣称首次编译冷启动。')
h('5. 固定多项式：详细配置')
table(['参数','Van der Pol','Brusselator'],[
['方程',"x'=y；y'=y−x−x²y", "x'=1−4x+x²y；y'=3x−x²y"],
['初始原盒','x∈[1.1,1.4]；y∈[2.35,2.45]','x∈[1.48,1.52]；y∈[2.98,3.02]'],
['阶数 / 固定步长','4 / 0.01','6 / 0.02'],['B1完整任务','1000步，T=10','1000步，T=20'],['B32吞吐任务','8×4真实分区，每盒20步，T=0.2','8×4真实分区，每盒20步，T=0.4'],['cutoff / 余项半径','1e−10 / 1e−4','1e−10 / 1e−4'],['SR队列容量','100','1000'],['验证迭代上限 / stop ratio','491 / 0.99','491 / 0.99']],[.26,.37,.37])
para('初盒由精确十进制目标向外转成 binary64 端点；每个真实分区和实际十六进制编码均在 configs/plant_contracts.json。B32 不是复制同一个盒32次。我们使用 Horner/graph/structural/defer_polynomial；Huan/Xiangru 保留原版 monomial/eager 路径。相同物理任务及参数不意味着内部计算算法完全相同。')
para('原式 VDP 的字符串固定为 y−x−x*x*y。部署试验曾用代数等价的 (1−x*x)*y−x；有限精度和区间计算的操作顺序不同，其宽度不能替代原式验收。本报告主比较全部使用原式。')
h('6. 固定多项式：时间比较')
table(['任务','我们 strict','Huan strict','Xiangru strict','Flow*'],core_rows,[.25,.1875,.1875,.1875,.1875])
para('核心求解时间，单位秒，五次中位数。来源 S1。完整逐次数据位于 evidence/plant_raw_timing.json；本报告构建时重新计算了所有24组中位数。')
figure('plant_core.pdf','图1：四项固定任务的完整核心求解；柱顶为秒数，各子图采用不同纵轴范围。')
table(['任务','我们 / Flow* 配对比中位','≤1.20时间门槛'],[[pnames[p]+f' B{b}',num(tr(p,b,'ours')['paired_ratio_median']),'达到' if tr(p,b,'ours')['paired_ratio_median']<=1.2 else '未达到'] for p,b in cases],[.35,.4,.25])
para('Brusselator B1 和 B32 对 Flow* 有明显核心优势；VDP B32 已接近 Flow*；VDP B1 仍约慢一倍。相对 Huan/Xiangru 的 B1 加速是真实完整实现差异，但不能全部归因于“GPU”：三个被比较的引擎本来都使用 GPU，我们同时改变了表达式组织和执行调度。')
sub('把启动也算进去')
table(['任务','我们 strict','Huan strict','Xiangru strict','Flow*'],process_rows,[.25,.1875,.1875,.1875,.1875])
para('Time-only 完整进程中位秒。短 B32 任务受 Python/库启动影响很大，核心略快不保证进程更快。该列仍不包含独立范围输出。')
h('7. 固定多项式：全轨迹宽度比较')
table(['任务','我们 strict','Huan strict','Xiangru strict'],width_rows,[.25,.25,.25,.25])
para('每格为最差通道 p95 / max，分母是匹配 Flow*，Flow* 自身为1/1。来源 S2，全部范围按同一 outward 观察器计算；完整80通道数据及绝对边界差见附录。')
para('我们原式 VDP B1 的 p95=1.123238、max=1.271071，仍略超过1.10/1.25；对应最差最大值出现在 endpoint-y 第982步。其余三个固定任务的最差通道均达到宽度门槛。Huan/Xiangru 在短 B32 上略紧，但长期 B1 尤其 Brusselator 的范围明显更宽。')
figure('van_der_pol_widths.pdf','图2：原式 Van der Pol B1 全1000步，endpoint/tube及两个坐标分别显示。Huan与Xiangru strict曲线逐行一致后合并。')
figure('brusselator_widths.pdf','图3：Brusselator B1 全1000步。宽度优势必须与误差覆盖依据一起解读，不能独立当作正确性证明。')
h('8. 作者 NNCS：配置与控制器')
para('NNCS 是神经网络控制的连续系统。每个控制周期先对当前状态集合计算神经网络控制界，再在固定控制条件下推进若干 ODE 小步；然后重复。这里对齐的不只是动力学，还包括初盒分区、周期、控制器文件、输入布局、传输精度和判定时域。')
table(['参数','TORA','Single Pendulum','CartPole'],[
['物理维度 / 总状态','4 / 6（含t,u）','2 / 4（含t,u）','4 / 6（含t,u）'],['B / 每盒ODE步数','12 / 200','1 / 100','1 / 200'],['阶数 / ODE步长','3 / 0.1','2 / 0.01','6 / 0.005'],['控制周期 / 周期数','1.0 / 20','0.05 / 20','0.02 / 50'],['总时长','20','1','1'],['cutoff','1e−6','1e−6','1e−6'],['余项猜测区间','[−0.01,0.01]','[−0.01,0.01]','[−0.1,0.1]'],['空间分区','x1分4份、x2分3份','不分区','没有split_vars，实际B1'],['SR实际队列容量','1000（驱动固定值）','1000（驱动固定值）','1000（驱动固定值）'],['输入shape','[-1,1,1,4]','[-1,1,2]','[-1,4]']],[.28,.24,.24,.24])
para('三个共同实验均用相同 ONNX 模型、box CROWN、same-slope relaxation、flat 输入映射和 rpc-float32 传输合同；输出 scale=1、offset=0。GPU 使用共享 Xiangru 驱动，保留我们修复后的 endpoint 算术；native 使用既有 RPC 服务与固定步数/共同 binary64 初盒支路。原 native 每周期尾步浮点截短的原始支路单独保留，不与固定步数支路混用。')
table(['任务','初始物理状态','动力学'],[
['TORA','x1=[0.6,0.7]；x2=[−0.7,−0.6]；x3=[−0.4,−0.3]；x4=[0.5,0.6]',"x1'=x2；x2'=−x1+0.1sin(x3)；x3'=x4；x4'=u−10"],
['单摆','x1=[1,1.175]；x2=[0,0.2]',"x1'=x2；x2'=2sin(x1)+8u"],
['CartPole','x1=[−0.0375,−0.03125]；x2=[−0.015625,−0.0125]；x3=[−0.00625,0]；x4=[−0.007375,−0.00625]',"x1'=x2；x2'=2u；x3'=x4；x4'=(0.08·0.41·(9.8sin(x3)−2u cos(x3))−0.0021x4)/0.0105"]],[.15,.39,.46])
para('所有 NNCS 的附加状态 t、u 初值为0；周期内 t′=1、u′=0。原始 YAML 与精确执行版本 *_executed.yaml 同时保留。完整 YAML 保留安全/不安全/目标约束、模型路径及形状等字段。TORA 使用 controllerTora.onnx；单摆使用 controller_single_pendulum.onnx；CartPole 使用 model.onnx。SHA 身份记录随 evidence/ 中的 *_identity.json、campaign 与 contract 交付，权重文件本身保留服务器。')
para('CartPole 选的是作者 registry 的历史小初盒 full profile。它与4096分区 profile、大初盒10秒 ARCH-COMP profile不同。原 src/configs/cartpole.yaml 的 ODE步长0.05大于控制周期0.02，是另一份无效配置，没有据此猜测修参数。本次有效配置为 registry 原值，实际运行与本报告一致。')
h('9. TORA 与单摆：相同实验的时间和宽度')
for case,title in [('tora','TORA：B12×200'),('single_pendulum','Single Pendulum：B1×100')]:
 sub(title)
 rows=[]
 for r in nn:
  if r['case']==case:rows.append([armnames[r['arm']],f"{r['process_median_s']:.6f}",f"{r['process_min_s']:.3f}–{r['process_max_s']:.3f}",f"{r['accepted']}/{r['requested']}",'未完整' if not r['complete'] else f"{r['width_p95']:.6f} / {r['width_max']:.6f}"])
 table(['实现','进程中位秒','五次最小–最大','成功分区步','全程p95 / max'],rows,[.22,.16,.19,.16,.27])
para('来源 S3。时间为五次完整进程调用，不是作者内部打印的 plant 计时。宽度来自同配置的独立完整数值记录；无记录计时样本检查身份、完成状态及完整GPU任务最终 HULL 的12位有效数字一致，未对每次计时的全部中间状态重采样。')
figure('nncs_time.pdf','图4：六组计时表中的 strict 和 native 主比较。TORA 的作者 strict 未完整，斜线柱仅展示耗时，不计算成功加速比。')
para('TORA：新 v2 全2400分区步成功，五次逐轮 native/我们 进程时间比中位为1.802031。Huan/Xiangru strict 为2357/2400，不能把其不完整时间作为成功速度基准。Parity 全程成功且范围几乎与 Flow* 一致，但不能拿它替代 strict 的保证。')
para('单摆：所有实现都完成100步。我们宽度p95/max为1.036731/1.040936，优于原版 strict 的1.067393/1.075450。完整进程我们5.266秒、Flow*4.987秒，逐轮native/我们为0.947724，换算我们约慢5.5%。内部计时看起来更快，正说明计时边界不能混用。')
figure('nncs_width.pdf','图5：我们 v2 全轨迹每一步的最坏通道/分区宽度比。此图的逐时刻最大曲线与“先逐通道p95再取最坏值”不同，表中的统计仍按正式定义。')
sub('主要数值改进：TORA 从部分失败到全部成功')
table(['阶段','成功分区步','解释'],[['初始严格版','2354/2400','首个失败出现在第189步'],['直接变量v1','2393/2400','首个失败推迟到第197步'],['线性路径v2','2400/2400','仅沿加/减/取负到输出的路径保留完整多项式，再由原验证积分承担误差']],[.25,.23,.52])
para('与初始版本共同前188步比较，最差 p95/max 从2.555528/8.854653改善到1.383703/1.884253；不是拿旧部分轨迹分布与新完整轨迹直接比较。新v2全程仍为1.592229/2.710606，后段范围偏宽尚未解决。所有非线性消费者仍阻止这一线性保留策略，未通过全局清零余项取得表面紧度。')
h('10. CartPole：六组都未完成全程')
rows=[]
for r in cr:
 a=r['arm'];w=next((x for x in ca['comparisons'] if x['arm']==a),None)
 rows.append([armnames[a],f"{r['process_wall_s']:.3f}",'141/200' if a=='native' else '156/200','1 / 1' if a=='native' else f"{w['p95']:.9f} / {w['maximum']:.9f}"])
table(['实现','单次带记录进程秒','成功步','共同前141步p95 / max'],rows,[.24,.22,.17,.37])
para('来源 S4。Native 保存141步成功模型后提前终止；五个GPU实现均接受156步，第157步明确拒绝，随后43步未尝试。只有共同前141步能计算相对 native 的宽度比；GPU第142–156步有范围但无native分母，比例留空。')
para('共同前缀范围非常接近，但这既不表示200步全程通过，也不能解释两类实现拒绝时刻为何不同。当前只确认输出与停止位置，尚未给出拒绝差异的完整因果分析。完成步数、记录方式和计时边界不同，不对上表计算加速比。9,600请求行、7,368可用物理范围和12项完整Fraction检查已经本地复核。')
h('11. 实际加速进展与代价')
para('大方向上的加速包括：让完整状态驻留GPU；减少稀疏组合的重复组织和小kernel调度；将固定流程捕获为CUDA图；复用SR准备工作；加快向外舍入的范围展开。下面两项有独立的旧/新五配对证据，不能与更早作者实验的单次时间混成新配对。')
output_rows=[['VDP B1','7.266908','5.688225','21.8%'],['VDP B32','2.744789','2.324720','15.3%'],['Brusselator B1','74.380369','41.760145','43.6%'],['Brusselator B32','10.311151','7.933436','23.1%']]
table(['完整输出任务','旧中位秒','新中位秒','配对耗时下降'],output_rows,[.31,.23,.23,.23])
para('来源 S5，40个输出样本、20对完整模型在服务器一致，131,200行范围与冻结四方表本地对照。输出后端仍包含CPU工作：这项提升来自将多项式乘法放入原生CPU实现，保留操作/舍入顺序；不能称为纯GPU加速。Brusselator输出从约74秒降至约42秒，但原版单次约10.6秒，差距仍大。')
table(['GPU候选试验','f536旧中位秒','d8新中位秒','配对变化'],[['VDP B1×1000，regrouped RHS','2.831174','2.733734','下降3.30%']],[.4,.2,.2,.2])
para('该GPU试验显式使用第九个乘法误差融合库，五对逐对比中位0.966981；完整模型逐位资格保持。它没有新跑native，因此不能借用历史native时间声称新的正式Flow*比；其regrouped RHS也不能替代本报告原式主对照。')
para('暂停前最后一次诊断已经完成：f536与d8在原式VDP上各做完整计时/trace诊断及阶段测量，六次完整1000步数值记录均与冻结原式一致。最后捕获的compose图单独重放中位约541→453微秒，kernel数288→252；normalize约238微秒，validpost约229微秒。该图测量排除了输入拷贝，也不是完整事务或五次正式任务计时，仅说明继续减少组合调度仍是可检验方向。')
h('12. 更广 benchmark 覆盖到哪里')
para('已经盘点Huan的12个原生plant registry任务、Xiangru的14个NNCS配置及一个单列的native QUAD合同。27个合同不等于27个任务都完成四方计时和宽度；目前完整多方主比较集中在四个固定plant任务、TORA、单摆和CartPole。')
para('12个native registry plant的原时域尝试中11个完成，Robertson在原h=0.1/order3/queue0配置第一步拒绝；早期严格CPU短前缀10/12完成，Quadrotor超时。它们是可运行性筛查，使用不同历史版本与计时边界，不并入本报告GPU速度排名。')
full=read('evidence/nncs_full_coverage.json')['rows'];coverage=[]
for r in full:
 status='完整接受' if r.get('configured_horizon_completed') else ('Unsafe协议提前停止' if r['id']=='double_pendulum_more_robust' else '数值拒绝')
 coverage.append([r['id'],f"{r.get('B')}×{r.get('full_ODE_steps')}",str(r.get('accepted_lane_steps')),status])
table(['早期NNCS全时域筛查','请求B×步','接受分区步','该次运行状态'],coverage,[.40,.15,.18,.27])
para('这是9d83162d历史版本、native-f64控制器传输的独立筛查；12项中9项完整接受。不能与后续rpc-float32多方实验直接位比较；TORA后续v2已经补到完整成功。DP-more的66/80是Unsafe协议提前停止，不是程序属性异常。Airplane有规模/内存限制，原QUAD YAML参数无效，另立的native QUAD合同不能代替原配置。')
h('13. 数值保证、尚未达标项与暂停点')
para('本阶段保留了strict/outward路径、完整误差余项、SR历史及原子事务：验证失败不能把未接受状态提交给下一步。采用源码身份、实际库身份、精确有理数Fraction检查、CPU/CUDA路径、跨SR重置和混合失败回滚测试共同限定可用范围。完整模型一致和范围重算是证据的一部分，不是全程序形式证明。')
para('完整NNCS仍有明确缺口：CROWN及控制器仿射结果采用round-to-nearest计算/注入，尚未完成全过程向外舍入误差资格。因此“strict”在这些NNCS实验中首先描述plant设置；不能称已获得端到端严格控制系统证书。')
table(['总目标要求','当前证据','状态'],[
['四固定任务核心≤1.20×native','VDP B1约2.015；其余三项达到','未全部达到'],['每通道p95≤1.10、max≤1.25','原式VDP B1为1.123238/1.271071；其他三项达到','未全部达到'],['实测改善完整范围输出','四任务输出五配对改善15.3%–43.6%','已取得阶段证据'],['五次匹配配对和完整宽度','固定plant已完成；TORA/单摆另完成；CartPole不完整','按任务分别成立'],['统一入口与可复现交付','plant基线入口已在adapter；fff9/d8/输出候选仍独立','候选尚未全面集成'],['作者benchmark和严格NNCS','已经扩展并如实保存失败；控制器资格未完成','未完成']],[.34,.44,.22])
para('用户要求完成本次文档、幻灯片及新分支推送后暂停。暂停时不再开启新优化或benchmark。未来恢复时优先处理原式VDP核心成本与最差步宽度、TORA末段范围及控制器误差资格，再做新的五次同任务验收；这些是待执行方向，不是本次已经实现的成果。')
h('附录A. endpoint/tube四通道具体宽度')
para('以下保留原式固定任务的全部strict通道统计。max绝对边界差是上下边界差绝对值的较大者，单位随状态坐标；零native宽度计数在原始JSON中保留。完整CSV另含所有parity数据。')
for p,b in cases:
 sub(pnames[p]+f' B{b}')
 rows=[]
 for i in impl[:3]:
  for ch,s in wr(p,b,i)['channels'].items():rows.append([labels[i],ch,num(s['p95']),num(s['max']),f"{s['max_abs_boundary_delta']:.6g}"])
 table(['实现','通道','p95','max','最大绝对边界差'],rows,[.15,.24,.19,.19,.23])
sub('B1 最后一步的实际宽度（并非全程最大值）')
para('下面给出状态坐标单位下的实际宽度，而不仅是相对比例。两任务均为第1000步、lane0；Flow*列是同一步同一观察域的参考。此表仅帮助理解量级，全程质量仍以上述p95/max及曲线为准。')
absolute_rows=[]
for p in plants:
 for view,coord in [('endpoint','x'),('endpoint','y'),('tube','x'),('tube','y')]:
  selected=[next(x for x in width_data if x['plant']==p and x['batch']=='1' and x['step']=='1000' and x['lane']=='0' and x['backend']==i and x['mode']=='strict' and x['view']==view and x['coordinate']==coord) for i in impl[:3]]
  absolute_rows.append([pnames[p],view+'-'+coord]+[f"{float(x['candidate_width']):.8g}" for x in selected]+[f"{float(selected[0]['native_width']):.8g}"])
table(['任务','通道','我们宽度','Huan宽度','Xiangru宽度','Flow*宽度'],absolute_rows,[.20,.16,.16,.16,.16,.16])
export_csv('last_step_absolute_width.csv',['plant','channel','ours_width','huan_width','xiangru_width','native_width'],absolute_rows)
h('附录B. 作者parity原式plant参考')
table(['任务','Huan秒','Xiangru秒','共同p95 / max'],[[pnames[p]+f' B{b}',num(tr(p,b,'huan','parity')['core_median']),num(tr(p,b,'xiangru','parity')['core_median']),wratio(wr(p,b,'huan','parity'))] for p,b in cases],[.29,.20,.20,.31])
para('Parity与strict分别保留，不能以本表的紧度替代strict误差保证。原始全80通道JSON同时保留endpoint/tube及绝对边界差。')
h('附录C. 证据与复现索引')
table(['编号','本包中的证据','覆盖范围'],[
['S1','evidence/plant_timing.json；plant_raw_timing.json','120个原始时间样本与24组统计'],['S2','evidence/plant_widths.json；tables/plant_widths.csv.gz','65,600行、80通道；全部边界、比例、差值'],['S3','evidence/nncs_timing_audit.json；nncs_timing_campaign.json','72次调用；60次计入；完成状态和宽度'],['S4','evidence/cartpole_audit.json；cartpole_campaign.json；tables/cartpole_*.csv.gz','六组不完整尝试；完整缺失/拒绝行；共同前缀'],['S5','evidence/output_pairs.json；gpu_product_pairs.json；product_audit.json','40次输出计时、10次GPU计时；scope保留'],['S6','configs/*.yaml；configs/plant_contracts.json','ODE、初盒、模型、分区、周期、库与源码身份'],['S7','evidence/linear_leaf_audit.json；tables/*linear_v2.csv.gz','TORA/单摆完整范围与v2资格'],['S8','evidence/nncs_full_coverage.json；native_registry_full.json','更广任务覆盖；不能冒充主比较']],[.08,.56,.36])
para('本报告包是可独立重画图表、重算汇总数字的轻量交付，不包含全部ONNX权重、大型模型流或已编译CUDA二进制。它们的版本/路径/哈希保留在证据中；完整数值资格的原始归档仍在本项目服务器结果目录。报告中的数字来自上述固定证据，不依赖访问私人对话链接。')
para('重新生成：在本目录运行 python build_materials.py（需要NumPy和Matplotlib），随后运行 sh build.sh（需要XeLaTeX/TeX Live、ctex和Fandol字体）。无需服务器即可重新编译 report.pdf 和 slides.pdf；现成PDF也已随包提供。')
para('实验源目录统一位于 /srv/local/shengenli/flowstar_acceleration_20260921T153643Z/runs。关键run为 four_way_records_v1、four_way_paired_v1、four_way_widths_v1、author_nncs_timing_v4、linear_leaf_validation_v2、cartpole_matched_v1。已有source identities和命令保留在evidence；若换机器执行，应先建立相同依赖和模型路径，不直接照抄本机绝对路径。')
# TeX/Markdown rendering helpers.
def esc(s):
 s=str(s);return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_\allowbreak{}', '/':r'/\allowbreak{}', '≤':r'$\leq$', '≥':r'$\geq$', '∈':r'$\in$', '′':r'$\prime$','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in s)
def tex_table(head,rows,widths=None,slide=False):
 n=len(head);widths=widths or [1/n]*n
 env='tabular'
 spec=''.join('>{\\raggedright\\arraybackslash}p{'+f'{w*.92:.3f}\\linewidth'+'}' for w in widths)
 out=[r'{\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.25}',r'\begin{'+env+'}{'+spec+'}',r'\toprule', ' & '.join(r'\textbf{'+esc(x)+'}' for x in head)+r' \\',r'\midrule']
 # All report tables fit a page; keep each table together.
 for row in rows:out.append(' & '.join(esc(x) for x in row)+r' \\')
 return ('\\par\\noindent\n' if not slide else '')+'\n'.join(out+[r'\bottomrule',r'\end{'+env+'}',r'}'])+('\n\\par\n' if not slide else '')
md=['# PyTorch/CUDA 可达域工程：阶段进展与实验对照','', '2026-09-23 · 面向首次接触项目的读者 · 整理后暂停','']
tex=[r'''\documentclass[UTF8,fontset=fandol,11pt,a4paper]{ctexart}
\usepackage[margin=22mm]{geometry}
\usepackage{booktabs,longtable,array,graphicx,xcolor,hyperref,amsmath,needspace}
\definecolor{navy}{HTML}{20364C}\definecolor{teal}{HTML}{147D92}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal}
\setlength{\parskip}{5pt}\setlength{\parindent}{0pt}
\emergencystretch=3em
\title{\color{navy}PyTorch/CUDA 可达域工程\\阶段进展与实验对照}
\author{项目阶段整理}\date{2026年9月23日}
\begin{document}\pagestyle{plain}\begin{titlepage}\maketitle\thispagestyle{empty}
\begin{center}面向首次接触项目的读者；资料整理与分支推送后暂停\end{center}
\end{titlepage}
{\setlength{\parskip}{0pt}\tableofcontents}\clearpage
''']
for typ,val in content:
 if typ in ['h','sub']:
  md += [('## ' if typ=='h' else '### ')+val,''];tex += [('\\Needspace{6\\baselineskip}\\section*' if typ=='h' else '\\Needspace{12\\baselineskip}\\subsection*')+'{'+esc(val)+'}']
  if typ=='h':tex += ['\\addcontentsline{toc}{section}{'+esc(val)+'}']
 elif typ=='p':md += [val,''];tex += [esc(val)+'\n']
 elif typ=='table':
  head,rows,widths=val;md += ['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]+[''];tex += ['\\small\n'+tex_table(head,rows,widths)+'\n\\normalsize']
 elif typ=='figure':
  file,cap=val;md += [f'![{cap}](figures/{file})',''];tex += ['\\begin{center}\\begin{minipage}{\\linewidth}\\includegraphics[width=\\linewidth]{figures/'+file+'}\\par\\small '+esc(cap)+'\\end{minipage}\\end{center}']
(D/'REPORT.md').write_text('\n'.join(md)+'\n');(D/'report.tex').write_text('\n'.join(tex)+r'\end{document}'+'\n')
(D/'REPORT_BUILD_CHECK.json').write_text(json.dumps({'status':'passed','plant_original_samples_recomputed':120,'plant_timing_groups':24,'plant_width_rows':len(width_data),'plant_width_channels_recomputed':80,'scope':'Numerical report data and figure inputs; PDF layout checked separately.'},indent=2)+'\n')
print('report and figures generated')
