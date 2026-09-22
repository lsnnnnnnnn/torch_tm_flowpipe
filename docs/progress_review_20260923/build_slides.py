"""Editable native Beamer slides; figures and tables come from build_materials.py."""
from build_materials import *
slides=[];notes=[]
def frame(title,body,note):
 slides.append('\\begin{frame}{'+esc(title)+'}\n'+body+'\n\\end{frame}')
 notes.append(f'## {len(slides)}. {title}\n\n{note}\n')
def bullets(*xs):return '\\begin{itemize}\n'+'\n'.join('\\item '+esc(x) for x in xs)+'\n\\end{itemize}'
def txt(x):return esc(x)+'\\par\n'
def source(x):return '\\par\\vfill{\\fontsize{6.5}{8}\\selectfont\\color{gray}'+esc(x)+'} '
def tab(head,rows,widths=None):return '{\\fontsize{8.5}{11}\\selectfont\n'+tex_table(head,rows,widths,True)+'\n}\\par\n'
def plot(file,height='5.7cm'):return '\\begin{center}\\includegraphics[width=\\linewidth,height='+height+',keepaspectratio]{figures/'+file+'}\\end{center}'
frame('PyTorch/CUDA 可达域工程',r'''\vspace{0.5cm}{\LARGE 阶段进展与实验对照}\par\vspace{0.6cm}
{\large 做了什么，哪些任务更快，范围差距在哪里}\par\vspace{0.8cm}
我们 / Huan / Xiangru / Native Flow*\par
\vspace{0.3cm}2026年9月23日\par
\vfill\color{teal}本次资料整理和新分支推送完成后暂停工作''','开场先说明目标：把完整可达域求解迁移到GPU，而不是展示某个小算子很快。当前有明确进展，也有未达标项；本次不宣称总目标完成。')
frame('当前进展',bullets('完整 GPU 引擎已跑通：四个固定多项式任务有完整时间与宽度对照。','TORA 新候选从部分失败改善到 2400/2400 分区步成功。','范围输出取得 15.3%–43.6% 的五次配对耗时下降。','主要缺口：原式 VDP 单盒仍慢且略宽；TORA 后段明显偏宽；控制器严格误差资格未完成。')+source('证据：S1–S5、S7；各项使用不同且明确分开的实验口径。'),'先给大方向：已经不是只做局部kernel。强调三个成果属于不同测量，不要说“所有任务都已经接近Flow*”。')
frame('可达域计算的目标',r'''\begin{columns}[T]
\begin{column}{0.55\textwidth}
给定方程、初始状态集合和时间：
\begin{itemize}
\item 包住所有可能轨迹
\item 范围越紧，安全判断越有用
\item 算得快，也要保持误差覆盖
\end{itemize}
\vspace{0.3cm}普通模拟只回答“这一条轨迹怎么走”；这里计算整个集合的包络。
\end{column}
\begin{column}{0.40\textwidth}
\begin{tikzpicture}[x=0.54cm,y=0.54cm]
\draw[->,gray] (0,0)--(7.5,0) node[right]{时间};
\draw[->,gray] (0,0)--(0,5.3) node[above]{状态};
\fill[teal!15] (0,1.3) .. controls (2,1.4) and (4,2.3) .. (7,2.6) -- (7,4.6) .. controls (4,4.0) and (2,3.3) .. (0,2.7) -- cycle;
\draw[teal,thick] (0,1.3) .. controls (2,1.4) and (4,2.3) .. (7,2.6);
\draw[teal,thick] (0,2.7) .. controls (2,3.3) and (4,4.0) .. (7,4.6);
\draw[navy] (0,2) .. controls (2,2.1) and (4,3.5) .. (7,3.7);
\node[teal] at (4.3,4.9) {集合包络示意};
\end{tikzpicture}
\end{column}\end{columns}''','这张图只是概念示意，不是实验结果。用它解释为什么不能只看运行时间，也不能拿一条仿真轨迹落在范围内当作验证器正确性的证明。')
frame('完整计算链',r'''\begin{center}
\begin{tikzpicture}[node distance=0.7cm, every node/.style={align=center}]
\node[draw=teal,fill=teal!7,text width=2.6cm,minimum height=1.2cm] (a) {初始状态盒\\与系统方程};
\node[draw=teal,fill=teal!7,text width=2.8cm,minimum height=1.2cm,right=of a] (b) {Taylor model 推进\\多项式 + 区间余项};
\node[draw=teal,fill=teal!7,text width=2.6cm,minimum height=1.2cm,right=of b] (c) {验证与 SR 历史\\成功才提交};
\node[draw=teal,fill=teal!7,text width=2.3cm,minimum height=1.2cm,right=of c] (d) {物理范围\\endpoint / tube};
\draw[->,thick,teal] (a)--(b);\draw[->,thick,teal] (b)--(c);\draw[->,thick,teal] (c)--(d);
\end{tikzpicture}\end{center}
\vspace{0.4cm}
\begin{itemize}
\item Taylor model 保留状态依赖；余项承担截断及浮点误差
\item SR 保留跨步误差关系，减轻反复装箱导致的膨胀
\item 本阶段优化完整推进和范围输出，失败仍按原规则回滚
\end{itemize}''','先讲清端到端结构，再讲性能。PyTorch张量承载主要状态和计算，必要CUDA扩展负责有向舍入；这不等于无扩展的纯PyTorch重写。')
frame('三个仓库与四类实现',tab(['角色','使用内容','比较版本'],[['我们：torch_tm_flowpipe','工程、适配与严格优化','plant f536；NNCS fff9'],['Huan：flowstar-gpu','原始GPU引擎及证明背景','d5f0b68f'],['Xiangru：CROWN-Reach','集成驱动、模型、benchmark','1c16d4ef'],['Native Flow*','C++可达域参考','固定匹配构建']],[.36,.37,.27])+bullets('我们复用已有引擎和修复；主比较是具体版本的完整实现差异。','NNCS 中 Huan/Xiangru 分别接入同一个 Xiangru 驱动。')+source('详细仓库URL、完整源码/库身份：REPORT.md 第3节与 configs/plant_contracts.json。'),'不要把Huan对照讲成Huan自己的独立NNCS产品；它是原引擎接到共享驱动。fff9和d8是不同候选，不要说最新候选已经统一合并。')
frame('主要改动',tab(['方向','做了什么','希望改善什么'],[['完整GPU执行','稀疏/Horner计算、CUDA图、SR准备复用','减少完整任务调度与组织成本'],['数值表示','线性路径保留更多多项式信息，沿原积分验证承载误差','减少不必要的余项膨胀与拒绝'],['范围输出','保留向外舍入次序，原生CPU乘法加快展开','让用户真正拿到范围的过程更快'],['相同实验','固定配置、完整轨迹、五次计时、所有失败记录','让时间和宽度的比较可信']],[.19,.50,.31]),'只讲四个大方向，不展开每个kernel。最后一行是工程结果可信的基础，并非用审计替代加速。')
frame('时间与宽度如何读取',tab(['数字','含义','容易误读的地方'],[['核心秒数','完整推进含初始化、图捕获、SR和同步','不含进程启动和独立输出'],['进程秒数','runner启动到退出','含控制器等启动；仍不含独立导出'],['宽度比','当前范围宽度 / Flow*范围宽度','1接近；更小不是正确性证明'],['p95 / max','逐通道统计，再取最差通道','不能删最差步或只报中位'],['成功分区步','全部真实分区的接受步数','不完整任务不能作成功加速分母']],[.20,.38,.42])+source('Endpoint：步末范围；tube：整步范围。报告附录保留两类输出和每一物理坐标。'),'这里统一听众对表的理解。后续每个时间表都明确单位和边界；宽度全部相对匹配Flow*。')
frame('固定多项式实验配置',tab(['参数','Van der Pol','Brusselator'],[['原盒','[1.1,1.4] × [2.35,2.45]','[1.48,1.52] × [2.98,3.02]'],['阶数 / h','4 / 0.01','6 / 0.02'],['完整单盒','B1×1000，T10','B1×1000，T20'],['真实分区','8×4，B32×20，T0.2','8×4，B32×20，T0.4'],['cutoff / 余项半径','1e−10 / 1e−4','1e−10 / 1e−4'],['SR容量','100','1000']],[.25,.375,.375])+txt('V100 16GB / Xeon Gold 6138；单CPU线程；每组五次；共120次。')+source('S1/S6。VDP 主比较固定原式 y−x−x*x*y；分区不是复制同一盒。'),'强调单盒长轨迹与32真实分区短轨迹是两种工作量，不将吞吐实验假装完整长时域。')
frame('固定任务：完整核心时间',plot('plant_core.pdf','5.45cm')+txt('B1 显著快于原版 strict；VDP B1 仍慢于 Flow*。')+source('S1：五次中位秒，全部完成。初始化/捕获计入；进程启动与输出排除。'),'右图、左图刻度不同。别只说Brusselator的漂亮结果，紧接着指出VDP还没有达标。')
frame('固定任务：具体时间与达标情况',tab(['任务','我们','Huan','Xiangru','Flow*'],core_rows,[.26,.185,.185,.185,.185])+txt('单位：核心秒，五次中位数；三种 GPU 实现均为 strict。')+r'\vspace{0.25cm}'+tab(['我们 / Flow* 配对比','VDP B1','VDP B32','Bruss B1','Bruss B32'],[['目标 ≤1.20','2.015 未达','1.129 达到','0.322 达到','0.111 达到']],[.28,.18,.18,.18,.18])+source('S1；比值来自同轮配对，中位数之比不作为配对比替代。'),'主表是绝对秒数；下一行是配对比例，小于1表示我们更快。总目标四项需要全部达到。')
frame('固定任务：具体全程宽度',tab(['任务','我们 strict','Huan strict','Xiangru strict'],width_rows,[.25,.25,.25,.25])+r'\vspace{0.35cm}'+bullets('每格：最差通道 p95 / max；Flow* 自身为 1 / 1。','目标：p95 ≤1.10，max ≤1.25；原式 VDP B1 仍未达到。','其余三项达到；B1 长轨迹比作者 strict 更接近 Flow*。')+source('S2：65,600行、80通道；全部真实分区与全轨迹。'),'Huan和Xiangru宽度一样是完整数值对照验证过的结果，不是为了简化表格而假定一样。')
frame('VDP：范围差距出现在后段',plot('van_der_pol_widths.pdf','6cm')+source('原式 B1×1000；分开展示 endpoint/tube 与 x/y；虚线为最大值门槛1.25。'),'曲线说明只看平均宽度会掩盖局部差距。我们最大比值1.271071出现在endpoint-y第982步。')
frame('Brusselator：长轨迹范围明显改善',plot('brusselator_widths.pdf','6cm')+source('B1×1000；我们最差 p95/max=1.012228/1.038326。'),'Brusselator核心时间和范围都有明显进展，但稍后还要看范围输出成本，不能据此声称整个流程都领先。')
frame('作者闭环控制实验配置',tab(['参数','TORA','单摆','CartPole'],[['B×ODE步数','12×200','1×100','1×200'],['阶数 / h','3 / 0.1','2 / 0.01','6 / 0.005'],['控制周期 × 周期数','1.0 × 20','0.05 × 20','0.02 × 50'],['总时长','20','1','1'],['余项猜测','±0.01','±0.01','±0.1']],[.31,.23,.23,.23])+bullets('同模型、box / same-slope / flat / rpc-float32 控制合同。','匹配 native 固定步数和共同 binary64 初盒；原始支路另存。','TORA/单摆各六组五次；CartPole 六组单次完整尝试。')+source('S3/S4/S6。模型、全部初盒、方程与约束在附录和 YAML。'),'让听众理解这是周期控制+ODE推进，不是只有一条ODE。CartPole本次是小初盒历史full profile。')
frame('TORA：全程成功，进程更快，范围仍宽',tab(['实现','进程秒','接受分区步','全程p95 / max'],[[armnames[r['arm']],f"{r['process_median_s']:.3f}",f"{r['accepted']}/2400",'未完整' if not r['complete'] else f"{r['width_p95']:.6f} / {r['width_max']:.6f}"] for r in nn if r['case']=='tora'],[.28,.16,.21,.35])+r'\vspace{0.25cm}'+txt('Flow* 耗时 / 我们耗时：完整成功进程配对比为 1.802。')+txt('Huan/Xiangru strict 未完整；parity 只作参考，不替代严格保证。')+source('S3/S7；五次进程中位秒，含启动及控制器；范围来自独立完整轨迹。'),'不用成功速度去排名Huan和Xiangru的失败运行。尽管1.8倍更快，2.71的最大宽度比仍是重要不足。')
frame('TORA：主要数值进展',tab(['阶段','成功分区步','首个失败步'],[['初始严格版','2354 / 2400','189'],['直接变量候选','2393 / 2400','197'],['线性路径 v2','2400 / 2400','全程完成']],[.38,.34,.28])+r'\vspace{0.3cm}'+bullets('改进多项式信息的保留方式，未改步长、阶数、初盒或控制器。','共同前188步 p95/max：2.556/8.855 → 1.384/1.884。','新v2全程仍为1.592/2.711；完成率问题解决，紧度问题仍在。')+source('S7；仅加、减、取负的安全线性路径适用；非线性消费者回退。'),'讲清因果方向：不是人为放大余项猜测让实验强行接受，而是减少不必要的多项式截断信息损失。仅指定路径适用。')
frame('单摆：范围接近，完整进程略慢于 Flow*',tab(['实现','进程秒','成功步','全程p95 / max'],[[armnames[r['arm']],f"{r['process_median_s']:.3f}",'100/100',f"{r['width_p95']:.6f} / {r['width_max']:.6f}"] for r in nn if r['case']=='single_pendulum'],[.28,.16,.18,.38])+r'\vspace{0.25cm}'+txt('我们比原版 strict 更紧；与 Flow* 相比，完整进程约慢 5.5%。')+source('S3；五次进程中位秒；作者内部打印计时边界不同，不能替换本表。'),'这项结果比较平衡：宽度改善确实有用，但启动和控制器总成本使我们的进程总耗时仍比Flow*高。')
frame('CartPole：共同前缀接近，六组都提前结束',tab(['实现','带记录进程秒','成功步','共同141步p95 / max'],[[armnames[r['arm']],f"{r['process_wall_s']:.3f}",'141/200' if r['arm']=='native' else '156/200','1 / 1' if r['arm']=='native' else f"{next(w for w in ca['comparisons'] if w['arm']==r['arm'])['p95']:.6f} / {next(w for w in ca['comparisons'] if w['arm']==r['arm'])['maximum']:.6f}"] for r in cr],[.24,.20,.18,.38])+r'\vspace{0.2cm}'+txt('GPU第157步拒绝；native仅保存141步。没有全程成功或正式加速比。')+source('S4：单次诊断；B1×200，h=.005，小初盒registry full。不是4096分区或10秒ARCH-COMP任务。'),'这张表直接回答CartPole比较，但不要过度解读native更早终止的原因；当前缺少拒绝差异的因果分析。')
frame('核心之外：范围输出仍影响实际体验',tab(['任务','旧输出秒','新输出秒','配对耗时下降'],output_rows,[.31,.23,.23,.23])+r'\vspace{0.3cm}'+bullets('相同完整展开、范围、gzip与CSV；旧/新各五次，模型保持一致。','改进来自原生 CPU 多项式乘法，不能称纯 GPU 加速。','Brusselator B1 降到41.76秒，仍高于作者原版单次约10.6秒。')+source('S5；不含求解，不将两个阶段中位数相加冒充实测端到端。'),'让听众看到为什么不只盯核心时间：模型展开和输出还会决定拿到结果要等多久。')
frame('完整任务的速度不能用小算子倍率代替',bullets('GPU乘法候选：VDP完整1000步，2.831 → 2.734秒，五配对下降3.30%。','这组使用regrouped RHS且未新跑native，不能覆盖原式验收。','暂停前当前原式诊断确认组合计算仍占重要成本；诊断不是新五次正式加速结论。','下一次恢复后应针对大瓶颈和最差宽度步，再做完整同任务验收。')+source('S5；最后诊断 f536/d8 各三次1000步数值记录与冻结原式一致。'),'这一页只说明工程取舍，不介绍微小kernel实现。3.3%的收益是真实但不足以解决约两倍的VDP差距。')
frame('更广的 benchmark：覆盖与缺口',tab(['范围','已完成的工作','不能声称的结论'],[['12个Huan registry plant','native原时域11完成；Robertson第一步拒绝','不是12项都有当前GPU四方正式对照'],['12个可运行NNCS完整筛查','早期9项完整接受；后续TORA已改善','历史native-f64不能直接混入rpc-float32对照'],['TORA / 单摆 / CartPole','相同配置六组完整尝试；前两项五次计时','CartPole未完成，不能作全时域成功比较'],['Airplane / 原QUAD','规模限制或配置问题已明确记录','另立合同不能冒充原配置完成']],[.28,.37,.35])+source('S8；详细任务名、成功步数及配置在报告第12节。'),'用覆盖矩阵代替一串杂乱实验日志。整体探索范围很大，但成熟比较的范围必须实话实说。')
frame('严格性与剩余工作',bullets('保留：向外误差承载、完整区间余项、SR历史、失败原子回滚。','已检查：源码与实际库身份、全模型一致、Fraction范围、CPU/CUDA及跨SR重置。','仍缺：CROWN控制器舍入/注入的完整严格资格；不能称NNCS端到端严格证书。','仍缺：原式VDP时间和宽度、TORA末段紧度、候选统一入口与剩余benchmark对照。')+source('S1–S7；测试通过与范围接近均不是全程序数学证明。'),'这页解释结论边界，避免听众将strict标签理解为整个神经控制系统已经形式证明。')
frame('交付与暂停点',tab(['交付','内容'],[['详细报告','背景、版本、所有主比较、实验参数、失败与附录'],['演示材料','本Beamer幻灯片、讲稿、LaTeX源码、向量图'],['可复算证据','原始时间样本、完整范围CSV、审计JSON和配置'],['独立新分支','从4beba860基线建立；只新增本次汇报资料']],[.23,.77])+r'\vspace{0.45cm}'+txt('本次资料推送完成后暂停，不继续启动新优化或实验。')+txt('总目标仍未完成；恢复工作需继续按原验收条件推进。'),'结尾明确当前是阶段交付与暂停，不是把未达到的目标改成已完成。')
# Appendix: concrete contracts/extra requested comparisons.
slides.append(r'\appendix')
frame('附录：完整进程时间（固定plant）',tab(['任务','我们 strict','Huan strict','Xiangru strict','Flow*'],process_rows,[.25,.18,.18,.18,.18])+bullets('五次中位秒；含启动，不含独立范围导出。','B32短任务中启动成本显著，核心领先不保证进程领先。')+source('S1，全部120个原始time-only样本。'),'被问到冷启动或端到端时使用本页；这不是首次安装编译时间，也不是模型输出全过程。')
frame('附录：作者parity的固定plant参考',tab(['任务','Huan核心秒','Xiangru核心秒','共同p95 / max'],[[pnames[p]+f' B{b}',num(tr(p,b,'huan','parity')['core_median']),num(tr(p,b,'xiangru','parity')['core_median']),wratio(wr(p,b,'huan','parity'))] for p,b in cases],[.28,.21,.21,.30])+source('S1/S2；parity与strict误差保证不同，不计入strict达标。'),'用于解释“Xiangru说时间和宽度都像Flow*”可能指另一模式、另一硬件、另一任务，不能直接套到本表的strict原式VDP。')
for p in plants:
 rows=[]
 for i in ['ours','huan']:
  for ch,s in wr(p,1,i)['channels'].items():rows.append(['我们' if i=='ours' else 'Huan=Xiangru',ch,num(s['p95']),num(s['max'])])
 frame('附录：'+pnames[p]+' B1 四通道宽度',tab(['实现','通道','p95','max'],rows,[.30,.28,.21,.21])+source('S2；全部1000步；绝对边界差与B32逐通道表在详细报告附录A。'),'每个通道单独列出，防止只用最坏汇总隐藏endpoint和tube之间的区别。')
frame('附录：NNCS初盒与方程',tab(['任务','初始物理状态','方程'],[['TORA','[0.6,0.7]；[−0.7,−0.6]；[−0.4,−0.3]；[0.5,0.6]',"x1'=x2；x2'=−x1+0.1sin(x3)；x3'=x4；x4'=u−10"],['单摆','[1,1.175]；[0,0.2]',"x1'=x2；x2'=2sin(x1)+8u"],['CartPole','[−.0375,−.03125]；[−.015625,−.0125]；[−.00625,0]；[−.007375,−.00625]',"x1'=x2；x2'=2u；x3'=x4；x4'见完整YAML"]],[.13,.42,.45])+txt('附加状态：t、u 初值为0；每周期内 t′=1、u′=0。')+source('S6；全部约束、模型、形状、CartPole完整表达式均随configs/*.yaml交付。'),'模型权重没有打进报告包，模型路径和SHA在证据中，防止误以为zip包含完整运行环境。')
frame('附录：证据与复现',bullets('REPORT.md / report.pdf：详细阅读；slides.pdf：直接演示。','build_materials.py：由固定JSON/CSV重算表格并生成图形。','build.sh：使用XeLaTeX编译；字体采用TeX Live自带Fandol。','evidence/、tables/、configs/：版本、原始样本、全范围表和配置。','大型模型、ONNX权重及CUDA二进制保留服务器，包内不冒充包含它们。')+source('索引：REPORT.md 附录C；归档SHA与文件校验表随交付。'),'说明如何使用材料：可以直接看PDF，也可以在本地或Overleaf用XeLaTeX修改slides.tex重新编译。')
preamble=r'''\documentclass[UTF8,fontset=fandol,aspectratio=169,10pt]{ctexbeamer}
\usepackage{booktabs,array,graphicx,tikz,amsmath}
\usetikzlibrary{positioning,arrows.meta}
\definecolor{navy}{HTML}{20364C}\definecolor{teal}{HTML}{147D92}
\setbeamercolor{normal text}{fg=navy,bg=white}
\setbeamercolor{structure}{fg=teal}
\setbeamercolor{frametitle}{fg=navy,bg=white}
\setbeamerfont{frametitle}{size=\large,series=\bfseries}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{\hfill\color{gray}\scriptsize\insertframenumber\hspace{0.5cm}\vspace{0.2cm}}
\setbeamersize{text margin left=0.7cm,text margin right=0.7cm}
\setbeamertemplate{itemize item}{\color{teal}\small$\bullet$}
\setlength{\parskip}{5pt}\emergencystretch=2em
\begin{document}
'''
(D/'slides.tex').write_text(preamble+'\n'.join(slides)+r'\end{document}'+'\n')
(D/'SPEAKER_NOTES.md').write_text('# 演示讲稿\n\n建议主讲约20–25分钟，附录用于回答配置与实验口径问题。\n\n'+'\n'.join(notes))
print('generated',len(notes),'slides')
