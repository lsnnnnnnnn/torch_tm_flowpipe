#!/usr/bin/env python3
"""Assemble current reports from saved evidence only; no solver, checker or digest."""
from contextlib import ExitStack
import argparse, csv, hashlib, importlib.util, json, os, re
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/evidence/results/archcomp26_report_20261005'
SOURCE=ROOT/'docs/ARCHCOMP26_FINAL_REPORT_DRAFT.md'
METHODS=('pytorch_gpu','huan','xiangru','flowstar_native')
LABELS={'pytorch_gpu':'P3','huan':'Huan','xiangru':'Xiangru','flowstar_native':'原生 Flow*'}
SELECTED={'acc-safe-distance':'acc_p3_fast1_20261005_001','nav-robust':'nav_robust_p3_fast32_20261005_001','quad-reach':'quad_paper_p3_private256_full1000_20261005_001','tora-reach-sigmoid':'tora_sigmoid_official_u11_p3_fused1_20261005_001','unicycle-reach':'unicycle_p3_fused1_20261005_001'}
BASE_DRIVER={'acc-safe-distance':4.720203388,'nav-robust':15.531658911,'quad-reach':1350.525528709,'tora-reach-sigmoid':9.793604707,'unicycle-reach':13.61720345634967}
BASE_OUTER={'acc-safe-distance':8.790101008,'nav-robust':None,'quad-reach':1357.554950926,'tora-reach-sigmoid':13.958589417,'unicycle-reach':17.819782309234142}

def read(path): return json.loads(path.read_text())
def f(x): return '—' if x is None else f'{x:.8g}'
def sec(x): return '—' if x is None else f'{x:.3f}'
def rel(path): return os.path.relpath(ROOT/path, SOURCE.parent)
def link(label,path): return f'[{label}]({rel(path)})'
def table(headers,rows):
    def clean(v): return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(clean,row))+' |' for row in rows]])+'\n'

def source_line(cells):
    return '原始结果：'+'；'.join(link(LABELS[c['method']], c['current_selected_receipts'][0]['path']) if c['current_selected_receipts'] else LABELS[c['method']]+'未启动' for c in cells)+'。'

def load_builder():
    path=ROOT/'docs/evidence/results/archcomp26_20261001/stage_report/build_archcomp26_report_20261004.py'
    spec=importlib.util.spec_from_file_location('archived_docx_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.SOURCE=SOURCE;module.OUT=OUT/'ARCHCOMP26_REPORT_20261005.docx'
    return module

def docx_build():
    h=load_builder()
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Pt, Cm, RGBColor
    from docx.package import ImageParts
    def addtable(doc,block):
        rows=[h.table_cells(x) for x in block];assert h.is_divider(rows[1])
        rows=[rows[0],*rows[2:]];cols=len(rows[0]);assert 2<=cols<=5
        widths={2:[4.0,13.5],3:[4.5,6.5,6.5],4:[5.2,4.1,4.1,4.1],5:[4.5,3.25,3.25,3.25,3.25]}[cols]
        t=doc.add_table(rows=0,cols=cols);t.autofit=False;t.style='Table Grid'
        for ridx,row in enumerate(rows):
            assert len(row)==cols,(len(row),cols,row)
            cells=t.add_row().cells
            for j,value in enumerate(row):
                c=cells[j];c.width=Cm(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                h.shade(c,'E8EDF1' if ridx==0 else ('F7F8FA' if ridx%2==0 else 'FFFFFF'))
                p=c.paragraphs[0];p.paragraph_format.keep_with_next=(ridx==0);p.paragraph_format.space_before=Pt(2);p.paragraph_format.space_after=Pt(2);p.paragraph_format.line_spacing=1.08
                if j: p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                h.inline(p,value)
                for r in p.runs:r.font.size=Pt(9.3);r.bold=ridx==0
            h.avoid_row_split(t.rows[ridx])
        h.repeat_header(t.rows[0]);p=doc.add_paragraph();p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(0);p.paragraph_format.line_spacing=0.3
    with ExitStack() as guard:
        for name in (*hashlib.algorithms_guaranteed,'new','file_digest'):guard.enter_context(patch.object(hashlib,name,h.forbid_digest))
        guard.enter_context(patch.object(ImageParts,'get_or_add_image_part',h.image_part_by_bytes))
        doc=Document();h.initialize(doc)
        for name,size in [('Normal',10.5),('Stage Note',10),('Evidence Body',10),('List Bullet',10.5),('Caption',9),('Title',20),('Heading 1',14),('Heading 2',11.5)]:
            doc.styles[name].font.size=Pt(size)
        doc.styles['Normal'].paragraph_format.line_spacing=1.17
        doc.styles['Normal'].paragraph_format.space_after=Pt(5)
        doc.sections[0].footer.paragraphs[0].clear()
        foot=doc.sections[0].footer.paragraphs[0];foot.text='2026 年 10 月 6 日  ·  时间和保存宽度比较  ·  ';foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT;h.add_page_field(foot)
        for r in foot.runs:h.set_font(r.font,r._element);r.font.size=Pt(8)
        doc.core_properties.title='ARCH COMP26 四方时间与区间宽度比较报告'
        lines=SOURCE.read_text().splitlines();i=0
        while i<len(lines):
            line=lines[i]
            if not line.strip():i+=1;continue
            if line.startswith('|'):
                block=[]
                while i<len(lines) and lines[i].startswith('|'):block.append(lines[i]);i+=1
                addtable(doc,block);continue
            if line.startswith('# '):doc.add_paragraph(line[2:],'Title')
            elif line.startswith('## '):
                p=doc.add_heading(line[3:],level=1)
                if line.startswith('## 1 ') or line.startswith('## 附录 A') or line.startswith('## 附录 B'):p.paragraph_format.page_break_before=True
            elif line.startswith('### '):doc.add_heading(line[4:],level=2)
            elif line.startswith('![矢'):
                pass
            elif line.startswith('!['):
                match=h.IMAGE_LINK.search(line)
                if match:h.add_figure(doc,*match.groups())
            elif line.startswith('- '):h.add_text(doc,line[2:],'List Bullet')
            elif line.startswith('> '):h.add_text(doc,line[2:],'Stage Note')
            else:
                p=h.add_text(doc,line)
                if line.startswith(('**共同终点','**截至共同','**各法自身')):p.paragraph_format.keep_with_next=True
            i+=1
        doc.save(h.OUT)
    return h

def assemble():
    metadata=read(OUT/'report_metadata.json');status=read(OUT/'blockers/status_64cells.json');timing=read(OUT/'timing/timing_index.json');width=read(OUT/'widths/summary.json');runs=read(ROOT/'research/p3_speed_tightness_20261005/RUN_INDEX.json')['runs']
    ws={(r['instance_id'],r['method'],r['state']):r for r in width};cells={(r['instance_id'],r['method']):r for r in status['cells']};new={r['run_id']:r for r in runs if r['stage']=='run_001'}
    specs={r['instance_id']:r for r in read(OUT/'widths/instances.json')}
    def chosen(key,method):
        runkey=SELECTED[key]+'/run_001' if key in SELECTED and method=='pytorch_gpu' else timing['benchmarks'][key]['methods'][method]['displayed_run']
        return timing['runs'].get(runkey) if runkey else None
    def timer(key,method,layer):
        r=chosen(key,method);return r['timings'].get(layer) if r else None
    lines=[]
    def add(value=''):lines.append(value)
    add('# ARCH COMP26 四方时间与区间宽度比较报告')
    add('证据截点 2026 年 10 月 5 日　编制于 2026 年 10 月 6 日北京时间　16 个 benchmark × 4 种方法')
    add('当前 P3 已在五个完整实例上记录到保持已保存输出不变的单次时间下降，但还没有做到所有 benchmark 都比 Huan、Xiangru 和原生 Flow* 更快，也没有新增全时全状态紧度优势。论文 QUAD 是最明显的速度短板：新进程 1004.197 秒，Huan / Xiangru 的既存完整进程为 94.583 / 108.018 秒；其高度终点区间也更宽。ACC、Attitude、Unicycle 有保存终点坐标较窄的结果，其他实例要逐态看，不能合成一个“整体更紧”的结论。')
    add('本版重写当前总报告、时间表、宽度表与逐方法失败说明。冻结的 289 条尝试保持原样，另列 13 个优化与资格阶段（12 个新数值候选和 1 个 GPU 资格门）。64 个方法格仍为 38 个本轮数值完整、8 个可复用的同合同历史完整、14 个没有完整数值时域、4 个 Airplane discrete 合同阻塞。五个新完整 P3 结果改进已有格，不增加覆盖格数。本次报告整理没有重跑旧实验或旧数值检查器。')
    add('读数规则：时间单位均为秒；宽度为保存上界减下界。正文按物理状态原单位列数值，不把不同单位加总。宽度保留八位有效数字，时间保留三位小数，完整 binary64 可读数、上下界、逐步范围、路径和行号见 CSV / JSON。“—”表示未保存、无有效全初集前缀或合同未执行；安全前缀不适用时另标注，原因逐节说明，不表示零。')
    add('## 全部 benchmark 的完整进程时间')
    add('下表仅列已完整覆盖具名时域且有明确外层 process wall 的所选单次进程；五个 P3 新候选使用 10 月 5 日结果，其余保持原所选进程。历史复用注明“历史”。无全程格不填失败耗时，失败进程时间仍完整列入各节和原始时间表。单次、新旧日期、硬件路径与并发不同，因此此表用于查看实际代价，不是稳定速度排行榜。')
    time_rows=[]
    for meta in metadata:
        key=meta['instance_id'];row=[meta['title']]
        for m in METHODS:
            c=cells[key,m];v=timer(key,m,'process_wall_s')
            row.append(sec(v)+(' 历史' if c['coverage_classification']=='historical_same_contract_numeric_full' and v is not None else '') if c['numeric_progress']['full_numeric_horizon'] else '未全程' if c['status']!='contract_blocked' else '合同缺失')
        time_rows.append(row)
    add(table(['实例','P3','Huan','Xiangru','原生'],time_rows))
    add('NAV 旧 P3 的 28.999885 / 18.703898 是 payload wall，不能填入外层 process wall。NAV 与 TORA tanh 的部分历史作者时间是 driver call，另有内部 driver elapsed，两者也不混合。所有可读时间层级在各节展开；未知外层保持空缺。')
    add('### 四组既有重复进程计时')
    add('以下各格为原 campaign 后五个独立进程的中位数 [最小值, 最大值]，每格 n=5；不包含首进程，不混入新优化单次。原 campaign 在共享服务器轮换执行，不能推出跨机器或独占资源下的稳定排名。')
    rows=[]
    for meta in metadata:
        key=meta['instance_id'];methods=timing['benchmarks'][key]['methods']
        if not methods['pytorch_gpu'].get('campaign'):continue
        row=[meta['title']]
        for m in METHODS:
            d=methods[m]['campaign']['later_statistics']['process_wall_s'];row.append(f"{sec(d['median'])} [{sec(d['min'])}, {sec(d['max'])}]")
        rows.append(row)
    add(table(['原 campaign','P3','Huan','Xiangru','原生'],rows))
    add('全部首进程、后续进程、四层计时及附加 native / driver-call 时间见 '+link('逐进程 CSV','docs/evidence/results/archcomp26_report_20261005/timing/runs.csv')+' 与 '+link('分布 CSV','docs/evidence/results/archcomp26_report_20261005/timing/campaign_statistics.csv')+'。')
    add('## 五个新 P3 完整结果')
    add('下表比较同一 driver 内部计时边界的保存参考与新候选。百分比为本次单样本描述性下降，启动路径差异与共享机器负载尚未通过重复、交错实验隔离。新候选没有删除绘图所需逐步输出。')
    rows=[]
    for meta in metadata:
        key=meta['instance_id']
        if key not in SELECTED:continue
        n=new[SELECTED[key]];old=BASE_DRIVER[key];now=n['driver_elapsed_s']
        rows.append([meta['title'],f'{old:.3f} → {now:.3f}',f'{100*(old-now)/old:.2f}%',sec(n['outer_wall_s']),'保存宽度相同'])
    add(table(['实例','driver 旧 → 新','单次下降','新 process','紧度变化'],rows))
    add('ACC、QUAD、sigmoid、Unicycle 的匹配外层保存样本分别为 8.790→8.037、1357.555→1004.197、13.959→9.392、17.820→11.698 秒；NAV 没有对应旧外层记录，只比较内部 15.532→12.667 秒。新 NAV payload 把 Torch 导入移出计时，不能拿它和旧 payload 做净加速比例。QUAD 新全程与三条 GPU2 短作业在不同 GPU 上有时间重叠，资源和时间窗均在索引。')
    add('“保存宽度相同”指收据实际比过的对象。ACC、NAV、sigmoid、Unicycle 比较了完整保存范围及相应状态/配置字段；QUAD 比较 1000 条 pooled tube/endpoint 观察、接受记录与科学终态字段，未保存可供比较的逐盒全程几何，也没有证明内部 TM/SR 对象逐项相同。它们都不是新取得的独立 NNCS 浮点证书。')
    add('![五个完整候选的内部 driver 时间对比](evidence/results/archcomp26_report_20261005/figures/driver_before_after.png)')
    add('### 速度改进来自哪里')
    add('恢复了历史快分支中已存在的私有输出分配与 Horner 内核绑定；按小批量实际盒数使用 1 / 32 行加权图，减少无效填充；论文 QUAD 使用 256 行加权分块减少图调用。单盒 fused 候选把原本两轮映射纳入同一图，仍要求两轮条件满足并保留原路径回退；首个合格输入与原细化的直接比较耗时计入新进程。ACC fused 外层 8.089 秒没有胜过较简单的 small1 8.037 秒，因此当前 ACC 选择 small1。')
    add('已找回的旧快分支是 codex/progress-report-20260923，对应服务器 engine_linear_leaf_v2。它在当时 NAV robust、TORA tanh、旧 sigmoid、旧 Unicycle、ACC 五组保存中位数较低；Attitude 和 Single Pendulum 并非全胜。旧 sigmoid 是 22(f−0.5)，新合同是 11f；旧 Unicycle 扰动方程也不同，且旧计时不含现在的逐步几何导出。因此不能把旧数字直接替换当前主表。'+link('历史时间和合同差异','research/p3_speed_tightness_20261005/README.md')+'保留原来源。')
    add('论文 QUAD 的已知开销差异包括：Huan / Xiangru 使用 work / point / validation 阶 2/1/1 与 parity 路径，P3 使用 3/2/4、严格区间误差账本和两轮已接受路径细化。四方主配置均是 1024 盒、h=0.005、1000 小步；P3 的 K=20 是每 20 步重算完整保留历史，并非只留 20 步。尚无匹配的分阶段消融，不能给这些差异分配因果百分比。')
    add('## 宽度比较口径')
    add('每步先对同一方法的全部有效初盒取坐标并集，保存 lo、hi 和 width=hi−lo。endpoint 是传播终点，tube 是整个小步。时钟、保持控制和 Unicycle 常值扰动等辅助量不并入物理态比较，原始记录仍保留。几何源进程与所选计时进程可能不同，各自按源表追溯，不伪装为同一次实验。正文第一张表在四方均覆盖全初集的共同终点时刻比较 endpoint；第二张表取从起点至共同数值时刻各小步 tube 宽度的最大值。后者不是“全时间并集的宽度”：运动距离不被当作单步包络松弛。全时并集、每盒均值/最大值只在对应来源实际保存时另列，不能由 pooled 曲线倒推。')
    add('提前停止时共同数值时刻与性质安全时刻分开：Balancing raw4 为 0.415 秒；DP more 数值 0.32 秒、安全 0.30 秒；TORA remain h=0.1 数值 18.9 秒、安全 18.4 秒。各法自身最后记录和完整初集最后记录另外保存，不把不同时间点的宽度互相比。QUAD 作者两法只有终点，故全时 tube 栏明确缺失。')
    add('宽度较小只说明这个保存投影窄，不保证集合包含。'+link('逐态成对差值与包含关系','docs/evidence/results/archcomp26_report_20261005/widths/pairwise_comparisons.csv')+'另列 P3 对三方的绝对差、相对差和 endpoint 区间包含；没有把不同物理量汇总成一个紧度分数。')
    comparisons=[]
    for meta in metadata:
        key=meta['instance_id'];cs=[cells[key,m] for m in METHODS];spec=specs[key]
        add(f"## {meta['number']} {meta['title']}")
        add(meta['contract'])
        add('### 数值范围与未完成原因')
        rows=[]
        for c in cs:
            p=c['numeric_progress'];full=p['full_numeric_horizon'];prefix=p['full_initial_set_accepted_or_valid_saved_prefix_steps']
            rows.append([LABELS[c['method']], '完整' if full else '合同缺失' if c['status']=='contract_blocked' else '无全程', f"{prefix if prefix is not None else '—'} / {p['planned_substeps_or_transitions']}", f(p['full_initial_set_prefix_time_s']),('性质窗见正文' if key=='single-pendulum-reach' else '不适用' if key in ('quad-reach','tora-reach-sigmoid','tora-reach-tanh','unicycle-reach') else f(p['saved_all_boxes_safe_prefix_time_s']))])
        add(table(['方法','数值状态','全初集步数','数值前缀 t','保存安全前缀 t'],rows))
        grouped={}
        for c in cs:
            text=c['reason_zh']+' '+c['needed_evidence_or_change_zh'];grouped.setdefault(text,[]).append(LABELS[c['method']])
        for text,labels in grouped.items(): add('**'+' / '.join(labels)+'：** '+text)
        verdicts={}
        for c in cs:verdicts.setdefault(c['verdict']['saved_geometry_observation_zh'],[]).append(LABELS[c['method']])
        for text,labels in verdicts.items():add('保存性质观察（'+' / '.join(labels)+'）：'+text)
        add(source_line(cs))
        if key in SELECTED:add('当前 P3 使用 '+link(SELECTED[key],f"research/p3_speed_tightness_20261005/results/{SELECTED[key]}/run_001/RESULT.json")+'；以下宽度继承其已直接比对相同的保存参考对象。')
        add('### 所选进程各层时间')
        add('下表为各方法所选单次；未全程方法的数值只是这次失败或早停的耗时，不能参与完整运行速度比较。缺失表示该层没有独立保存，不能从另一层代填。')
        layers={'process_wall_s':'外层 process','wrapper_wall_s':'候选 wrapper','payload_wall_s':'runner payload','driver_elapsed_s':'内部 driver','driver_call_wall_s':'调用 driver','native_process_wall_s':'原生子进程','server_startup_s':'控制器启动','watchdog_process_wall_s':'watchdog'}
        rows=[]
        for field,label in layers.items():
            vals=[timer(key,m,field) for m in METHODS]
            if any(v is not None for v in vals):rows.append([label,*[sec(v) for v in vals]])
        if rows:add(table(['计时层级','P3','Huan','Xiangru','原生'],rows))
        elif key=='airplane-discrete':add('四方法均未在权威离散合同下启动，因此没有运行耗时或资源记录；另立的 CPU 诊断不填方法格。')
        else:add('没有具备已确认计时边界的所选进程。已保存入口尝试和未分类的原始 index wall 保留于逐进程 CSV，未猜测成求解时间。')
        res=[]
        for m in METHODS:
            r=chosen(key,m)
            if not r:continue
            rr=r['resources'];res.append(LABELS[m]+f" CPU={rr.get('cpu_affinity') if rr.get('cpu_affinity') is not None else '未记录'}，GPU={rr.get('physical_gpu') if rr.get('physical_gpu') is not None else '无或未记录'}")
        if res:add('保存资源：'+'；'.join(res)+'。进程起止、同批并发窗口、字段路径详见时间索引。')
        add('### 全部物理状态宽度')
        states=spec['physical_states'];ctimes=sorted({ws[key,m,s]['common_time'] for m in METHODS for s in states if ws[key,m,s]['common_time'] is not None})
        if ctimes:add('四方共同数值终点 t='+', '.join(f(x) for x in ctimes)+' 秒。若某一状态没有此共同终点，保持空白。')
        else:add('主合同不存在四方共同有效数值终点，以下空栏标明数据缺失，不借用其他合同或分盒短诊断。')
        if key.startswith('airplane-'):
            add('全部 12 个状态 '+', '.join(states)+'：四方法均无主合同有效 endpoint / tube 宽度，不能填终点或全时数值。逐状态缺失格仍全部列在 summary.csv 和 missing_fields.csv。')
        else:
            add('**共同终点 endpoint 绝对宽度**')
            add(table(['状态','P3','Huan','Xiangru','原生'],[[s,*[f(ws[key,m,s]['common_endpoint_width']) for m in METHODS]] for s in states]))
            add('**截至共同数值时刻的最大单步 tube 绝对宽度**')
            add(table(['状态','P3','Huan','Xiangru','原生'],[[s,*[f(ws[key,m,s]['common_max_tube_width']) for m in METHODS]] for s in states]))
        for other in METHODS[1:]:
            narrow=[];wide=[];equal=[]
            for s in states:
                a=ws[key,'pytorch_gpu',s];b=ws[key,other,s]
                for field in ('common_endpoint_width','common_max_tube_width'):
                    x,y=a[field],b[field]
                    if x is None or y is None:continue
                    contained=(a['common_endpoint_lo']>=b['common_endpoint_lo'] and a['common_endpoint_hi']<=b['common_endpoint_hi']) if field=='common_endpoint_width' else None
                    comparisons.append(dict(instance_id=key,state=s,geometry=field,reference_method=other,common_time=a['common_time'],p3_width=x,reference_width=y,difference=x-y,relative_difference_percent=100*(x/y-1) if y else None,p3_endpoint_subset=contained))
                x,y=a['common_endpoint_width'],b['common_endpoint_width']
                if x is None or y is None:continue
                (narrow if x<y else wide if x>y else equal).append(s)
            if narrow or wide or equal:add('共同终点 P3 对 '+LABELS[other]+'：'+('较窄 '+', '.join(narrow)+'；' if narrow else '')+('较宽 '+', '.join(wide)+'；' if wide else '')+('相等 '+', '.join(equal)+'；' if equal else '')+'逐态差值见配套 CSV。该比较不外推到其他时刻或未保存相关方向。')
        if not all(c['numeric_progress']['full_numeric_horizon'] for c in cs) and ctimes:
            add('**各法自身最后完整初集终点宽度**（时刻不同，仅记录，不横向排名）')
            add(table(['状态','P3','Huan','Xiangru','原生'],[[s,*[f(ws[key,m,s]['own_last_complete_endpoint_width'])+' @'+f(ws[key,m,s]['own_last_complete_t']) for m in METHODS]] for s in states]))
        if key=='quad-reach':
            add('P3 主表使用观察器最后 endpoint；另存 driver final_hull，二者 x5 宽度相差约 5.69×10⁻⁶，不能互换。Huan / Xiangru 只有 driver 终态坐标范围，逐步 tube 未保存，因此不能比较其全时紧度。原生已有 SCAN.json 保存了完整 1000 步 × 12 态 pooled 界，本版直接读取该保存扫描补齐；没有重新扫描远端大文件，来源和记录数在补充回执。')
            add('原生历史终点 VERIFIED 不是参与者 reach-and-remain 全时 checker。保存数值包络的连续入高度带后缀：native 约 3.87 秒，P3 约 3.95 秒；P3 需联合 tube 与 endpoint，不能忽略末位舍入差。它们不补齐 Huan / Xiangru 的逐步数据或独立全时证书。')
            add('原生八方向生产门仍关闭。已保存的修补版全 1024 盒首个 h=0.005 plant 条件性门覆盖 20,480 个合成物理态界；lane 0 第二步门只涉及一盒。10 月 4 日控制余项构造回放已检查 1024 盒、3072 输出、196608 个精确仿射顶点，四个阶段 exit 0，但依赖原实数 CROWN 包络有效，未调用 NN/CROWN 或 ODE。它不认证后续控制或整个时域。'+link('已完成回放与限制','docs/evidence/results/archcomp26_20261001/native_quad_allbox_adaptive_remainder_replay_20261004_001/README.md')+'。')
        if key=='single-pendulum-reach':add('x3 空栏专门保留固定 MATLAB 第三返回量的来源缺口；论文的两个物理态结果仅为具名两态 profile，不能凭 dx3=1 自行补第三初值和执行入口。')
        if key=='tora-remain':add('另有 h=0.05 四方全程补充合同，4800/4800 盒步均保存；T=20 的 x4 endpoint 并集宽度 P3/H/X/原生为 0.354314/0.408728/0.408728/0.346793。该步长变体不替换 h=0.1 主格。'+link('独立合同完整四态宽度与时间来源','docs/evidence/results/archcomp26_20261001/tora_remain_h005_fourway_saved_20261002/SUMMARY.md')+'。')
        if key=='airplane-continuous':add('64 个二分子盒各完成首个 0.01 秒 plant 步，其中 8 盒保存安全、56 盒 Unknown；高角子盒随后只完成 4/10 小步到 0.04 秒，第 5 步 x/y 自包含失败。这是另列数值诊断，不填主合同的 T=2 格，也不是实际不安全轨迹。')
        source_ids=sorted({sid for m in METHODS for s in states for sid in ws[key,m,s]['source_ids'].split(';') if sid})
        add('宽度来源编号：'+(', '.join(source_ids) if source_ids else '无有效主合同范围')+'；'+link('来源路径与大小','docs/evidence/results/archcomp26_report_20261005/widths/sources.json')+'、'+link('上下界和全部逐步宽度','docs/evidence/results/archcomp26_report_20261005/widths/widths_long.csv')+'、'+link('缺失逐格说明','docs/evidence/results/archcomp26_report_20261005/widths/missing_fields.csv')+'。')
    add('## 附录 A 新候选全记录与紧度尝试')
    add('13 个新阶段全部保留，不只展示选中的最快值。以下 driver 与 process 是每阶段单次；短前缀不能估成全程。')
    rows=[]
    for r in runs:
        rows.append([r['run_id'].replace('_20261005_001','')+'/'+r['stage'],str(r.get('completed_substeps') or '资格门 无 ODE'),sec(r.get('driver_elapsed_s')),sec(r.get('outer_wall_s'))])
    add(table(['新阶段','小步数','driver','process'],rows))
    add('QUAD two-slope 仅完成 40 步，不是完整 1000 步结果。x5 在 40 步 tube 和 endpoint 均较窄，但 x9 的 tube 有 37/40 步、endpoint 有 39/40 步更宽；最后一步 x1–x6 窄、x7–x11 宽、x12 相等。960 个不同状态/时刻/几何比较只有 392 项区间包含，不把“667 窄、80 相等、213 宽”跨单位求和当作总体优势。当前没有推广为默认，也没有宣称更紧的全程结果。'+link('候选与原始比较','research/p3_speed_tightness_20261005/README.md')+'。')
    add('## 附录 B 图形功能与未闭合证据')
    add('所有当前出图使用 Python；正式 CLI 能从已有 ranges.bin 或兼容几何导出初盒、tube/endpoint、按时间定义的 Safe/Target、同轴多方法 PNG/PDF 和 JSON/CSV。历史 MATLAB 文件只保留归档身份。本版表格与逐步数据没有删减这些接口所需字段。归档坐标盒只能画轴对齐投影，不能从中恢复相关八方向包络。')
    add('![论文 QUAD 的原始保存粒度](evidence/results/archcomp26_20261001/quad_paper_fourway_saved_20261002/quad_paper_fourway_t_x3_pooled_tube.png)')
    add('![Unicycle 论文合同保存范围](evidence/results/archcomp26_20261001/unicycle_paper_speed_w_constant_v1/plots/fourway_saved_20261002/unicycle_paper_speed_fourway_saved.png)')
    add('数值完整、作者标签、保存几何观察、独立端到端证书分别报告。Docking 数值全程但性质 UNKNOWN；SP 官方执行材料、Balancing 五特征网络和 Airplane discrete 转移顺序的缺口不能靠猜测补齐。Airplane continuous、Balancing raw4、DP more 和 TORA remain 两作者 h=0.1 的实际停点与原因见各节。当前四方均不据这些保存结果宣称获得独立端到端 NNCS 浮点证明。')
    add('## 附录 C 数据与报告入口')
    add('本版正文是唯一当前总报告。'+link('前一版正文快照','docs/ARCHCOMP26_REPORT_HISTORY_20261004.md')+'及旧日期 DOCX/PDF 保留历史身份，原始 START/RESULT 和旧 289 条索引没有被重写。报告数据、现有数值尝试与保存资格门分别列账，没有启动旧求解或旧检查器。')
    add('- '+link('完整报告包及可编辑 Word','docs/evidence/results/archcomp26_report_20261005/README.md'))
    add('- '+link('64 方法格状态与具体阻断','docs/evidence/results/archcomp26_report_20261005/blockers/README.zh.md'))
    add('- '+link('全部逐进程时间','docs/evidence/results/archcomp26_report_20261005/timing/runs.csv')+'；'+link('完整结构和来源字段','docs/evidence/results/archcomp26_report_20261005/timing/timing_index.json'))
    add('- '+link('364 行逐态宽度概要','docs/evidence/results/archcomp26_report_20261005/widths/summary.csv')+'；'+link('全部逐步上下界与宽度','docs/evidence/results/archcomp26_report_20261005/widths/widths_long.csv')+'；'+link('P3 对三方差值','docs/evidence/results/archcomp26_report_20261005/widths/pairwise_comparisons.csv'))
    SOURCE.write_text('\n\n'.join(lines)+'\n')
    target=OUT/'widths/pairwise_comparisons.csv'
    with target.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    (OUT/'widths/pairwise_comparisons.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2)+'\n')
    (OUT/'REPORT_ASSEMBLY.json').write_text(json.dumps(dict(instance_sections=16,status_cells=len(cells),width_summary_rows=len(width),pairwise_rows=len(comparisons),new_stages=len(runs),solver_runs=0,old_checker_runs=0,digest_operations=0,report_source=str(SOURCE.relative_to(ROOT))),indent=2)+'\n')
    print(f'Markdown: {len(lines)} blocks; {len(comparisons)} pairwise comparisons')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--docx',action='store_true');parser.add_argument('--render-dir',type=Path);args=parser.parse_args()
    # assemble() is defined below in the report-data adapter.
    if 'assemble' in globals(): assemble()
    if args.docx or args.render_dir:
        helper=docx_build()
        if args.render_dir:helper.render(args.render_dir)
