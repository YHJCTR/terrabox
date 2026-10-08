import sys, json
from pathlib import Path
sys.path.insert(0, 'tmp/plotenv')
import pandas as pd
import numpy as np
from plotnine import *
from mizani.formatters import percent_format
from matplotlib import rcParams
rcParams['pdf.fonttype']=42  # plotnine PDF backend: embed TrueType, avoid Type 3 fonts.

# Publication parameters: plotnine, vector PDF, no panel title, Wong colors.
TEXT_SIZE=8; LINE=0.65; WIDTH=180; WONG=['#E69F00','#56B4E9','#009E73','#CC79A7','#D55E00','#F0E442','#0072B2','#000000']; VIRIDIS='viridis'
OUT=Path('ExperienceEvo_LongCat_OEA_图表_20260917/论文级/figures_v2'); OUT.mkdir(parents=True,exist_ok=True)
ROOT=Path('tmp/trajectories')
METHODS={'Base':ROOT/'longcat_base_matched_v4clean_dockerfixed_20260818/standard','ExperienceEvo':ROOT/'experience_evo_v4_clean_oea_train2000_longcat_eval_dockerfixed_20260816/standard','Reflection':ROOT/'reflection_current_code_train2000_eval_20260805/standard','MemRL':ROOT/'memrl_current_code_train2000_eval_20260805/standard','ExpeL':ROOT/'expel_live_oea_train2000_longcat_eval_20260807/standard','ACE':ROOT/'ace_playbook_repro_oea_train2000_longcat_eval_20260810/standard'}
COLORS={m:WONG[i] for i,m in enumerate(METHODS)}

def load(d):
    rows=[]
    for f in (d/'results').glob('*.json'):
        try:
            x=json.loads(f.read_text()); z=x.get('metrics') or {}
            rows.append({'task_id':x.get('task_id',f.stem),'f1':float(z.get('f1',np.nan)),'multiset_f1':float(z.get('multiset_f1',np.nan)),'success':bool(x.get('success',False)),'status':x.get('status','unknown'),'calls':len(x.get('tool_calls',[]) or []),'tool_error':bool(x.get('has_tool_error',False)),'time_sec':float(x.get('time',np.nan))})
        except Exception: pass
    return pd.DataFrame(rows)
FR={m:load(d) for m,d in METHODS.items()}
theme_pub=theme_bw(base_size=TEXT_SIZE,base_family='Times New Roman')+theme(text=element_text(size=TEXT_SIZE),axis_text=element_text(size=TEXT_SIZE),axis_title=element_text(size=TEXT_SIZE),legend_text=element_text(size=TEXT_SIZE),legend_title=element_text(size=TEXT_SIZE),strip_text=element_text(size=TEXT_SIZE),plot_title=element_blank(),plot_subtitle=element_blank(),panel_grid_minor=element_blank(),panel_grid_major=element_line(color='#D9D9D9',size=.25),axis_line=element_line(color='#333333',size=.35),legend_background=element_blank(),legend_key=element_blank(),strip_background=element_rect(fill='white',color='#333333',size=.35),legend_position='bottom')
def save(p,name,h=78):
    if p.data is not None and 'method' in p.data:
        p.data['method']=pd.Categorical(p.data['method'],categories=list(METHODS),ordered=True)
    p=p+theme_pub+theme(figure_size=(WIDTH/25.4,h/25.4)); p.save(OUT/(name+'.pdf'),verbose=False,dpi=300); p.save(OUT/(name+'.png'),verbose=False,dpi=300)

# 6. Complete checkpoint log; do not splice abandoned runs into this history.
tc=Path('tmp/agent_rl_runs/rl_grpo/qwen25_3b_swift_grpo_catalogprompt_obplaceholder_turn6_resp128_train2000_autofallback_20260912/checkpoints/swift/v4-20260913-152155/checkpoint-1900/trainer_state.json')
t=pd.DataFrame(json.loads(tc.read_text())['log_history']).sort_values('step').drop_duplicates('step',keep='last')
assert len(t)==1900 and t.step.min()==1 and t.step.max()==1900
metrics={'reward':'(a) Reward','kl':'(b) KL','num_turns':'(c) Turns','grad_norm':'(d) Gradient norm','learning_rate':'(e) Learning rate','step_time':'(f) Step duration (s)','completions/clipped_ratio':'(g) Clipped completion fraction','frac_reward_zero_std':'(h) Zero-variance reward fraction'}
z=pd.concat([t[['step',k]].rename(columns={k:'value'}).assign(metric=v) for k,v in metrics.items()])
z['smooth']=z.groupby('metric')['value'].transform(lambda s:s.rolling(25,min_periods=1).mean())
save(ggplot(z,aes('step','value'))+geom_line(size=.23,alpha=.25,color=WONG[1])+geom_line(aes(y='smooth'),size=LINE,color=WONG[0])+facet_wrap('~metric',scales='free_y',ncol=2)+labs(x='Training step',y=''),'图6_训练过程多指标面板',205)
t.to_csv(OUT/'training_metrics.csv',index=False)

# 7. Matched task-level deltas.
b=FR['Base'].rename(columns={'f1':'base_f1','calls':'base_calls'}); e=FR['ExperienceEvo'].rename(columns={'f1':'evo_f1','calls':'evo_calls'}); pair=b[['task_id','base_f1','base_calls']].merge(e[['task_id','evo_f1','evo_calls']],on='task_id'); pair['Δ Set F1']=pair.evo_f1-pair.base_f1; pair['Δ tool calls']=pair.evo_calls-pair.base_calls
q=pair[['Δ Set F1','Δ tool calls']].melt(var_name='metric',value_name='delta'); save(ggplot(q,aes('delta',fill='metric'))+geom_histogram(bins=35,alpha=.75,color='white',size=.15)+geom_vline(xintercept=0,linetype='dashed',size=.4)+facet_wrap('~metric',scales='free',ncol=2)+scale_fill_manual(values=WONG[:2])+labs(x='ExperienceEvo − Base',y='Number of tasks',fill=''),'图7_逐任务性能增益分布')

# 8. Cumulative task success.
cc=[]
for m,d in FR.items():
    z=d[['task_id','success']].copy(); z['sort_key']=z.task_id.str.extract(r'(\d+)$').astype(int); z=z.sort_values('sort_key'); z['order']=np.arange(1,len(z)+1); z['cum_success']=z.success.astype(int).cumsum()/z.order; z['method']=m; cc.append(z[['order','cum_success','method']])
cc=pd.concat(cc); save(ggplot(cc,aes('order','cum_success',color='method'))+geom_line(size=LINE)+scale_color_manual(values=COLORS)+scale_y_continuous(labels=percent_format())+labs(x='Sorted task index',y='Cumulative success rate',color=''),'图8_方法级累积成功率')

# 9. Outcome composition.
rows=[]
for m,d in FR.items():
    def cat(r):
        if r.success:return 'Successful'
        if r.status=='failed':return 'Failed execution'
        if r.tool_error:return 'Tool-error / incomplete'
        return 'Completed without success'
    for k,v in d.apply(cat,axis=1).value_counts(normalize=True).items(): rows.append({'method':m,'category':k,'fraction':v})
co=pd.DataFrame(rows); order=['Successful','Completed without success','Tool-error / incomplete','Failed execution']; co['category']=pd.Categorical(co.category,categories=order,ordered=True)
save(ggplot(co,aes('method','fraction',fill='category'))+geom_col(width=.72)+scale_y_continuous(labels=percent_format())+scale_fill_manual(values=WONG[:4],drop=False)+labs(x='',y='Fraction of tasks',fill='Outcome'),'图9_任务结果构成')

# 10. Pareto-style success / call cost.
ss=pd.DataFrame([{'method':m,'success':d.success.mean(),'calls':d.calls.mean()} for m,d in FR.items()]); save(ggplot(ss,aes('calls','success',color='method'))+geom_point(size=3)+scale_color_manual(values=COLORS)+scale_y_continuous(labels=percent_format())+labs(x='Mean tool calls per task',y='Success rate',color=''),'图10_成功率与工具调用效率')

# 11. ExperienceEvo decision-step behavior.
rows=[]
for f in (METHODS['ExperienceEvo']/'results').glob('*.json'):
    try:
        x=json.loads(f.read_text())
        for t in x.get('evolution_trace') or []:
            rec=t.get('recommended_tools') or []; sel=t.get('selected_tool'); rows.append({'step':int(t.get('step',0)),'has_recommendation':bool(rec),'adopts':bool(sel and sel in rec),'error':t.get('tool_result_status') not in (None,'success')})
    except Exception: pass
st=pd.DataFrame(rows)
if len(st):
    counts=st.groupby('step').agg(n=('step','size'),has_recommendation=('has_recommendation','mean'),adopts=('adopts','mean')).reset_index()
    g=counts.melt('step',var_name='metric',value_name='value'); g['metric']=g.metric.map({'has_recommendation':'(a) Recommendation coverage','adopts':'(b) Adoption / all logged decisions','n':'(c) Logged decisions (n)'})
    save(ggplot(g,aes('step','value'))+geom_line(size=LINE,color=WONG[0])+geom_point(size=1.3,color=WONG[0])+facet_wrap('~metric',scales='free_y',ncol=3)+labs(x='Decision step (conditional on reaching that step)',y=''),'图11_经验注入逐步行为',78)
    counts.to_csv(OUT/'decision_step_counts.csv',index=False)

# 12. Task-level distributions.
lo=pd.concat([d[['f1','calls']].assign(method=m) for m,d in FR.items()]).melt(['method'],var_name='metric',value_name='value'); lo['metric']=lo.metric.map({'f1':'Set F1','calls':'Tool calls'})
save(ggplot(lo,aes('method','value',fill='method'))+geom_boxplot(outlier_alpha=.12,width=.65)+facet_wrap('~metric',scales='free_y',ncol=2)+scale_x_discrete(labels=lambda xs:['Experience\nEvo' if x=='ExperienceEvo' else x for x in xs])+scale_fill_manual(values=COLORS)+labs(x='',y='Task-level value',fill=''),'图12_任务级指标分布',92)

meta={'figures':{'图6':'完整 checkpoint-1900 的 reward/KL/turns/梯度范数/学习率/单步耗时/截断比例/组内奖励无差异比例；浅线原始值，橙线为后向25步均值。','图7':'Matched task-level ExperienceEvo minus Base deltas','图8':'Cumulative success by numeric task id; not a temporal learning curve','图9':'Outcome composition from status/success/tool-error','图10':'Method success versus mean tool calls','图11':'ExperienceEvo decision-step recommendation/adoption/error rates; conditional on reaching that step','图12':'Task-level Set F1 and call-count distributions'},'caveats':['不同方法历史运行配置并不完全一致，主结论以 matched Base/ExperienceEvo 为准。MemRL 是 label-augmented adapted 对照。','未记录 PPL/MFU，不推算未知吞吐量。学习率重启和恢复阶段需要披露。','ExperienceEvo 推荐采纳率不等于任务成功率。success 是保存结果中的 success 标记，不是独立答案 judge 准确率。']}

# 13. Inference prefix curves: fixed task denominator, stopped trajectories carry forward.
from collections import Counter
curves=[]; raw={}; audit={}
for method,path in METHODS.items():
    data={}
    for f in (path/'results').glob('*.json'):
        x=json.loads(f.read_text()); assert x['task_id'] not in data; data[x['task_id']]=x
    raw[method]=data
    report=json.loads((path/'report.json').read_text())
    assert len(data)==report['total_tasks']==1162
    assert abs(FR[method].success.mean()-report['success_rate'])<1e-8
    assert abs(FR[method].f1.mean()-report['avg_f1'])<1e-8
    audit[method]={'directory':str(path),'n':len(data),'mean_f1':FR[method].f1.mean(),'success':FR[method].success.mean()}
common=set.intersection(*(set(d) for d in raw.values()))
assert len(common)==1162
for method,data in raw.items():
    for k in range(0,21):
        f1s=[]; recalls=[]
        for tid in sorted(common):
            x=data[tid]; gold=set(x['expected_tools']); actual=set(x['tool_calls'][:k]); hit=len(gold&actual)
            f1s.append(2*hit/(len(gold)+len(actual)) if gold or actual else 1.)
            if gold: recalls.append(hit/len(gold))
        curves.extend([{'method':method,'calls':k,'value':np.mean(f1s),'metric':'(a) Prefix Set F1'},{'method':method,'calls':k,'value':np.mean(recalls),'metric':'(b) Expected-tool recall'}])
for tid in common:
    a=raw['Base'][tid]; b=raw['ExperienceEvo'][tid]
    assert a['question']==b['question'] and a['expected_tools']==b['expected_tools'] and set(a['allowed_slugs'])==set(b['allowed_slugs'])
cv=pd.DataFrame(curves)
save(ggplot(cv,aes('calls','value',color='method'))+geom_line(size=LINE)+facet_wrap('~metric',ncol=2)+scale_color_manual(values=COLORS)+scale_x_continuous(breaks=[0,5,10,15,20])+labs(x='Tool-call budget (stopped trajectories carried forward)',y='Mean over all tasks',color=''),'图13_推理工具预算与链路覆盖',88)
cv.to_csv(OUT/'inference_prefix_metrics.csv',index=False)

# 14. Paired task bootstrap CIs, conditional on reference-chain length.
rng=np.random.default_rng(42); rr=[]
for name,selector in [('All tasks',lambda n:True),('1-2 calls',lambda n:n<=2),('3-4 calls',lambda n:3<=n<=4),('5+ calls',lambda n:n>=5)]:
    ids=[tid for tid in sorted(common) if selector(len(raw['Base'][tid]['expected_tools']))]
    for label,field,scale in [('(a) Success gain (pp)','success',100),('(b) Set F1 gain','f1',1),('(c) Calls saved','calls',1)]:
        vals=[]
        for tid in ids:
            a,b=raw['Base'][tid],raw['ExperienceEvo'][tid]
            val=(int(b['success'])-int(a['success']))*scale if field=='success' else (b['metrics']['f1']-a['metrics']['f1'] if field=='f1' else len(a['tool_calls'])-len(b['tool_calls']))
            vals.append(val)
        vals=np.array(vals); means=np.array([rng.choice(vals,len(vals),replace=True).mean() for _ in range(2000)])
        rr.append({'group':f'{name}\n(n={len(ids)})','metric':label,'mean':vals.mean(),'lo':np.quantile(means,.025),'hi':np.quantile(means,.975)})
ci=pd.DataFrame(rr); ci['group']=pd.Categorical(ci.group,categories=list(dict.fromkeys(ci.group)),ordered=True)
save(ggplot(ci,aes('group','mean'))+geom_hline(yintercept=0,linetype='dashed',size=.3)+geom_errorbar(aes(ymin='lo',ymax='hi'),width=.15,color=WONG[0])+geom_point(color=WONG[0],size=2)+facet_wrap('~metric',scales='free_y',ncol=3)+labs(x='Reference tool-chain length (post-hoc strata)',y='Paired difference with 95% bootstrap CI'),'图14_工具链长度分层配对增益',85)
ci.to_csv(OUT/'paired_bootstrap.csv',index=False)
pair.to_csv(OUT/'paired_task_deltas.csv',index=False)
pd.concat([d.assign(method=m) for m,d in FR.items()]).to_csv(OUT/'all_task_metrics.csv',index=False)
meta['figures']['图13']='真实工具序列前缀的 Set F1 / expected-tool recall；停止轨迹保持最终工具序列，全体任务固定分母。gold 只在离线统计时使用。'
meta['figures']['图14']='按 expected tools 链长分层的配对增益和2000次任务bootstrap 95%区间（seed=42），不反映跨训练seed方差。'
meta['audit']=audit
meta['training_source']=str(tc)
train_log=pd.read_csv(OUT/'training_metrics.csv')
meta['training_first_last_100']={k:{'first100':float(train_log[k].head(100).mean()),'last100':float(train_log[k].tail(100).mean())} for k in metrics}
meta['figures']['图11']='逐决策步推荐覆盖率、全决策分母的采纳率和样本数；只对到达该步的轨迹统计，含 final 决策。'
(OUT.parent/'图表说明_v2.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print('generated',len(list(OUT.glob('*.pdf'))),'figures')
