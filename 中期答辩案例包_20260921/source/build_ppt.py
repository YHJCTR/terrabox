from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

src='iOPEN-PPT模板-宽屏模板v1.1.pptx'; out='ExperienceEvo_PromptEvo_中期答辩_模板风格副本.pptx'
p=Presentation(src)
for _ in range(len(p.slides)): p.slides._sldIdLst.remove(p.slides._sldIdLst[-1])
L=p.slide_layouts; NAVY=RGBColor(29,57,91); BLUE=RGBColor(48,104,160); CYAN=RGBColor(65,153,171); RED=RGBColor(188,75,66); GOLD=RGBColor(213,151,55); TEXT=RGBColor(42,48,55); LIGHT=RGBColor(239,245,249); PALE=RGBColor(248,250,252); GREEN=RGBColor(71,133,92); FONT='Microsoft YaHei'
def tb(s,x,y,w,h,t,z=18,c=TEXT,b=False,a=PP_ALIGN.LEFT,fill=None,line=None):
 sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)); f=sh.text_frame; f.clear(); f.word_wrap=True; f.margin_left=f.margin_right=Inches(.08); f.vertical_anchor=MSO_ANCHOR.MIDDLE; q=f.paragraphs[0]; q.alignment=a; r=q.add_run(); r.text=t; r.font.name=FONT; r.font.size=Pt(z); r.font.bold=b; r.font.color.rgb=c
 if fill: sh.fill.solid(); sh.fill.fore_color.rgb=fill
 else: sh.fill.background()
 if line: sh.line.color.rgb=line
 else: sh.line.fill.background()
 return sh
def bx(s,x,y,w,h,t,fill=LIGHT,line=BLUE,z=15,c=TEXT,b=False):
 sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h)); sh.fill.solid(); sh.fill.fore_color.rgb=fill; sh.line.color.rgb=line; sh.line.width=Pt(1.2); f=sh.text_frame; f.clear(); f.word_wrap=True; f.vertical_anchor=MSO_ANCHOR.MIDDLE; f.margin_left=f.margin_right=Inches(.1); q=f.paragraphs[0]; q.alignment=PP_ALIGN.CENTER; r=q.add_run(); r.text=t; r.font.name=FONT; r.font.size=Pt(z); r.font.bold=b; r.font.color.rgb=c; return sh
def ar(s,x1,y1,x2,y2,c=BLUE):
 q=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2)); q.line.color.rgb=c; q.line.width=Pt(2); q.line.end_arrowhead=True
def ti(s,t,sub,n):
 tb(s,.62,.28,11,.5,t,25,NAVY,True); q=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(.62),Inches(.86),Inches(1.05),Inches(.05)); q.fill.solid(); q.fill.fore_color.rgb=CYAN; q.line.fill.background(); tb(s,.62,.93,11.5,.3,sub,10,RGBColor(100,110,120)); tb(s,11.85,7.05,.8,.22,f'{n}/10',9,RGBColor(105,110,115),False,PP_ALIGN.RIGHT); tb(s,.62,7.04,6,.2,'西北工业大学 | 光电与智能研究院 (iOPEN)',8,RGBColor(120,125,130))
# 1
s=p.slides.add_slide(L[0]); tb(s,1,1.55,11.3,1,'地理空间 Agent 的经验与提示词自进化',31,NAVY,True,PP_ALIGN.CENTER); tb(s,1.25,2.65,10.8,.55,'ExperienceEvo & PromptEvo：从可复用经验到可回滚协议补丁',18,BLUE,False,PP_ALIGN.CENTER); tb(s,2,5.5,9.3,.4,'中期答辩  |  汇报人：________',15,TEXT,False,PP_ALIGN.CENTER); tb(s,2,5.95,9.3,.3,'西北工业大学 · 光电与智能研究院 (iOPEN)',11,RGBColor(100,110,120),False,PP_ALIGN.CENTER)
# 2
s=p.slides.add_slide(L[0]); ti(s,'汇报目录','研究问题—方法—案例—实验—下一步',2)
for i,(n,h,d) in enumerate([('01','研究背景与问题','多工具、多轮任务中的经验复用与提示词回归'),('02','ExperienceEvo','产物状态转移、检索注入与边界自进化'),('03','PromptEvo','Stage1/Stage2 对比归因与 typed patch'),('04','实验与案例','OEA、ToolBench、API-Bank、AgentDojo、RL'),('05','总结与下一步','产物来源感知信用分配 GRPO')]):
 y=1.45+i*.92; bx(s,.95,y,.72,.48,n,NAVY,NAVY,16,RGBColor(255,255,255),True); tb(s,1.9,y-.02,3.1,.35,h,18,NAVY,True); tb(s,5,y,6.9,.38,d,13)
# 3
s=p.slides.add_slide(L[0]); ti(s,'1 研究背景：工具成功不等于任务正确','地理空间 Agent 的真实工具闭环',3); bx(s,.75,1.45,3.55,3.85,'用户任务\n\n跨模态感知\nGIS / OSM 分析\n多轮工具链',PALE,BLUE,20,NAVY,True); ar(s,4.45,3.35,5.05,3.35); bx(s,5.2,1.45,3.2,3.85,'真实工具执行\n\n成功返回 ≠\n输入产物绑定正确\n\n成功返回 ≠\n输出适合后续消费',RGBColor(255,247,238),GOLD,18,TEXT,True); ar(s,8.55,3.35,9.15,3.35); bx(s,9.3,1.45,3.25,3.85,'核心问题\n\n经验如何复用？\n提示词如何演化？\n错误如何归因？\n\n→ 可验证、可回滚',RGBColor(242,248,243),GREEN,18,NAVY,True); tb(s,.9,5.75,11.6,.62,'数据不均衡、长链路信用稀疏、工具错误与语义绑定错误混杂，使“平均成功率”难以解释方法为何有效。',15,RED,True,PP_ALIGN.CENTER)
# 4
s=p.slides.add_slide(L[0]); ti(s,'2 ExperienceEvo：产物状态转移经验自进化','把历史轨迹压缩为可验证的“输入产物—工具—输出产物”',4); xs=[.65,2.75,4.85,6.95,9.05,11.15]; ls=['真实 rollout','轨迹抽取','产物状态\n转移经验','Q/N/R +\n风险排序','Top-K 注入','当前状态\n验证与停止']
for i,(x,l) in enumerate(zip(xs,ls)): bx(s,x,2.45,1.45,1.25,l,LIGHT if i<4 else RGBColor(242,248,243),BLUE if i<4 else GREEN,13,NAVY,True); ar(s,x+1.5,3.08,xs[i+1]-.08,3.08) if i<len(xs)-1 else None
tb(s,.9,4.35,11.6,.65,'经验不是复制历史答案，而是根据当前 product_state 检索下一步可行转移；每次工具返回后更新状态，并由 verifier 判断证据是否足够。',15,TEXT,False,PP_ALIGN.CENTER); bx(s,1.2,5.35,3.15,.75,'经验表示：C → Tool → O',NAVY,NAVY,16,RGBColor(255,255,255),True); bx(s,5.1,5.35,3.15,.75,'运行时绑定当前 artifact',BLUE,BLUE,16,RGBColor(255,255,255),True); bx(s,9,5.35,3.15,.75,'边界版本：反例 → 条件修正',GREEN,GREEN,16,RGBColor(255,255,255),True)
# 5
s=p.slides.add_slide(L[0]); ti(s,'3 ExperienceEvo 边界自进化：控制经验负迁移','独立模式，可回退到 v4-clean',5); xs=[.85,4,7.15,10.3]; ls=['经验被使用','构造反例\n语义保持 / 语义改变','确定性审核\n输入 / 输出契约','更新边界\n补条件 / 分裂 / 停用'];
for i,(x,l) in enumerate(zip(xs,ls)): bx(s,x,1.55,2.8 if i<3 else 2.2,1.1,l,LIGHT if i in [0,2] else RGBColor(255,247,238) if i==1 else RGBColor(242,248,243),BLUE if i in [0,2] else GOLD if i==1 else GREEN,15,NAVY,True); ar(s,x+(2.8 if i<3 else 2.2),2.1,xs[i+1]-.08,2.1) if i<3 else None
ar(s,11.4,2.75,11.4,4.2,GREEN); ar(s,11.4,4.2,2.2,4.2,GREEN); ar(s,2.2,4.2,2.2,2.75,GREEN); tb(s,1,4.45,11.2,.55,'闭环：使用经验 → 发现反例 → 学习适用条件 → 运行时过滤 → 减少跨任务负迁移',16,GREEN,True,PP_ALIGN.CENTER); tb(s,1.15,5.35,11,.75,'【图示占位】Boundary v1 pipeline 图\n独立 store + boundary rules + runtime filter/fallback',13,RGBColor(115,115,115),False,PP_ALIGN.CENTER,PALE,RGBColor(190,195,200))
# 6
s=p.slides.add_slide(L[0]); ti(s,'4 PromptEvo：从失败轨迹到可回滚协议补丁','Stage1 生成候选，Stage2 做对比归因与回归门控',6); xs=[.75,3.85,6.95,10.05]; ls=['失败轨迹\n重复 / 缺参 / 工具错误','Stage1\n候选规则 patch','Stage2\nBase vs Candidate\n对比归因','typed patch\n编译 + dev gate\naccept / rollback'];
for i,(x,l) in enumerate(zip(xs,ls)): bx(s,x,1.45,2.45 if i<3 else 2.5,1.1,l,RGBColor(255,247,238) if i==0 else LIGHT if i<3 else NAVY,RED if i==0 else BLUE if i<3 else NAVY,15,RGBColor(255,255,255) if i==3 else NAVY,True); ar(s,x+(2.45 if i<3 else 2.5),2,xs[i+1]-.08,2) if i<3 else None
tb(s,.95,3.3,11.5,.5,'Patch 不是自由改写完整 system prompt，而是带 trigger / scope / condition / priority 的最小协议增量。',16,TEXT,False,PP_ALIGN.CENTER); bx(s,1.2,4.3,3.2,1.15,'保护目标\n减少重复调用、缺参、无终止',LIGHT,BLUE,16,NAVY,True); bx(s,5.05,4.3,3.2,1.15,'对比证据\npaired trace + 行为变化归因',LIGHT,BLUE,16,NAVY,True); bx(s,8.9,4.3,3.2,1.15,'失败处理\n拒绝候选或回滚，不污染 Base',RGBColor(255,247,238),GOLD,16,NAVY,True)
# 7 case
s=p.slides.add_slide(L[0]); ti(s,'5 真实案例：从无证据循环到当前产物闭环','同一 OEA 任务，展示方法如何改变工具链',7); tb(s,.85,1.25,5.7,.35,'Base：15 次调用，最终 max-turn 失败',17,RED,True); tb(s,6.85,1.25,5.7,.35,'ExperienceEvo：1 次 count，证据可回答',17,GREEN,True); bx(s,.85,1.85,5.55,2.9,'OCR\n→ VLM × 13\n→ count_given_object\n→ VLM\n\n多次观察互相矛盾\n299.65s | F1=.333 | failed',RGBColor(255,247,238),RED,17,TEXT); bx(s,6.85,1.85,5.55,2.9,'当前状态：input:image\n→ count_given_object\n→ result: count=1\n→ verifier=likely_ready\n\n33.80s | 1 call | completed',RGBColor(242,248,243),GREEN,17,TEXT); tb(s,1,5.25,11.3,.85,'案例原始 JSON、完整 conversation_history、bbox 和置信度已随材料包提供。可用 TG_40010.jpg 替换此页占位。',14,RGBColor(105,110,115),False,PP_ALIGN.CENTER,PALE,RGBColor(190,195,200))
# 8 metrics
s=p.slides.add_slide(L[0]); ti(s,'6 指标对比：PromptEvo 与 RL 的真实结果','主表只展示可追溯口径；污染/未完成实验单独标注',8); tb(s,.9,1.35,5.4,.3,'ToolBench strict paper split（n=160）',15,NAVY,True); bars=[('Base',86.25,BLUE),('PromptEvo Stage1',88.12,GREEN),('PromptEvo Stage2',84.38,RED)]
for i,(l,v,c) in enumerate(bars): y=1.85+i*.58; tb(s,.95,y,1.55,.25,l,11); q=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(2.55),Inches(y+.02),Inches(v/100*3.5),Inches(.28)); q.fill.solid(); q.fill.fore_color.rgb=c; q.line.fill.background(); tb(s,6.2,y,1,.25,f'{v:.2f}%',11,NAVY,True)
tb(s,7.15,1.35,5.2,.3,'OEA RL full test（n=1162）',15,NAVY,True); bars=[('Qwen3B Base',71.86,BLUE),('Pure GRPO',70.65,RED),('GRPO + ExperienceEvo',73.75,GREEN)]
for i,(l,v,c) in enumerate(bars): y=1.85+i*.58; tb(s,7.2,y,1.75,.25,l,11); q=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,Inches(9),Inches(y+.02),Inches(v/100*3),Inches(.28)); q.fill.solid(); q.fill.fore_color.rgb=c; q.line.fill.background(); tb(s,12.15,y,.65,.25,f'{v:.2f}%',11,NAVY,True,PP_ALIGN.RIGHT)
bx(s,.9,4.05,3.55,1.25,'PromptEvo Stage1\nSuccess +1.87 pp\nTool error 14.37% → 7.50%',RGBColor(242,248,243),GREEN,16,NAVY,True); bx(s,4.9,4.05,3.55,1.25,'RL + ExperienceEvo\nSuccess +1.89 pp vs Base\nTool error 48.45% → 34.17%',RGBColor(242,248,243),GREEN,15,NAVY,True); bx(s,8.9,4.05,3.55,1.25,'边界与限制\nSet-F1 基本持平\nStage2 / 污染 eval 不作主结论',RGBColor(255,247,238),GOLD,15,NAVY,True); tb(s,.9,5.85,11.6,.55,'【图表占位】可替换为案例包 tables/ 中 CSV 生成的论文风格指标图。',13,RGBColor(115,115,115),False,PP_ALIGN.CENTER,PALE,RGBColor(190,195,200))
# 9 rl
s=p.slides.add_slide(L[0]); ti(s,'7 下一步：产物来源感知的局部信用分配 GRPO','保留 GRPO 组内优势，把信用从整条轨迹细化到工具动作',9); xs=[.75,4.15,7.7,11.1]; ls=['完整 rollout\nEpisode reward Rᵢ','产物依赖图\n来源 / 语义 / 下游消费','局部信用 cᵢ,ₜ\n同前缀动作对比','Ãᵢ,ₜ\n对齐 token'];
for i,(x,l) in enumerate(zip(xs,ls)): bx(s,x,1.55,2.55 if i<3 else 1.5,1,l,LIGHT if i<2 else RGBColor(255,247,238) if i==2 else NAVY,BLUE if i<2 else GOLD if i==2 else NAVY,14,RGBColor(255,255,255) if i==3 else NAVY,True); ar(s,x+(2.55 if i<3 else 1.5),2.05,xs[i+1]-.08,2.05) if i<3 else None
tb(s,.95,3.05,11.5,.8,'普通 GRPO：同一 rollout 的 assistant token 共享 Aᵢ。\n方案 A：保留 Aᵢ，并根据产物来源、输出契约和下游消费得到 δᵢ,ₜ，只作用于对应工具调用 token。',16,TEXT,False,PP_ALIGN.CENTER); tb(s,1,4.35,11.4,1,'cᵢ,ₜ = wₛSᵢ,ₜ + wₒOᵢ,ₜ + wₚPᵢ,ₜ − wₙNᵢ,ₜ\nÃᵢ,ₜ = (1−λcᵢ,ₜ)Aᵢ + λcᵢ,ₜ · clip(Dᵢ,ₜ/s, −aₘₐₓ, aₘₐₓ)',15,BLUE,True,PP_ALIGN.CENTER,PALE,BLUE); tb(s,1,5.85,11.4,.55,'当前为待验证算法方案；需要 veRL 级别的 action-span 对齐和标准 GRPO / 过程奖励 / 动作级信用消融。',13,RED,True,PP_ALIGN.CENTER)
# 10
s=p.slides.add_slide(L[0]); tb(s,1,1.4,11.3,.8,'总结与展望',31,NAVY,True,PP_ALIGN.CENTER); tb(s,1.2,2.7,10.9,1.4,'ExperienceEvo：让工具轨迹变成可验证的产物状态经验\nPromptEvo：让失败轨迹变成可回滚的协议 patch\n下一步：把产物语义绑定引入 Agentic RL 的局部信用分配',21,TEXT,False,PP_ALIGN.CENTER); tb(s,2,5.35,9.3,.45,'谢谢各位老师，请批评指正',18,BLUE,True,PP_ALIGN.CENTER); tb(s,2,5.9,9.3,.3,'西北工业大学 · 光电与智能研究院 (iOPEN)',11,RGBColor(100,110,120),False,PP_ALIGN.CENTER)
p.save(out); print(out,len(p.slides))
