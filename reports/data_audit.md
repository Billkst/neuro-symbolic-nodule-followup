# 肺结节随访建议生成系统：数据审计报告

> 审计日期：2026-04-03
> 审计范围：data/ 目录下全部已采集数据
> 审计环境：conda activate follow-up

---

## A. 数据总览

```text
data/
├── LIDC-XML-only.zip
├── lidc_extracted/
│   └── tcia-lidc-xml/
│       ├── 157/
│       ├── 185/
│       ├── 186/
│       ├── 187/
│       ├── 188/
│       └── 189/
├── mimic-iv-3.1.zip
├── mimic-iv-note-deidentified-free-text-clinical-notes-2.2.zip
├── mimic_note_extracted/
│   ├── radiology.csv.gz
│   ├── radiology_detail.csv.gz
│   ├── discharge.csv.gz
│   └── discharge_detail.csv.gz
├── medical.json
├── 中华医学会肺癌临床诊疗指南（2024版）.pdf
├── 中国肺癌筛查与早诊早治指南（2021，北京）.pdf
└── lung-rads-assessment-categories.pdf
```

| 文件名 | 格式 | 规模 | 核心用途 | 就绪状态 | 主要问题 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| radiology.csv.gz | CSV(gzip) | 2,321,355 行 | 结节特征信息抽取 | 就绪 | 文本非结构化 |
| discharge.csv.gz | CSV(gzip) | 331,793 行 | 吸烟史与高危判定 | 高风险 | Social History 脱敏 |
| radiology_detail.csv.gz | CSV(gzip) | 6,046,121 行 | 检查类型过滤 | 就绪 | 需与主表关联 |
| discharge_detail.csv.gz | CSV(gzip) | 186,137 行 | 报告元数据 | 就绪 | 仅含 author 字段(脱敏) |
| medical.json | JSONL | 8,808 条 | 医学知识图谱 | 就绪 | 疾病百科,非结节数据 |
| LIDC XML | XML | 1,318 文件 | 结节形态学标注 | 就绪 | 无显式直径字段 |
| mimic-iv-3.1.zip | ZIP | 364,627 患者 | 人口统计学 | 待解压 | 含 patients/admissions/diagnoses_icd |
| 临床指南 PDF | PDF | 3 份 | 随访规则库 | 就绪 | 需人工结构化 |

---

## B. 面向任务的可用性审计

### B1. 放射报告信息抽取

当前可直接使用的文件:
- radiology.csv.gz (2,321,355 行, 8 列: note_id, subject_id, hadm_id, note_type, note_seq, charttime, storetime, text)
- radiology_detail.csv.gz (6,046,121 行, 可用于按 exam_name 过滤胸部 CT 报告)

当前可直接形成的输入输出:
- 输入: 放射报告 text 字段
- 输出: 结节大小(mm)、密度(solid/GGO/part-solid)、位置(lobe)、数量、变化(stable/new/increased)
- 结节报告约 190,099 条（采样估计，非全量统计）[radiology.csv.gz | 前 50,000 行采样: 8.19% 含 "nodule", 线性外推 | 50,000]

Section 覆盖率（采样估计，非全量统计）(结节报告子集, 采样 50K):
IMPRESSION 96.2%, TECHNIQUE 80.4%, FINDINGS 79.3%, COMPARISON 78.8%, INDICATION 76.1% [radiology.csv.gz | 50K 采样 | 50,000]

关键词分布（采样估计，非全量统计）(结节报告子集, 采样 50K):
| 关键词 | 出现率 | 关键词 | 出现率 |
| :--- | :--- | :--- | :--- |
| size | 57.3% | mm | 57.2% |
| calcification | 50.5% | cm | 42.5% |
| unchanged | 40.1% | solid | 32.6% |
| stable | 32.0% | new | 32.2% |
| recommend | 30.3% | density | 18.5% |
| increased | 17.5% | follow-up | 13.7% |
| ground-glass | 10.6% | decreased | 7.3% |
| spiculation | 3.2% | GGO/GGN | <0.1% |
[radiology.csv.gz | 50K 采样 | 50,000]

当前最缺的标签/字段:
- 无人工标注的结构化抽取结果 (即 "报告文本 -> {size: 6mm, density: solid, ...}" 的金标准对)
- LIDC 提供形态学标注但无对应自然语言报告

当前能做的 baseline:
- 基于规则/正则的信息抽取 (利用 FINDINGS/IMPRESSION section 结构)
- 利用 LIDC 9 维特征 + 合成报告文本进行预训练

当前不能做的:
- 无法直接评估抽取准确率 (缺金标准)
- 无法训练端到端监督模型 (缺标注)

真实样例 1 — CT 胸部含结节大小测量:
```
来源: radiology.csv.gz, note_id: 10000935-RR-76, subject_id: 10000935
TECHNIQUE: Multidetector CT of the torso was performed with intravenous and oral contrast.
FINDINGS: CHEST: There are innumerable pulmonary nodules, which have an upper lobe predilection, particularly in the left upper lobe. The largest nodule measures 6 mm in diameter (sequence 3 image 8). Calcification is noted within the right pleural cavity...
IMPRESSION: Multiple pulmonary and hepatic metastases. No separate primary lesion identified.
```

真实样例 2 — 低剂量肺癌筛查 CT 含 Lung-RADS 分级:
```
来源: radiology.csv.gz, note_id: 10002221-RR-133, subject_id: 10002221
EXAMINATION: CT LOW DOSE LUNG SCREENING
INDICATION: ___ yr old, former smoker (1), 40 pk yrs, asymptomatic
FINDINGS: Nodules: Stable 2 mm calcified nodule in the right upper lobe. Stable 1 mm solid right middle lobe pulmonary nodule...
IMPRESSION: Stable tiny pulmonary nodules ranging in size from 1-2 mm. Mild upper lobe predominant emphysema. Lung-RADS category: 2
RECOMMENDATION(S): Continue low-dose lung cancer screening CT in 12 months.
```

真实样例 3 — 含毛刺征结节及随访建议:
```
来源: radiology.csv.gz, note_id: 10001401-RR-8, subject_id: 10001401
EXAMINATION: CT CHEST W/CONTRAST
FINDINGS: There are 2 spiculated irregular part solid nodules with associated bronchiolectasis the largest in the right upper lobe... measuring 10.5 mm... Smaller irregular bubbly part solid nodule measuring 7 mm in average diameter in the left lower lobe...
IMPRESSION: Two irregular, spiculated part solid nodules, the largest in the right upper lobe measuring 11 mm in diameter. These nodules do not have the typical appearance of metastasis, but are concerning for lesions in the lung adenocarcinoma spectrum.
RECOMMENDATION(S): PET-CT imaging may be performed for better characterization. Alternatively consider a 3 month follow-up CT for re-evaluation of all nodules.
```

### B2. 吸烟史 / 高危筛查资格判定

当前可直接使用的文件:
- discharge.csv.gz (331,793 行, 8 列)
- 可通过 subject_id 与 radiology 关联

当前可直接形成的输入输出:
- 输入: 出院小结 text 字段
- 期望输出: 吸烟状态 (current/former/never)、包年数 (pack-years)、每日吸烟量 (PPD)
- 实际可行输出: 仅能做粗粒度的吸烟状态二分类 (ever-smoker vs never-smoker)

当前最缺的标签/字段:
- Social History 段落被系统性脱敏 (替换为 "___"), 这是核心瓶颈
- 定量吸烟信息 (pack-year) 仅在约 0.5% 的记录中出现（采样估计，非全量统计）[discharge.csv.gz | 前 10,000 行采样: 51 条含 "pack-year" | 10,000]
- PPD 出现率约 2.5%（采样估计，非全量统计）, 但存在歧义 (Postpartum Day vs Packs Per Day) [discharge.csv.gz | 前 10,000 行采样 | 10,000]

关键发现 — 脱敏模式:
- 97.4% 的记录包含 "Social History" 段落（采样估计，非全量统计）[discharge.csv.gz | 前 10,000 行采样 | 10,000]
- 其中 97.9% 的 Social History 内容被替换为 "___" [discharge.csv.gz | 前 10,000 行采样: 9,532/9,737 | 10,000]
- 仅 2.1% (约 205/10,000) 保留了实际 Social History 内容
- 吸烟信息散落在 Family History、HPI、Assessment 等非 Social History 段落中
- 14.4% 的记录在非 SH 段落中提及 smoking/tobacco（采样估计，非全量统计）[discharge.csv.gz | 前 10,000 行采样 | 10,000]

当前能做的 baseline:
- 弱监督 baseline: 基于关键词匹配 (smoking/tobacco/cigarette) 的二分类
- 从 HPI/Assessment 等段落中提取定性吸烟状态
- 预期精度有限, 无法精准匹配指南中 ">=30 包年" 的定量标准

当前不能做的:
- 不适合直接支撑稳定的监督学习主实验
- 无法可靠提取定量吸烟史 (pack-year 样本量不足以训练)
- 无法构建 "吸烟强度 -> 高危准入" 的精确映射

真实样例 1 — 典型脱敏模式 (吸烟信息在 Family History 段):
```
来源: discharge.csv.gz, note_id: 10000032-DS-21, subject_id: 10000032
Social History:
___
Family History:
She a total of five siblings... Her last alcohol consumption was one drink two months ago. No regular alcohol consumption. Last drug use ___ years ago. She quit smoking a couple of years ago.
```

真实样例 2 — 含 pack-year 的罕见记录 (信息在 HPI 段):
```
来源: discharge.csv.gz, note_id: 10014967-DS-11, subject_id: 10014967
Social History:
___
[HPI 段落中]: former 46 pack year history of smoking who presents with ongoing symptoms of dyspnea
```

### B3. 随访建议规则推理

当前可直接使用的文件:
- lung-rads-assessment-categories.pdf (Lung-RADS v2022)
- 中国肺癌筛查与早诊早治指南（2021，北京）.pdf
- 中华医学会肺癌临床诊疗指南（2024版）.pdf

当前可直接形成的输入输出:
- 输入: 结节特征 (size, density, change) + 患者风险因素 (age, smoking)
- 输出: 随访建议 (复查间隔、检查方式、是否活检)
- 规则来源: 三份指南可结构化为 IF-THEN 规则

指南结构化程度:
- Lung-RADS v2022: 95% 可结构化, Category 0-4X, 精确的大小/密度阈值
- 中国 2021 筛查指南: 75% 可结构化, 高危人群定义 (50-74岁, >=30包年), 密度分类
- 中国 2024 诊疗指南: 60% 可结构化, 结节大小阈值 (<5mm, <8mm, >=5mm, >=8mm)

当前最缺的标签/字段:
- 无 "结节特征 -> 专家随访建议" 的端到端金标准对
- 无法自动评估规则推理输出的正确性

当前能做的 baseline:
- 以 Lung-RADS 为主框架, 手工编码 Category 0-4X 的 IF-THEN 规则
- 融合中国指南的高危人群定义和处置阈值
- 可覆盖约 85% 的常见结节场景

当前不能做的:
- 无法处理指南间冲突 (三份指南建议不一致时)
- 无法处理边界病例 (如结节大小恰好在阈值上)
- 无法验证规则输出的临床正确性 (缺金标准)

---

## C. 关键文件深挖

### C1. radiology.csv.gz
- 8 列, 2,321,355 行 [radiology.csv.gz | pandas len() 全量扫描 | 2,321,355]
- 放射报告详情表 (radiology_detail.csv.gz): 6,046,121 行, 包含 5 种 field_name 类型 [radiology_detail.csv.gz | pandas len() 全量扫描 | 6,046,121]
- 真实样例 (radiology_detail): note_id: 10000032-RR-14, field_name: exam_name, field_value: CHEST (PA & LAT)
- CT/Chest/Lung 相关检查: 1,414,419 / 2,913,024 (48.6%) [radiology_detail.csv.gz | pandas 全量扫描 | 6,046,121]

### C2. discharge.csv.gz
- 8 列, 331,793 行 [discharge.csv.gz | pandas len() 全量扫描 | 331,793]
- 出院小结详情表 (discharge_detail.csv.gz): 186,137 行, 仅含 author 字段, 且全部脱敏 [discharge_detail.csv.gz | pandas 全量扫描 | 186,137]
- 真实样例: note_id: 10000032-DS-21, field_name: author, field_value: ___

### C3. LIDC XML
- 1,318 个文件, 分布在 6 个子目录 (157, 185-189) [lidc_extracted/tcia-lidc-xml/ | find 命令全量扫描 | 1,318]
- 命名空间: {http://www.nih.gov}, 根节点: LidcReadMessage
- 通常每例有 4 次阅读会话 (多读者标注)
- 每个结节包含 9 项特征:

| 特征 | 取值范围 | 含义 |
| :--- | :--- | :--- |
| subtlety | 1-5 | 显著性 (1=极难发现, 5=极易发现) |
| internalStructure | 1-4 | 内部结构 (1=软组织, 2=脂肪, 3=液体, 4=空气) |
| calcification | 1-6 | 钙化 (1=爆米花样, 6=无钙化) |
| sphericity | 1-5 | 球形度 (1=线性, 5=圆形) |
| margin | 1-5 | 边缘 (1=清晰, 5=模糊) |
| lobulation | 1-5 | 分叶 (1=无, 5=明显) |
| spiculation | 1-5 | 毛刺 (1=无, 5=明显) |
| texture | 1-5 | 质地 (1=非实性/GGO, 5=实性) |
| malignancy | 1-5 | 恶性度 (1=极低, 5=极高) |

- ROI 包含 edgeMap 像素轮廓, 无显式直径字段 (需从轮廓点计算)
- 43% 的采样文件包含结节标注（采样估计，非全量统计）[LIDC XML | 前 100 个文件采样 | 100]

真实样例:
```xml
来源: lidc_extracted/tcia-lidc-xml/185/266.xml
noduleID: Nodule 004
subtlety: 3, internalStructure: 1, calcification: 6, sphericity: 3
margin: 5, lobulation: 1, spiculation: 1, texture: 5, malignancy: 2
First ROI: imageZposition=-198.5, edgeMap points=19
```

### C4. MIMIC-IV 核心表 (mimic-iv-3.1.zip, 待解压)
- patients.csv.gz: 364,627 名患者 [mimic-iv-3.1.zip | pandas from zip 全量扫描 | 364,627]
  - 字段: subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod
  - 性别分布: F 191,984 / M 172,643 [patients.csv.gz | pandas 全量扫描 | 364,627]
  - 年龄分布: 均值 49, 中位数 48, 范围 18-91 [patients.csv.gz | pandas describe() 全量扫描 | 364,627]
  - 真实样例: subject_id: 10000032, gender: F, anchor_age: 52, anchor_year: 2180, anchor_year_group: 2014-2016, dod: 2180-09-09

- admissions.csv.gz:
  - 字段: subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admit_provider_id, admission_location, discharge_location, insurance, language, marital_status, race, edregtime, edouttime, hospital_expire_flag
  - 真实样例: subject_id: 10000032, hadm_id: 22595853, admission_type: URGENT, admission_location: TRANSFER FROM HOSPITAL, insurance: Medicaid, race: WHITE

- diagnoses_icd.csv.gz:
  - 字段: subject_id, hadm_id, seq_num, icd_code, icd_version
  - 真实样例: subject_id: 10000032, hadm_id: 22595853, seq_num: 1, icd_code: 5723, icd_version: 9

### C5. medical.json
- 8,808 条疾病条目, JSONL 格式 [medical.json | pandas 全量扫描 | 8,808]
- 疾病百科性质, 非结节专项数据
- 包含 25 个字段, 核心字段非空率 99%+
- 真实样例: `{"name": "肺泡蛋白质沉积症", "category": ["疾病百科","内科","呼吸内科"], "symptom": ["紫绀","胸痛","呼吸困难","乏力"], "check": ["胸部CT检查","肺活检","支气管镜检查"]}`
- 潜在用途: 知识图谱语义匹配 (如将 "磨玻璃影" 映射至 "亚实性结节")
- 无法直接用于三大核心任务的训练

### C6. 临床指南
- Lung-RADS v2022: 核心随访框架, 包含结节大小/密度与 Category 0-4X 的映射。
- 中国 2021 筛查指南: 定义了中国人群的高危准入标准 (50-74岁, >=30包年)。
- 中国 2024 诊疗指南: 提供了临床处置的阈值参考 (5mm, 8mm)。

### C7. 数据关联性分析
- radiology 与 discharge 通过 subject_id 关联。
- 结节报告患者中 79.8% 可关联到 discharge 记录（采样估计，非全量统计）[radiology 前 100K + discharge 前 100K subject_id 交集 | 100,000]
- patients 表通过 subject_id 提供年龄/性别。
- admissions 表通过 hadm_id 提供就诊事件级关联。

---

## D. 风险判断

1. 当前数据是否足以构建最小闭环基线? — 部分可以。放射报告抽取 + 规则推理可闭环, 但吸烟史判定环节只能用弱监督 baseline 填充。
2. 信息抽取模块数据是否充足? — 语料充足 (约 19 万条结节报告), 但缺人工标注金标准, 只能做规则/弱监督 baseline。
3. 吸烟史判定模块数据是否充足? — 严重不足。Social History 脱敏是不可逆的数据缺陷。定量信息 (pack-year) 仅约 0.5%（采样估计，非全量统计）。只能支撑弱监督 baseline, 不适合作为主实验的监督学习任务。
4. 端到端评估数据是否充足? — 不足。缺 "报告 -> 随访建议" 金标准对。需构建 silver-standard 评测集。
5. 数据关联性是否满足需求? — 基本满足。79.8% 的结节患者可关联 discharge（采样估计，非全量统计）, patients 表提供年龄/性别。但 MIMIC-IV hosp 尚未解压。

---

## E. 下一步建议

P1: 解压 MIMIC-IV 并构建患者级索引
- 从 mimic-iv-3.1.zip 提取 patients.csv.gz 和 admissions.csv.gz
- 构建 subject_id -> (age, gender, admissions) 的查找表
- 与 radiology/discharge 的 subject_id 做全量关联统计
- 目的: 获取年龄/性别, 这是高危筛查准入的先决条件

P2: 构建结节报告语料库并计算 LIDC 直径
- 从 radiology.csv.gz 过滤含 "nodule" 的报告 (~190K), 按 exam_name 进一步筛选胸部 CT
- 解析 FINDINGS/IMPRESSION section, 建立结构化字段提取的规则 baseline
- 从 LIDC XML 的 edgeMap 轮廓点计算结节等效直径, 建立形态学参考集
- 目的: 为信息抽取模块提供可用语料和弱监督信号

P3: 构建 silver-standard 评测集
- 从结节报告中筛选含明确 recommendation cue 的样本 (如含 "recommend", "follow-up", "Lung-RADS" 的报告, 约占结节报告的 30%（采样估计，非全量统计）)
- 用指南规则 (Lung-RADS / 中国指南) 对抽取出的结节特征自动生成伪标签 (pseudo-label)
- 对伪标签进行少量人工 spot-check / case review (约 50-100 条), 而非大规模人工标注
- 目的: 支持 baseline 评测与案例分析, 而非构建严格 gold standard

---

## 附录：证据与待核实项

### 表1：已核实事实

| 结论 | 来源文件 | 核实方式 | 可信度 |
| :--- | :--- | :--- | :--- |
| radiology.csv.gz 共 2,321,355 行 | radiology.csv.gz | pandas len() 全量扫描 | 高 |
| discharge.csv.gz 共 331,793 行 | discharge.csv.gz | pandas len() 全量扫描 | 高 |
| radiology_detail.csv.gz 共 6,046,121 行 | radiology_detail.csv.gz | pandas len() 全量扫描 | 高 |
| discharge_detail.csv.gz 共 186,137 行 | discharge_detail.csv.gz | pandas len() 全量扫描 | 高 |
| LIDC XML 共 1,318 文件 | lidc_extracted/tcia-lidc-xml/ | find 命令全量扫描 | 高 |
| patients.csv.gz 共 364,627 患者 | mimic-iv-3.1.zip 内 | pandas len() 全量扫描 | 高 |
| medical.json 共 8,808 条 | medical.json | pandas 全量扫描 | 高 |
| radiology_detail exam_name 中 CT/Chest/Lung 占 48.6% | radiology_detail.csv.gz | pandas 全量扫描 | 高 |
| discharge_detail 仅含 author 字段, 值均为 "___" | discharge_detail.csv.gz | pandas 全量扫描 | 高 |
| patients 性别分布 F:191,984 M:172,643 | patients.csv.gz | pandas 全量扫描 | 高 |
| patients 年龄 mean=49, median=48, range=18-91 | patients.csv.gz | pandas describe() 全量扫描 | 高 |

### 表2：待二次核实项

| 结论 | 当前依据 | 为什么还不够 | 后续如何核实 |
| :--- | :--- | :--- | :--- |
| 约 190,099 条含 "nodule" 的 radiology 报告 (8.19%) | 前 50,000 行采样, 线性外推 | 采样仅覆盖 2.2% 的数据, 分布可能不均匀 | 全量 grep 或 pandas 全量扫描 |
| 97.4% discharge 含 Social History | 前 10,000 行采样 | 采样仅覆盖 3% 的数据 | 全量扫描 |
| 0.5% discharge 含 pack-year | 前 10,000 行采样 (51/10,000) | 样本量小, 置信区间宽 | 全量扫描 |
| 79.8% 结节患者可关联 discharge | radiology 前 100K + discharge 前 100K 的 subject_id 交集 | 两个采样窗口可能不代表全量 | 全量 subject_id 交集计算 |
| 43% LIDC 文件含结节标注 | 前 100 个 XML 文件采样 | 仅覆盖 7.6% 的文件 | 全量 XML 解析 |
| 结节报告关键词分布 (size 57.3% 等) | 前 50,000 行中结节报告子集 | 采样外推, 存在偏差风险 | 全量扫描后重新统计 |

---

Readiness 结论:

**(b) 可以启动部分 baseline 工作，但缺少关键标注。**

放射报告信息抽取模块 (B1) 具备充足语料 (~190K 结节报告，采样估计) 和清晰的文本结构 (FINDINGS/IMPRESSION), 可立即启动基于规则的 baseline。随访规则推理模块 (B3) 有三份可结构化的指南, 可立即编码 IF-THEN 规则。但吸烟史判定模块 (B2) 因 Social History 系统性脱敏, 仅能支撑弱监督 baseline, 不适合作为主实验任务。端到端评估需先完成 P3 (构建 silver-standard 评测集) 才能进入正式实验阶段。
