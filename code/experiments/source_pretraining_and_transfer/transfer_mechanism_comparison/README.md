# E33：三领域联合预训练的辅助任务冲突与训练进展平衡

## 当前证据：不能只归结为早停

2026-09-20 对 cluster E28/E29 原始 history.json 和模型代码的检查：

- E29 原始等权三领域后期50个epoch，共享参数的平均梯度范数为
  Bearing 0.358、Battery 1.239、Milling 0.286；电池约为刀具4.33倍。
  领域对的负梯度夹角出现率为41.0%–46.7%，平均余弦仅约0.007–0.018。
  这些说明尺度不均、梯度经常不一致，但近零余弦和随机批次噪声不能单独证明负迁移。
- E29 joint_norm、joint_cagrad 已到500 epoch（上限），不再是提前早停。
  joint_norm 选中492 epoch，三个领域各自的原始验证最低点分别在439/492/460。
  延长训练本身不保证下游收益；该批下游当时尚未完成，不能宣称这些方法失败。
- E28 three5 的选中epoch为289。Battery/Milling在该检查点的原始SSL验证损失
  只比本次轨迹中的各自最小值高约5.14%/2.33%。这不支持把全部下游差距都归因于错过最优epoch。
- 当前 encoder 的输入stem和block adapter按物理领域区分，但所有领域共用GRU+projection，
  将6个snapshot映射到下一snapshot。各领域采样间隔/退化速度不同，共享预测器可能施加不必要约束。
  这是可检验假设，不是已证实的根因。
- reconstruction 只按领域区分decoder和channel embedding。同一领域不同数据集的通道索引
  不代表同一种传感器：sensor_selection.py确认MATWI的force为2/3/4，LUH为0/1/2。
  从force-only升级全通道后，域内重建头需要处理更多不一致的通道语义；数据集身份可能被迫进入共享表示。
  本轮用数据集专用辅助重建头检验，保留全部真实信号。
- E28 single5只有刀具预训练，不能将它在轴承/电池上的迁移称为这两个领域的单领域预训练对照。

完整诊断可在超算运行 `python inspect_previous.py --output previous_diagnostics.json --compact`。

## 文献及其适用范围

1. **Does Cross-Domain Pre-Training Truly Help Time-Series Foundation Models?**
   ICLR 2025 FM-Wild **Workshop**（不是ICLR主会）。作者的跨领域时序实验提示采样频率和演化
   模式差异能引起负迁移。与当前异构系统接近，但并非本项目的轴承/电池/刀具实验。
   [作者机构页面](https://www.microsoft.com/en-us/research/publication/does-cross-domain-pre-training-truly-help-time-series-foundation-models/)
   [论文](https://openreview.net/pdf?id=PbhGeGBN7X)
2. **FAMO**, NeurIPS 2023：根据训练损失的相对下降调整权重，针对某些任务进展不足。
   相比GradNorm，它关注相对损失进展而非单纯追求梯度大小一致。采用官方默认logit LR=.025、
   weight decay=.01、lower bound=0；保留项目原有模型clip=5。公式用分领域累积梯度实现，
   与对FAMO标量目标自动求导做数值等价测试。为保留诊断，本实现存储各领域梯度，不宣称O(1)存储。
   [论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract.html)
   [官方代码](https://github.com/Cranial-XIX/FAMO)
3. **Recon**, ICLR 2023：将冲突集中的共享层变成任务专有层，提示仅做梯度修正可能不够。
   本实验受其启发，预先指定辅助动态头做分离；没有实现Recon的逐层搜索，因此不能称为完整Recon复现。
   [论文](https://arxiv.org/abs/2302.11289)
4. **DiMAE**, ECCV 2022：共享编码器配合多个域专用解码器，以减少重建域特征对共享表示的干扰。
   本实验借鉴多解码器原则到“物理领域内部的数据集”；不引入图像style-mix，也不强制对齐物理健康坐标。
   [论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136910147.pdf)
5. **Domain Separation Networks**, NeurIPS 2016：共享与私有因素分离，是辅助头隔离的背景依据；
   本轮不采用其对抗域对齐或正交表示损失。[论文](https://proceedings.neurips.cc/paper/2016/hash/45fbc6d3e05ebd93369ce542e8f2322d-Abstract.html)
6. **Aligned-MTL**, CVPR 2023：梯度条件数/独立分量对齐；可以后续比较。本轮已有E29的PCGrad/CAGrad，
   优先测试不同的结构与进展平衡机制，避免堆叠同类梯度处理。
   [论文](https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html)

不优先尝试强制跨域健康坐标对齐/DANN：目前没有证据表明不同物理系统的健康坐标或轨迹可直接对应。
也不以“让三领域训练得更久”为成功判据。成功需要同协议下的实际下游迁移收益。

## 预先固定的实验组

| 三领域模型 | 改动 | 主要检验 |
|---|---|---|
| joint_control | 固定初始训练损失归一化+固定等权 | gpu/FP32下的同期对照 |
| joint_famo | control + FAMO；结构相同 | 是否某领域训练进展被其他领域压制 |
| joint_private_dynamics | control + 每领域独立GRU/projection | 是否共享动态预测器造成负迁移 |
| joint_dataset_heads | control + 每数据集独立channel embedding/decoder | 全通道语义、观测重建的域内冲突 |
| joint_private_both | 同时隔离动态头和数据集重建头 | 两类辅助任务冲突是否叠加 |

另外每个物理领域各训练两种单领域对照：control、dataset_heads，共6个。
单领域只有一个活跃GRU，分离动态头与原来的共享头数学上等价，所以不重复训练第三种单领域。
FAMO仅一个任务时归一化后的梯度系数为1，同样用单领域control比较。
dataset_heads和private_both对比相应dataset_heads单领域，避免把单领域也能获得的结构收益误认为联合收益。

每个三领域模型是**一个共享encoder、一个选中checkpoint用于三个下游领域**。
新增GRU和重建头只在自监督预训练时使用，下游完全丢弃；下游网络容量保持原样。
各新增头从相应原始头精确复制，初始输出和encoder初始化相同，避免额外随机初始化差异。

## 数据、协议、计算

- 刀具：LUH、MATWI、Nonastreda、QIT-CEMC、HMoTP，全部有效通道；沿用E28/E29冻结缓存。
  PHM2010仅作下游，7通道。本轮不再增减数据集。
- 轴承、电池的来源、缓存、train-only标准化、设备划分保持E29。
- 每个活跃领域每epoch 20次参与更新、每次32样本、microbatch8、上游seed42。
  同epoch各方法的设备/行/窗口/mask随机种子配对，领域内仍round-robin。
- loss仍为masked MSE + 0.2×latent dynamics MSE；除以该数据集初始化时的TRAIN loss。
  验证分数按每数据集初始VAL loss归一化，再dataset macro、domain macro。
  验证只选模型/早停，不参与FAMO权重；所有组max500 epoch、patience30、min_delta1e-4。
- 训练最多500epoch是本轮筛查预算，达到上限不宣称收敛。每轮保留模型、AdamW、FAMO、随机状态，支持精确断点续训。
- 继承的QIT-CEMC/KAIST有无可用验证序列的情况单独记录；不伪造验证值。
- 下游：3领域×10%/20%/100%×seeds42–46；full fine-tune encoder3e-4/head1e-3，
  保留E28早停和划分。63个model/domain/fraction作业，每个顺序跑5seed，总计315个下游训练。
- gpu分区为V100，统一使用FP32（TF32关闭）避免BF16不兼容。所有新单领域、三领域对照一起重跑；
  旧A100/BF16只作背景，不作为唯一同期对照。下游自动禁用继承代码内BF16上下文，effective_protocol.json记录实际设置。
- 每次forward单源，新增源专用解码器数量多但单样本只激活一个；报告总训练参数及可迁移encoder参数。

## 提交、依赖、失败处理

运行：`python submit.py`；审阅：`python submit.py --dry-run`。

**所有新作业显式 `--partition=gpu`**。76个作业：1个真实数据smoke、11个上游、63个下游、1个汇总。
两条有界流水线；smoke成功后开始上游，每个下游afterok依赖自己的上游，模型之间用afterany排队。
每个作业1GPU，4CPU，32GB RAM；上游三领域10h、单领域4h、下游每model/domain/fraction2h。
依据E29 A100实测，联合1h04–2h42、单领域14–58分钟，给V100/FP32及FAMO额外前向留余量；
这些是时限上限，不表示实际需要跑满。gpu实际可启动时间由Slurm决定。
到时限前5分钟发信号，上游完成当前epoch保存断点；若最终超时仍须重新提交，不把未完成模型用于下游。
下游失败不阻塞其他独立模型，汇总明确列出缺失项。提交后不设置监控或定期轮询。

## 结果如何判断

所有预定组都报告，不仅报告最好的。RMSE/MAE/R2均值±样本标准差、逐seed胜负、配对RMSE差。
日志分开记录shared encoder、block0、block1、pool/fusion、共享动态头的梯度范数/夹角，
能区分“冲突在encoder还是辅助预测器”；记录各数据集mask/dynamics验证曲线、FAMO权重及参数量。
FAMO比较control；分离动态头比较control；数据集头比较control；组合比较两个单改动。
单领域结构对照用于确认是否存在额外的联合预训练收益。
仅一个上游seed，因此本轮是筛查，5个下游seed不能替代多个独立上游seed。
已多次查看的下游test不能用来无偏选择最终论文方法；确定候选后需要独立未见设备/划分或新的确认性实验。
