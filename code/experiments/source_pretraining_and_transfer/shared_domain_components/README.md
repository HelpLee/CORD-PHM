# 共享依赖

bearing_runtime、milling_runtime、battery_data 是已使用的冻结派生缓存，
包含源输入、mask、Global特征、历史split/scaler。所有实验通过套件内路径读取。
不会再依赖旧 reproduction_final 目录中的运行缓存。

battery_code 保存原电池的数据构造、State-Trend Head、验证划分和评分函数，
以及新增的 adapter_downstream.py。三个电池下游包通过各自config调用同一训练实现。
原始队列文件仅被导入split/score，不作为训练入口。

requirements-lock.txt 和 original_battery_environment.json 记录保存环境。
当前解释器：python。
Exact bundled-file hashes are recorded in the repository-root
`file_checksums.json`; external dataset inputs are listed in
`external_data_manifest.json`.
NPZ/raw是有意保留在工作区code/data_phm的外部依赖。
