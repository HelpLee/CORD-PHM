# 前处理（全套仅保留一份）

preprocess_health_tokens 包含 common、datasets、统一构建入口和逐NPY校验程序。
包内仅调整数据根目录定位，使其能从本目录找到工作区 code/data_phm/raw。
特征计算、传感器选择和NPZ格式未修改。外部位置可通过
HEALTHTOKEN_DATA_CODE_ROOT 指向包含 data_phm 的目录。

重建到新目录：python 00_preprocessing/run.py --out-root <candidate>
指定数据集时增加 --include <registry_names>。
逐字段验证：在本目录执行 python -m preprocess_health_tokens.verify_reproduction
--reference-root <existing_processed_health_tokens> --candidate-root <candidate>。

尚未重新处理全部raw并逐位比较；不能把代码整理与路径验证说成完整原始数据重建成功。
当前训练复现使用已保存的NPZ与01中的冻结缓存。原NPZ/raw不在本目录重复保存。
