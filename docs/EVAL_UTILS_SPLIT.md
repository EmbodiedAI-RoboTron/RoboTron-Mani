# Evaluation Utils 拆分总结

## 概述

将 `robouniview/eval/eval_utils.py` 拆分为两个独立的模块：
1. **Policy Server** (`examples/calvin/policy_server.py`) - 模型包装和策略服务器
2. **Evaluation Client** (`examples/calvin/main.py`) - 评估客户端和评估逻辑

## 文件结构

### 1. `examples/calvin/policy_server.py`

**职责：** 提供模型包装和策略服务器功能

**主要内容：**
- `ModelWrapper` 类：包装训练好的模型，实现 `CalvinBaseModel` 接口
  - 图像和文本预处理
  - 动作历史管理
  - 特征缓存（用于时序模型）
  - Diffusion 模型集成（如果使用）
- `get_gripper_camera_view_matrix()`: 从 PyBullet 获取夹爪相机视图矩阵
- `get_cast_dtype()`: 根据精度字符串获取数据类型
- `main()`: Policy server 入口点（待实现 WebSocket 服务器）

**关键功能：**
- 处理多模态输入（RGB图像、夹爪图像、状态）
- 支持多种融合模式（Temporal, two_way, vit_concat）
- 支持 Diffusion 和确定性动作头
- 处理动作历史队列和特征缓存

### 2. `examples/calvin/main.py`

**职责：** 评估客户端和评估逻辑

**主要内容：**
- `CalvinPolicyClient` 类：WebSocket 客户端包装器
  - 连接到 policy server
  - 预处理 Calvin 观察数据
  - 管理动作计划缓存
- `make_env()`: 创建 Calvin 环境
- `load_lang_task()`: 加载语言注释和任务oracle
- `evaluate_policy_ddp()`: 分布式评估主函数
- `evaluate_sequence()`: 评估一个任务序列
- `rollout()`: 执行单个子任务的rollout
  - 包含完整的动作转换逻辑（坐标系转换、旋转处理）
  - 支持动作分块处理
  - 调试模式下的视频保存

**关键功能：**
- 通过 WebSocket 与 policy server 通信
- 执行多步骤任务序列评估
- 收集和保存评估结果
- 支持调试模式和可视化

## 代码迁移映射

### 从 `eval_utils.py` 到 `policy_server.py`:
- ✅ `ModelWrapper` 类（完整迁移）
- ✅ `get_gripper_camera_view_matrix()` 函数
- ✅ `get_cast_dtype()` 函数

### 从 `eval_utils.py` 到 `main.py`:
- ✅ `evaluate_policy_ddp()` 函数（已存在，已更新）
- ✅ `evaluate_sequence()` 函数（已存在）
- ✅ `rollout()` 函数（已存在，已增强）
- ✅ `make_env()` 函数（已存在）

### 保留在 `eval_utils.py`:
- `eval_one_epoch_calvin()` - 用于训练循环中的评估
- `eval_one_epoch_calvin_ddp()` - 分布式训练中的评估
- `eval_one_epoch_calvin_with_dataloder()` - 使用数据加载器的评估
- `main()` - 命令行评估入口
- `generate_zero_shot_instr()` - 生成零样本指令
- `save_sequences()` - 保存评估序列

## 使用方式

### 启动 Policy Server

```bash
python examples/calvin/policy_server.py \
    --checkpoint /path/to/checkpoint.pt \
    --config /path/to/config.yaml \
    --host 0.0.0.0 \
    --port 8000
```

**注意：** WebSocket 服务器实现需要根据 `openpi_client` 的接口完成。

### 运行评估

```bash
python examples/calvin/main.py \
    --host 0.0.0.0 \
    --port 8000 \
    --dataset_path /path/to/calvin/task_D_D \
    --num_sequences 1000 \
    --eval_log_dir /path/to/logs
```

## 依赖关系

### `policy_server.py` 依赖：
- `calvin_agent.models.calvin_base_model.CalvinBaseModel`
- `robouniview.data.multi_cam_data` (preprocess_image, preprocess_text_calvin)
- `robouniview.utils` (tcp_to_world_frame)
- PyTorch, PyBullet, NumPy

### `main.py` 依赖：
- `openpi_client` (websocket_client_policy, image_tools)
- `calvin_agent.evaluation.utils`
- `calvin_env.envs.play_table_env`
- MoviePy, NumPy, PyBullet

## 待完成工作

1. **Policy Server WebSocket 实现**
   - 在 `policy_server.py` 的 `main()` 函数中实现 WebSocket 服务器
   - 使用 `openpi_client` 或类似的库
   - 处理客户端连接和请求

2. **测试和验证**
   - 确保拆分后的代码功能完整
   - 测试 policy server 和 client 的通信
   - 验证评估结果的一致性

3. **文档更新**
   - 更新使用文档
   - 添加 API 文档

## 优势

1. **关注点分离**：模型推理和评估逻辑分离
2. **可扩展性**：可以独立扩展 policy server 和评估客户端
3. **可重用性**：policy server 可以被多个客户端使用
4. **维护性**：代码结构更清晰，易于维护

