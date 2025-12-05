# VLA-Adapter 项目架构与核心机制解析

## 1. 项目概览
**VLA-Adapter** 是一个针对机器人控制任务的视觉-语言-动作（Vision-Language-Action）模型架构。它不从头训练大模型，而是通过轻量级的适配（Adapter）技术，将预训练的视觉骨干（Vision Backbone）和大型语言模型（LLM）转化为一个能够输出精确机械臂动作的控制器。

*   **核心基础**: 
    *   **Vision Backbone**: SigLIP / DINOv2（负责看）
    *   **LLM Backbone**: Qwen2.5-0.5B（负责思考）
*   **核心创新**: 引入了 **Action Query** 和 **Bridge Attention** 机制，实现了参数高效微调（PEFT）。

---

## 2. 核心架构与数据流 (Architecture & Data Flow)

### 2.1 整体流程 ("三明治"结构)
1.  **输入层 (Input)**:
    *   **图像**: 经过 Vision Backbone 提取特征 -> Projector 投影 -> `Visual Embeddings` (Bypass LLM 词表)。
    *   **文本**: 经过 Tokenizer -> `Text Embeddings` (查 LLM 词表)。
    *   **动作查询 (Action Query)**: 64 个可学习的 Embedding 向量 -> 拼接在序列末尾。
2.  **处理层 (LLM Backbone)**:
    *   输入序列 `[Image, Text, Action Query]` 流经冻结的 LLM。
    *   **Action Query 的作用**: 作为“特派员”进入 LLM，利用 Self-Attention 吸收前面的图像和文本信息，生成 **Action Hidden States ($h_a$)**。
    *   同时，LLM 每一层的图像部分输出被保留为 **Task Hidden States ($h_t$)**。
3.  **输出层 (Action Head)**:
    *   接收 LLM 提取的多层 $h_a$ 和 $h_t$。
    *   通过 24 层 MLP ResNet 逐步解码。
    *   最终输出 `(Batch, Chunk_Size, Action_Dim)` 的动作序列。

---

## 3. 关键机制详解

### 3.1 Action Query (动作查询)
*   **定义**: `nn.Embedding(64, llm_dim)`。
*   **本质**: 64 个**可学习的参数矩阵**。它们不是自然语言单词，而是专为激发 LLM 动作理解能力而训练的“机器语”Token。
*   **作用**: 
    *   **输入端**: 充当占位符，强行在 LLM 内部开辟一条“关注动作”的注意力通道。
    *   **输出端**: 作为特征容器，将融合了多模态信息的 Hidden States 传递给 Action Head。
*   **训练**: 虽然 LLM 是冻结的，但梯度会穿过 LLM 回传，更新这 64 个向量的值。

### 3.2 Bridge Attention (桥接注意力)
*   **位置**: 位于 Action Head 的每一层 Block 内部。
*   **类型**: **Cross-Attention (交叉注意力)**，且是 **Bidirectional (全可见)** 的。
*   **公式**: $Q$ (Action Head 当前状态) 去查询 $K, V$ (LLM 的特征)。
*   **查询对象**:
    1.  **Self**: 动作序列自身的连贯性。
    2.  **Task ($h_t$)**: LLM 对应层的**图像特征**（提供空间细节）。
    3.  **Adapter ($h_a$)**: LLM 对应层的**动作特征**（提供语义意图）。
*   **意义**: “层级对齐 (Layer-wise Alignment)”。它允许动作生成模块不断“回头看”，直接利用 LLM 每一层提取的丰富视觉细节，而不是仅依赖最后一层。

### 3.3 Action Head (动作头)
*   **结构**: 24 层 ResNet (MLP + Bridge Attention + Residual Connection)。
*   **初始输入 ($x$)**: 
    *   **推理时**: 全 0 向量。
    *   **训练时**: 全 0 向量 + **可学习的随机噪声**。
    *   **目的**: 作为一个空白载体（种子），让 Bridge Attention 把从 VLM 抓取的信息一层层“写”进去。加噪声是为了训练模型的鲁棒性，迫使其依赖 VLM 的视觉信息而非死记硬背。

---

## 4. 训练与微调 (Training & Finetuning)

### 4.1 训练脚本 (`finetune.py`)
*   **用途**: 训练模型参数。
*   **流程**: 加载模型 -> 冻结 VLM -> 初始化 Action Head -> 循环 (Forward -> Loss -> Backward -> Update)。

### 4.2 参数更新 (Trainable Parameters)
整个系统中，只有以下部分参与梯度更新（其余冻结）：
1.  **Action Head**: 所有参数 (Linear, LayerNorm)。
2.  **Action Query Embeddings**: 输入端的那 64 个向量。
3.  **Projector**: 视觉-语言对齐层。
4.  *(可选) LoRA Adapter*: 如果开启。

### 4.3 损失计算 (Loss Function)
*   **公式**: **Masked L1 Loss**。
*   **逻辑**: 计算 `|Predicted Action - Ground Truth|`。
*   **Mask 的作用**: 
    1.  过滤掉因 Batch 对齐而产生的 Padding 数据（无效步）。
    2.  确保 Loss 只关注动作输出，忽略图像和文本部分的 LLM 输出。

---

## 5. 离散化动作头重构 (Discrete Action Head Refactoring) - 2025/12 Update

为了提高模型对多模态动作分布的建模能力，将原有的连续回归（L1 Regression）改造为离散分箱分类任务（Discrete Binning Classification）。

### 5.1 核心改动
1.  **自适应分箱 (Adaptive Binning)**:
    *   **脚本**: `vla-scripts/compute_adaptive_bins.py`
    *   **逻辑**: 扫描数据集，计算每个动作维度的 Min/Max。
    *   **Margin**: 在 Min/Max 基础上各扩展 5% (共 10%) 的缓冲空间，防止动作溢出。
    *   **输出**: 生成 `dataset_statistics_256_bins.json` 供 DataLoader 和模型使用。

2.  **DiscreteActionHead**:
    *   **文件**: `prismatic/models/action_heads.py`
    *   **架构**: 继承 MLPResNet，输出维度改为 `Action_Dim * N_Bins` (默认 256)。
    *   **输出**: Reshape 为 `(Batch, Chunk, Action_Dim, N_Bins)` 的 Logits。

3.  **模型支持 (`OpenVLAForActionPrediction`)**:
    *   **推理**: 新增自动 Argmax 和 De-binning 逻辑。如果检测到 Head 输出为 Logits，自动将其映射回连续动作值，保持对外 API 不变。
    *   **Ground Truth 处理**: 新增 `discretize_actions` 方法，将连续标签转换为分箱索引。

4.  **训练流程 (`finetune.py`)**:
    *   **配置**: 新增 `--use_discrete_head` 和 `--n_action_bins`。
    *   **Loss**: 从 `L1Loss` 切换为 `CrossEntropyLoss`。
    *   **Metrics**: 新增 Action Accuracy 监控。

---

## 6. 双臂协同架构扩展 (Dual-Arm Architecture Extension) - 2025/12 Update

针对 ALOHA 等双臂机器人平台，将架构扩展为支持显式双臂协同（Explicit Inter-Arm Coordination）的版本。

### 6.1 核心改动
1.  **Action Tokens 扩展**:
    *   **Constants**: 对于 ALOHA 平台，`NUM_TOKENS` 从 64 增加到 128。
    *   **分配**: 前 64 个 Token 对应 Left Arm，后 64 个 Token 对应 Right Arm。

2.  **DualArmActionHead (`action_heads.py`)**:
    *   **输入拆分**: 自动将 VLM 提取的 `actions_hidden_states` (Batch, 128, D) 拆分为 `h_a_L` 和 `h_a_R`。
    *   **双流架构**: 内部维护两个独立的动作生成流 `x_L` 和 `x_R`。

3.  **Inter-Arm Attention (臂间注意力)**:
    *   **机制**: 在 `DualArmMLPResNetBlock` 中，引入了新的交叉注意力路径。
    *   **Left Stream**: $Q_L$ attend to $K_R$ (右臂当前状态)。
    *   **Right Stream**: $Q_R$ attend to $K_L$ (左臂当前状态)。
    *   **意义**: 这允许双臂在生成动作的每一步都互相感知对方的意图，从而实现防碰撞和精细协同。

4.  **训练支持**:
    *   **配置**: 新增 `--use_dual_arm`。
    *   **流程**: 复用 L1 Regression 训练流程，但底层模型替换为 `DualArmActionHead`。

---

## 7. 总结：VLA-Adapter 是如何工作的？
它通过 **Action Query** 在冻结的 LLM 中“劫持”了注意力，迫使 LLM 提取出与动作相关的特征。然后，它利用 **Action Head** 中的 **Bridge Attention**，像搭积木一样，一层层地从 LLM 的深层特征库中检索视觉和语义信息，最终精细地重构出机器人的控制指令。
