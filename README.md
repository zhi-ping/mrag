# MRAG — 基于 llama.cpp 的本地 RAG 问答系统

## 项目简介

MRAG（Minimal RAG）是一个完全运行在**本地**的检索增强生成（RAG）问答系统。它基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) C API 作为推理后端，支持 CUDA GPU 加速，可在不联网、不上传数据的前提下，对 TXT 小说进行文档切分、向量索引和智能问答。

**核心特性：**

- **纯本地运行** — 所有数据（文档、向量库、模型推理）均在本地处理，保障隐私安全
- **离线建库 + 在线问答** — 两阶段工作流，建库一次即可反复问答
- **流式输出** — 大模型生成结果实时逐字打印
- **热切换模型 / 配置** — 运行时无需重启即可切换 Embedding / Generation 模型或整个配置文件
- **多文档支持** — `--ingest` 可一次处理多个 TXT 文件
- **CUDA 加速** — 支持 NVIDIA GPU 推理（需在 WSL / Linux 下配置 CUDA 环境）

---

## 目录结构

```
mrag/
├── CMakeLists.txt                # CMake 构建文件
├── README.md                     # 使用说明文档
├── DESIGN.md                     # 设计说明文档
├── config.json                   # 示例配置文件
└── src/
    ├── main.cc                   # 程序入口，命令行解析
    ├── config.h                  # 配置管理：结构体定义
    ├── config.cc                 # 配置管理：JSON 解析与加载
    ├── chunk.h                   # 数据结构：Chunk 定义 + 序列化接口
    ├── chunk.cc                  # 数据结构：二进制序列化/反序列化实现
    ├── document.h                # 文档处理：章节识别 + 文本切分接口
    ├── document.cc               # 文档处理：UTF-8 边界、语义对齐、切分实现
    ├── vectordatabase.h          # 向量数据库：存储、余弦检索、持久化接口
    ├── vectordatabase.cc         # 向量数据库：最小堆检索、魔数校验实现
    ├── llamamodel.h              # 模型基类：RAII 封装 llama_model/context
    ├── llamamodel.cc             # 模型基类：模型加载、tokenize、资源释放
    ├── embeddingengine.h         # 向量引擎：文本转向量接口
    ├── embeddingengine.cc        # 向量引擎：batch 构造、decode、提取 embedding
    ├── generationengine.h        # 生成引擎：RAG 流式推理接口
    └── generationengine.cc       # 生成引擎：ChatML prompt、采样链、自回归生成
```

---

## 1. 依赖项安装

### 系统要求

- **操作系统：** Linux x86-64（推荐 Ubuntu 22.04 / 24.04，或 WSL2）
- **编译器：** GCC 15+（需支持 C++23 标准）或 Clang 19+
- **GPU（可选）：** NVIDIA GPU + CUDA Toolkit 12.x（用于 CUDA 加速推理）

### 安装命令

```bash
#  安装编译工具链
sudo apt update
sudo apt install build-essential

# 如需 CUDA 支持（WSL2 或原生 Linux）
# 参考：https://developer.nvidia.com/cuda-downloads
sudo apt install --no-install-recommends nvidia-cuda-toolkit
```

### 项目依赖

项目通过 CMake `FetchContent` **自动拉取**以下依赖，无需手动安装：

| 依赖库 | 版本 | 用途 |
|--------|------|------|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `b9245` | 大模型推理引擎 |
| [nlohmann/json](https://github.com/nlohmann/json) | `v3.12.0` | JSON 配置文件解析 |

> `CMakeLists.txt` 中已配置 `LLAMA_CUDA=ON`，如需纯 CPU 运行请改为 `OFF`。

---

## 2. 模型文件下载

### 推荐模型

| 用途 | 模型名称 | 显存占用 | 下载地址 |
|------|----------|------|----------|
| 文本嵌入 | `bge-small-zh-v1.5-f16.gguf` | ~200 MB | [Hugging Face](https://huggingface.co/CompendiumLabs/bge-small-zh-v1.5-gguf/tree/main) |
| 文本生成 | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | ~1 GB | [ModelScope](https://www.modelscope.cn/models/LoveSeaW/Qwen2.5-1.5b-instruct-gguf/files) / [Hugging Face ](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/tree/main) |
| 文本生成（高配） | `qwen2.5-7b-instruct-q4_k_m.gguf` | ~4.5 GB | [ModelScope](https://www.modelscope.cn/models/aplux/qwen2.5-7b-instruct-q4_k_m/files) / [Hugging Face ](https://huggingface.co/PatataAliena/Qwen2.5-VL-7B-Instruct-Q4_K_M-GGUF/tree/main) |

### 放置路径

下载后请将 `.gguf` 文件放在项目根目录下的 `models/` 文件夹中，与 `config.json` 中的路径保持一致：

```
mrag/
└── models/
    ├── bge-small-zh-v1.5-f16.gguf
    └── qwen2.5-1.5b-instruct-q4_k_m.gguf
```

> 如果使用其他模型文件或放在其他路径，请修改 `config.json` 中 `models.embedding` 和 `models.generation` 的值。

---

## 3. 编译

```bash
# 在项目根目录下执行
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

编译成功后会在 `build/` 目录下生成 `mrag` 可执行文件。

> **注意：** 如果使用 WSL2 + CUDA，编译前请确保已设置环境变量：
>
> ```bash
> export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
> ```

---

## 4. 运行示例

### 4.1 离线建库（`--ingest`）

将 TXT 小说切分为文本块（Chunk），逐块生成嵌入向量，存入向量数据库。

```bash
# 单文档建库
./build/mrag --ingest doc/king3.txt db/king3.db

# 多文档建库（最后一个参数为数据库输出路径）
./build/mrag --ingest doc/king3.txt doc/Monkey.txt db/novels.db
```

运行过程输出示例：

```
正在读取配置文件...
正在初始化计算引擎...
正在处理文本...
正在导入训练文本[1/1]
[建库进度信息...]
```

### 4.2 在线问答（`--chat`）

加载数据库，进入交互式问答循环。

```bash
./build/mrag --chat db/king3.db
```

交互示例：

```
>> 刘备、关羽、张飞是在哪里结义的？
[模型流式输出回答...]
>> 关羽的武器叫什么名字，重多少斤？
[模型流式输出回答...]
>> /exit
Bye from mrag~~ 
```

### 4.3 内置命令

在 `--chat` 模式下，支持以下内置命令（以 `/` 开头）：

| 命令 | 说明 |
|------|------|
| `/model -g <路径>` | 切换生成模型 |
| `/model -e <路径>` | 切换向量模型 |
| `/config <路径>` | 重载配置文件 |
| `/help` | 显示帮助信息 |
| `/exit` 或 `/quit` | 退出问答会话 |

---

## 5. 配置文件说明

`config.json` 示例及字段含义：

```json
{
  "models": {
    "embedding":   "./models/bge-small-zh-v1.5-f16.gguf",
    "generation":  "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "n_gpu_layers": 99
  },
  "document":  { "chunk_size": 500, "overlap_size": 50 },
  "retrieval": { "top_k": 3 },
  "generation": {
    "n_ctx": 4096, "n_batch": 4096, "n_ubatch": 512,
    "max_output_tokens": 512,
    "temperature": 0.7, "top_k": 40, "top_p": 0.9
  },
  "embedding": { "n_ctx": 512, "n_batch": 512, "n_ubatch": 512 }
}
```

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `models.embedding` | `./models/bge-small-zh-v1.5-f16.gguf` | 嵌入模型路径 |
| `models.generation` | `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf` | 生成模型路径 |
| `models.n_gpu_layers` | `99` | GPU 加速层数（99 = 全部） |
| `document.chunk_size` | `500` | 文本切分大小（字节） |
| `document.overlap_size` | `50` | 相邻块重叠大小（字节） |
| `retrieval.top_k` | `3` | 检索返回的片段数 |
| `generation.n_ctx` | `4096` | 生成模型上下文窗口 |
| `generation.temperature` | `0.7` | 采样温度 |
| `embedding.n_ctx` | `512` | 嵌入模型上下文窗口 |

> 配置文件不存在时，程序不报错，全部使用默认值。

---

## 6. 已知问题与局限性

1. **C++ 标准版本较高：** 项目使用了 C++23 特性（如 `std::println`、`std::ranges` 等），需要较新版本的编译器（GCC 13+），在老版本 Linux 发行版上可能无法直接编译。
2. **单线程处理：** 建库阶段逐块生成嵌入向量为串行处理，大量文本建库时耗时较长。
3. **仅支持 TXT 格式：** 当前文档处理模块仅支持 UTF-8 编码的纯文本小说，不支持 PDF、DOCX、HTML 等格式。
4. **嵌入模型与生成模型需匹配语言：** 嵌入模型建议使用中文优化模型（如 `bge-small-zh`），生成模型建议使用支持 ChatML 格式的中文模型（如 Qwen2.5 系列）。

---

## 参考文献

- llama.cpp 官方仓库：<https://github.com/ggml-org/llama.cpp>
- nlohmann/json 文档：<https://json.nlohmann.me>
- BGE 模型说明：<https://huggingface.co/BAAI/bge-small-zh-v1.5>
- Qwen2.5 模型说明：<https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
