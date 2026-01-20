# YouDub-webui: 优质视频中文化工具
## 目录
- [YouDub-webui: 优质视频中文化工具](#youdub-webui-优质视频中文化工具)
  - [目录](#目录)
  - [简介](#简介)
  - [主要特点](#主要特点)
  - [项目结构](#项目结构)
  - [安装与配置指南](#安装与配置指南)
    - [1. 环境要求](#1-环境要求)
    - [2. 克隆仓库](#2-克隆仓库)
    - [3. 安装依赖](#3-安装依赖)
    - [4. 环境配置](#4-环境配置)
  - [快速开始](#快速开始)
    - [1. 启动程序](#1-启动程序)
    - [2. 基本使用流程](#2-基本使用流程)
  - [批量处理功能](#批量处理功能)
    - [1. tasks.csv 文件格式](#1-tasks-csv-文件格式)
    - [2. 创建和编辑 tasks.csv](#2-创建和编辑-taskscsv)
    - [3. 运行批量处理](#3-运行批量处理)
    - [4. 检测任务完成情况](#4-检测任务完成情况)
    - [5. 从现有视频重建任务](#5-从现有视频重建任务)
  - [任务类型说明](#任务类型说明)
    - [1. download_only (仅下载)](#1-download_only-仅下载)
    - [2. full_process (完整处理)](#2-full_process-完整处理)
  - [处理流程详解](#处理流程详解)
    - [步骤 0: 视频下载](#步骤-0-视频下载)
    - [步骤 1: 音频分离](#步骤-1-音频分离)
    - [步骤 2: 语音识别](#步骤-2-语音识别)
    - [步骤 3: 字幕翻译](#步骤-3-字幕翻译)
    - [步骤 4: 语音合成](#步骤-4-语音合成)
    - [步骤 5: 视频合成](#步骤-5-视频合成)
    - [步骤 6: 生成视频信息](#步骤-6-生成视频信息)
    - [步骤 7: 上传 Bilibili](#步骤-7-上传-bilibili)
  - [常见问题与解决方案](#常见问题与解决方案)
  - [技术细节](#技术细节)
  - [贡献指南](#贡献指南)
  - [许可协议](#许可协议)
  - [支持与联系方式](#支持与联系方式)

## 简介
`YouDub-webui` 是一个强大的视频中文化工具，能够将 YouTube 和其他平台上的高质量视频翻译和配音成中文版本。该工具结合了最新的 AI 技术，包括语音识别、大型语言模型翻译，以及 AI 声音克隆技术，提供与原视频相似的中文配音，为中文用户提供卓越的观看体验。

`YouDub-webui` 适用于多种场景，包括教育、娱乐和专业翻译，特别适合那些希望将国外优秀视频内容本地化的用户。此工具的简洁界面和批量处理功能使得即使是非技术用户也能轻松上手，实现视频的快速中文化处理。

## 主要特点
- **视频下载**: 支持通过链接直接下载 YouTube 视频，包括单个视频、播放列表和频道。
- **批量处理**: 通过 `tasks.csv` 文件实现多视频批量处理，支持不同处理类型。
- **任务管理**: 提供任务完成检测和任务重建功能，方便管理大量视频处理任务。
- **AI 语音识别**: 利用先进的 WhisperX 技术，将视频中的语音高效转换为文字，并支持说话者分离。
- **大型语言模型翻译**: 结合大型语言模型实现快速且精准的中文翻译，支持多种翻译模型。
- **AI 声音克隆**: 通过 AI 声音克隆技术，生成与原视频配音相似的中文语音，保留原视频的情感和语调特色。
- **视频处理**: 综合了音视频同步处理、字幕添加、视频播放速度调整和帧率设置等多项功能。
- **自动上传**: 支持将最终视频自动上传到 Bilibili 平台。

## 项目结构
```
YouDub-webui/
├── YouDub-webui.26.1.4.7z          # 项目压缩包
├── YouDub-webui.7z                  # 项目压缩包
├── CosyVoice/                       # CosyVoice 语音合成模型
├── GPT-SoVITS/                      # GPT-SoVITS 语音合成模型
├── gpt_sovits_workspace/            # GPT-SoVITS 工作空间
├── logs/                            # 日志文件目录
├── models/                          # 模型文件目录
├── test_tts_output/                 # TTS 测试输出目录
├── test_video_folder/               # 测试视频文件夹
├── verify/                          # 验证脚本目录
├── videos/                          # 视频输出目录
├── voice/                           # 语音克隆样本目录
├── whisper.cpp/                     # Whisper C++ 实现
├── youdub/                          # 核心功能模块
│   ├── run_pipeline.py              # 处理管道主程序
│   ├── step000_video_downloader_csv.py # 视频下载模块
│   └── ...                          # 其他处理步骤模块
├── .env                             # 环境配置文件
├── .env.example                     # 环境配置示例
├── .gitignore                       # Git 忽略文件
├── app.py                           # WebUI 主程序
├── check_tasks_completion.py        # 任务完成检测脚本
├── config_balanced.yaml             # 配置文件
├── cookies.txt                      # 浏览器 cookies 文件
├── recreate_tasks.py                # 任务重建脚本
├── requirements.txt                 # 依赖清单
├── run_windows.bat                  # Windows 启动脚本
├── setup_windows.bat                # Windows 安装脚本
├── start_cmd.bat                    # 命令行启动脚本
└── tasks.csv                        # 批量任务配置文件
```

## 安装与配置指南
### 1. 环境要求
- **操作系统**: Windows 10/11
- **Python 版本**: Python 3.10 或更高版本
- **硬件要求**: 
  - CPU: 至少 4 核
  - GPU: NVIDIA GPU (推荐，用于加速 AI 处理)
  - 内存: 至少 8GB RAM
  - 存储空间: 至少 50GB 可用空间

### 2. 克隆仓库
1. 安装 Git: 从 [Git 官网](https://git-scm.com/) 下载并安装 Git
2. 打开命令提示符 (CMD) 或 PowerShell
3. 运行以下命令克隆仓库:
   ```bash
   git clone https://github.com/liuzhao1225/YouDub-webui.git
   cd YouDub-webui
   ```

### 3. 安装依赖
#### 自动安装 (推荐小白使用)
1. 双击运行 `setup_windows.bat` 文件
2. 脚本会自动创建虚拟环境并安装所有依赖
3. 安装过程可能需要几分钟，请耐心等待

#### 手动安装
1. 打开命令提示符，进入项目目录
2. 创建虚拟环境:
   ```bash
   python -m venv venv
   ```
3. 激活虚拟环境:
   ```bash
   venv\Scripts\activate
   ```
4. 安装基础依赖:
   ```bash
   pip install -r requirements.txt
   ```
5. 安装 TTS 依赖:
   ```bash
   pip install TTS
   ```

### 4. 环境配置
1. 将 `.env.example` 文件复制一份并重命名为 `.env`
2. 编辑 `.env` 文件，填写所需的环境变量:
   - `OPENAI_API_KEY`: OpenAI API 密钥 (可选，用于翻译)
   - `MODEL_NAME`: 翻译模型名称，如 'gpt-4' 或 'gpt-3.5-turbo'
   - `HF_TOKEN`: Hugging Face token (用于说话者分离功能)
   - `APPID` 和 `ACCESS_TOKEN`: 火山引擎 TTS 凭据 (可选)
   - `BILI_BASE64`: Bilibili 上传凭据 (可选，用于自动上传)

## 快速开始
### 1. 启动程序
#### 方式一: WebUI (推荐小白使用)
1. 双击运行 `run_windows.bat` 文件
2. 等待程序启动，会自动打开浏览器
3. 在浏览器中使用 WebUI 进行操作

#### 方式二: 命令行
1. 打开命令提示符，进入项目目录
2. 激活虚拟环境:
   ```bash
   venv\Scripts\activate
   ```
3. 启动 WebUI:
   ```bash
   python app.py
   ```
4. 在浏览器中访问显示的 URL (通常是 http://127.0.0.1:7860)

### 2. 基本使用流程
1. 在 WebUI 中选择 "全自动 (Do Everything)" 选项卡
2. 填写视频 URL、输出目录等参数
3. 点击 "Run" 按钮开始处理
4. 等待处理完成，查看输出视频

## 批量处理功能
### 1. tasks.csv 文件格式
`tasks.csv` 是批量处理的核心配置文件，包含以下字段:

| 字段名 | 说明 | 示例值 |
|-------|------|--------|
| url | 视频 URL | https://www.youtube.com/shorts/ieW8-Qjs5rU |
| resolution | 视频分辨率 | 1080p |
| mode | 处理模式 | all |
| status | 任务状态 | pending, processing, completed |
| title | 视频标题 | Look at the genius idea he had with this tree |
| duration | 视频时长 (秒) | 60 |
| video_id | 视频 ID | ieW8-Qjs5rU |
| publish_date | 发布日期 | 20251114 |
| uploader | 上传者 | Alex Demuner |
| progress | 处理进度 |  |
| start_time | 开始时间 |  |
| end_time | 结束时间 |  |
| output_path | 输出路径 | videos\Alex Demuner\20251114... |
| error_message | 错误信息 |  |
| task_type | 任务类型 | download_only, full_process |
| steps | 执行步骤 | 0 (仅下载), 0,1,2,3,4,5,6,7 (完整处理) |

### 2. 创建和编辑 tasks.csv
#### 方式一: 手动编辑
1. 使用文本编辑器 (如 Notepad++) 打开 `tasks.csv`
2. 按照上述格式添加或修改任务
3. 保存文件

#### 方式二: 使用 WebUI
1. 在 WebUI 中处理一个视频
2. 处理完成后，会自动更新 `tasks.csv` 文件

### 3. 运行批量处理
1. 打开命令提示符，进入项目目录
2. 激活虚拟环境:
   ```bash
   venv\Scripts\activate
   ```
3. 运行处理管道:
   ```bash
   python -m youdub.run_pipeline --use-task-steps
   ```
4. 程序会自动读取 `tasks.csv` 并处理所有待处理任务

### 4. 检测任务完成情况
使用 `check_tasks_completion.py` 脚本可以检测所有任务的实际完成情况:

1. 打开命令提示符，进入项目目录
2. 激活虚拟环境:
   ```bash
   venv\Scripts\activate
   ```
3. 运行检测脚本:
   ```bash
   python check_tasks_completion.py
   ```
4. 查看输出结果，了解每个任务的实际状态

### 5. 从现有视频重建任务
如果 `tasks.csv` 文件丢失或损坏，可以使用 `recreate_tasks.py` 脚本从现有视频文件重建:

1. 打开命令提示符，进入项目目录
2. 激活虚拟环境:
   ```bash
   venv\Scripts\activate
   ```
3. 运行重建脚本:
   ```bash
   python recreate_tasks.py
   ```
4. 程序会扫描 `videos` 目录并重建 `tasks.csv` 文件

## 任务类型说明
### 1. download_only (仅下载)
- **功能**: 仅下载视频文件，不进行后续处理
- **适用场景**: 只想保存视频，不需要翻译和配音
- **输出文件**: 
  - `download.mp4` (视频文件)
  - `download.info.json` (视频信息)
  - `download.webp` (视频缩略图)

### 2. full_process (完整处理)
- **功能**: 执行完整的处理流程，包括下载、音频分离、语音识别、翻译、TTS、视频合成等
- **适用场景**: 需要将视频完整翻译和配音成中文
- **输出文件**: 
  - 下载阶段文件 (同上)
  - `audio.wav` (分离的音频)
  - `transcript.json` (语音识别结果)
  - `translation.json` (翻译结果)
  - `subtitles.srt` (字幕文件)
  - `video.mp4` (最终合成视频)

## 处理流程详解
### 步骤 0: 视频下载
- **功能**: 从视频 URL 下载视频文件
- **模块**: `step000_video_downloader_csv.py`
- **输出**: `download.mp4`, `download.info.json`, `download.webp`

### 步骤 1: 音频分离
- **功能**: 将视频中的人声与背景音乐分离
- **模块**: `step010_demucs_vr.py`
- **输出**: 分离后的音频文件

### 步骤 2: 语音识别
- **功能**: 将分离的人声转换为文本
- **模块**: `step020_whisperx_silero_vad.py`
- **输出**: `transcript.json`

### 步骤 3: 字幕翻译
- **功能**: 将识别的文本翻译成中文
- **模块**: `step030_translation_vad_qwen.py`
- **输出**: `translation.json`, `subtitles.srt`

### 步骤 4: 语音合成
- **功能**: 将翻译后的中文文本转换为语音
- **模块**: `step040_tts_vox_cpm_qwen.py` 等
- **输出**: 合成的语音文件

### 步骤 5: 视频合成
- **功能**: 将原视频、合成语音和字幕合成为最终视频
- **模块**: `step050_synthesize_video.py`
- **输出**: `video.mp4`

### 步骤 6: 生成视频信息
- **功能**: 生成视频的元数据信息
- **模块**: `step060_genrate_info.py`
- **输出**: 视频信息文件

### 步骤 7: 上传 Bilibili
- **功能**: 自动将最终视频上传到 Bilibili 平台
- **条件**: 需要配置 Bilibili 凭据

## 常见问题与解决方案
### Q: 视频下载失败怎么办？
A: 检查网络连接，确保 YouTube 可以访问。如果使用代理，需要在配置中设置代理。

### Q: 处理速度很慢怎么办？
A: 确保使用了 GPU 加速，降低批量大小，或减少同时处理的任务数。

### Q: 语音合成质量不好怎么办？
A: 尝试更换 TTS 模型，或调整 TTS 参数。可以在 WebUI 中选择不同的 TTS 模型。

### Q: tasks.csv 文件格式错误怎么办？
A: 确保所有字段都用逗号分隔，字符串用引号括起来。可以使用 Excel 或其他表格软件编辑。

### Q: 程序报错 "ModuleNotFoundError" 怎么办？
A: 确保所有依赖都已正确安装，可以重新运行 `setup_windows.bat` 或手动安装缺失的模块。

### Q: 如何查看处理日志？
A: 查看 `logs` 目录下的日志文件，或在命令行中查看输出信息。

## 技术细节
### AI 语音识别
基于 [WhisperX](https://github.com/m-bain/whisperX) 实现，支持精确的语音到文本转换和说话者分离。

### 大型语言模型翻译
支持多种翻译模型，包括 OpenAI GPT 系列和本地部署的模型，提供高质量的翻译结果。

### AI 声音克隆
集成了多种语音合成技术，包括 Coqui AI TTS、GPT-SoVITS 等，支持生成自然流畅的中文语音。

### 视频处理
使用 FFmpeg 进行视频处理，确保音视频同步和高质量的最终输出。

## 贡献指南
欢迎对 `YouDub-webui` 进行贡献。您可以通过以下方式参与：
1. 报告问题：在 [GitHub Issues](https://github.com/liuzhao1225/YouDub-webui/issues) 中提交问题
2. 提交代码：通过 [Pull Request](https://github.com/liuzhao1225/YouDub-webui/pulls) 提交改进
3. 改进文档：帮助完善 README.md 和其他文档

## 许可协议
`YouDub-webui` 遵循 Apache License 2.0。使用本工具时，请确保遵守相关的法律和规定，包括版权法、数据保护法和隐私法。未经原始内容创作者和/或版权所有者许可，请勿使用此工具。

## 支持与联系方式
如需帮助或有任何疑问，请通过以下方式联系我们：

- **GitHub Issues**: [提交问题](https://github.com/liuzhao1225/YouDub-webui/issues)
- **Discord 服务器**: [加入讨论](https://discord.gg/vbkYnN2Rrm)
- **微信群**: 扫描下方二维码加入

![WeChat Group](17ab2707ec88ddd8ad3fbd5c705d076b.png)

---

**祝您使用愉快！** 🎉
