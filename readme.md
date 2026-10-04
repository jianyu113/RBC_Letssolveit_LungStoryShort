# LungStoryShort

胸部 X 光影像研究项目，计划使用 ResNet-18 进行图像分类，使用 Faster R-CNN
定位肺部不透明区域，并比较分类、检测和分数融合的排序效果。

项目目前处于初始化阶段，训练和评估代码尚未实现。

## 环境要求

- 环境管理：Conda，环境名为 `lungstoryshort`。
- Python：`3.12.15`。
- 当前依赖版本来源：本地 macOS ARM64 的 Conda 环境。
- `requirements.txt` 固定项目直接依赖的版本，间接依赖由 pip 解析。

## 首次安装

先安装 Conda，再进入克隆后的仓库根目录（包含 `requirements.txt` 的目录）。
每位成员在自己的电脑上创建环境：

```bash
conda create -n lungstoryshort python=3.12.15 pip
conda activate lungstoryshort
python -m pip install -r requirements.txt
```

如果已经创建了 `lungstoryshort` 环境，跳过第一条命令。

检查依赖和 Python 解释器：

```bash
python --version
python -m pip check
python -c "import sys, torch, torchvision, pydicom; print(sys.executable); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__)"
```

`pip check` 应显示 `No broken requirements found.`，解释器路径应指向 Conda 的
`lungstoryshort` 环境。

## 日常使用

打开新终端后激活环境：

```bash
conda activate lungstoryshort
```

退出环境：

```bash
conda deactivate
```

VS Code 的 Python 解释器和 Notebook 内核都选择 Conda 的 `lungstoryshort` 环境。

## 团队协作

团队共享 `requirements.txt`，每位成员独立创建本地环境。
依赖文件更新后，在已激活的环境中重新安装：

```bash
python -m pip install -r requirements.txt
python -m pip check
```

新增或升级直接依赖时，在 `requirements.txt` 中记录实际使用的版本并验证安装。
这个文件不锁定所有间接依赖，也不保证不同平台的运行结果完全一致。
Windows、Linux 或 NVIDIA GPU 环境需要按
[PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)
选择对应构建，并核对与项目固定版本的兼容性。

## 当前目录

```text
RBC_Letssolveit_LungStoryShort/
├── dataset/
│   └── scripts/       # Data preparation code
├── models/            # Model code
├── .gitignore
├── Agents.md
├── readme.md
└── requirements.txt
```

`dataset/scripts/` 和 `models/` 目前为空，后续按实际开发需要添加代码。
本地虚拟环境、Python 缓存和 Notebook 检查点由 `.gitignore` 排除。
