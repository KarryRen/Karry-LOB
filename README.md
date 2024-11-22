# Karry-LOB
The advanced deeplob model for QR. 

本项目不仅涵盖 LOB 的模型部分，还包括使用 LOB 进行因子挖掘和建模的全套流程。

⚠️ 所有的代码均需要在根目录 `Karrt-LOB/` 运行 ！

```python
Karry-LOB/
├── Images # The images.
    ├── images.pptx # Images constructing PPT.
├── Ops # The operating code for unified managing.
    ├── __init__.py # The interface.
    ├── datatype.py # Datatype management.
    └── util.py # Util functions.
├── TensorEngineering # The package for building up factors.
    ├── tensor_engineering
        ├── tensor_construction_algo # The algorithms for tensor construction.
            ├── __init__.py # The interface of tensor construction algorithms.
            ├── tensor_construction_base.py # The base class of tensor construction.
            ├── base # The base algorithms
                ├── __init__.py # The interface of base algorithms.
                ├── base_hftlabel.py # HFTLabel algorithms.
                └── base_lob.py # LOB algorithms.
            └── ... # Other algorithms for tensor construction.
        ├── io.py # IO functions.
        └── util.py # Util functions.
    ├── run_construction.sh # The entrance for construction.
    ├── run_test_construction.sh # The entrance for test construction.
    ├── main # Main function for tensor engineering.
        └── construction.py # The main function for construction.
    └── test # Test function for tensor engineering.
        └── test_construction.py # The test function for construction.
├── DeepLOB # The package for modeling the LOB fatcors.
    ├── source # The shell source files.
        ├── run_deeplob.sh # Entrance for running deeplob.
        └── train_test_deeplob.sh
└── 
```

## 1. Tensor Engineering

**因子挖掘模块，系统化使用原始的 LOB 交易数据计算、构建和存储因子数据（Construction），后续可能会在 Construction 的基础上进一步进行因子信息提取（Extraction），也即对基础因子值进行更多有意义的变换**。

这个部分之所以叫做 Tensor Engineering，是因为后续的所有建模流程都是基于深度学习的，Input 都是 Tensor，因此就将构造和处理 Input 的过程称为 Tensor Engineering。

### 1.1 Code Idea

> 将编码思路清晰地列出来更有助于理解，同时对其中的一些依赖也做详细的说明。
>
> **【Dependency】**注意此部分所用到的核心依赖有：
>
> - 模块高度依赖 [**xarray**](https://docs.xarray.dev/en/stable/index.html) 工具包，该工具包是基于 numpy 开发的高维数据构造包，构造出来的数据也有极高的可读性和可操作性。但是该包有比较明显的缺点，就是在 Windows 系统上极难安装，需要一定的耐心。因为搭建包时使用的是 Linux 系统，所以没有遇到包安装问题。
> - 所涉及的金融数据部分大都依赖 [**akshare**](https://akshare.akfamily.xyz/index.html) 工具包，该工具包以爬虫为基础，收集了众多金融数据供开源使用。比如获取交易日期等功能。
>
> **【Code Pipeline】**此处 Code 的编写是以 Shell 脚本为入口，各个 `.py` 文件进行串联

#### 1.1.1 Construction

**在构造核心上**：此处编码的核心在于构造了一个大而全的基类 `ConstructionAlgoBase`，以此基类向外拓展各种算法计算类。基于 `__init__()` 函数载入各种属性，通过重载 `cal_fea()` 函数实现不同的计算模式。

**在并行计算上**：因为计算资源受限以及高性能底层架构不完善，所以暂时没有使用高性能框架进行并行计算，后续可以尝试搭建基于 `sbatch` 的高性能计算平台，合理调用资源，高并行快速执行任务。但在代码中可以并行的部分，我使用了 `multiprocessing` 加速。

**在代码测试上**：引入了 `test/test_construction.sh` 文件，对特征生成算法做测试管理。同时通过设置 `n_process` 来控制全局测试。

### 1.2 How to use

**Step 1. 构建数据存储路径**：在本地创建一个数据存储文件夹，如 `TensorEngineering/`。

**Step 2. 修改存储路径**：将代码 `TensorEngineering/tensor_engineering/io.py` 中的 `TensorEngineeringRootDir` 宏变量修改为上述创建的绝对路径。

**Step 3. 进行特征生成**：在根目录 `Karry-LOB` 下运行 `./TensorEngineering/run_construcion.sh` 生成特征。

### 1.3 Result

基于本流程，可以得到如下的数据结构

```python
TensorEngineering/
├── code_1
    ├── date_1
        ├── feature_1.h5
        ├── feature_2.h5
        ├── ...
        └── feature_n.h5
    ├── date_2
    └── date_n
├── code_2
├── ...
└── code_n
```

### 1.4 Features

#### 1.4.1 LOB

基于 `base.base_lob` 获取最细粒度的 Limit Order Book 数据，每天一个 `LOB.h5` 文件，文件内存储的是 `xarray` 的数据。

```python
out_coords = {
    "T": list(range(28800)),
    "D": ["Bid", "Ask"],
    "L": list(range(1, 6)),
    "F": ["Price", "Volume"]
}  # the out coords
```

#### 1.4.2 HFTLabel

Label 其本质也可以看作一个 Feature。基于 `base.base_hftlabel` 获取每天的高频交易标签。核心目前有两种计算思路：

- **Way 1.**  (`rolling().mean()`)
  $$
  \text{Label} = \log(\frac{\frac{1}{k}\sum_{i=0}^k\text{MidPrice}_{t+i})}{\text{MidPrice}_{t}})
  $$

- **Way 3.** 
  $$
  \text{Label} = \log(\frac{\text{MidPrice}_{t+k}}{\text{MidPrice}_{t}})
  $$

Way1. 相比于 Way 3. 波动更小，但是在实际使用中我们一般会使用 Way 3.

```python
out_coords = {
    "T": list(range(28800)),
    "F": [
        f"label_ret_{self.label_step}_way1", f"label_ret_{self.label_step}_way3",
        f"label_weight_ret_{self.label_step}_way1", f"label_weight_ret_{self.label_step}_way3"
    ]
}  # the out coords
```

Label 的计算存在一定的细节，尤其是 Label Weight 的获取。在后续的建模中会省略掉一些无效的 tick 点，主要是通过 Label 的 Weight 来控制，在生成的时候主要考虑了交易状态。

## 2. DeepLOB

**数据建模模块，基于上述生成的因子，开始进行建模**。

### 2.1 Code Idea

此部分的编码思路较为直接，完全采用标准的 pytorch 框架进行搭建。仍然是采用 `shell` 脚本作为程序入口，使用 `.py` 文件进行串联。

- 从 `shell` 脚本出发，通过 `task_id` 来控制 Train，Valid 以及 Test 的日期区间，以及一些诸如随机种子的配置
- 在以下三组文件的综合作用下完成 `config` 的生成：`task_util.py`，`task_config.yaml`，`configs/`
- 使用 `datadict` 对原始数据做 `dict ` 管理，将其载入内存，同时进行数据检验和一些数据变化
- 使用可以进行 Freeze 和 Pretrain 的思路进行模型初始化
- 模型训练和测试做拆分和融合
- 最后做统一的 eval











