# RAG 系统：面向书籍的检索与生成
本项目是一个围绕中医古籍构建的检索增强生成（RAG）系统,采用语义分块。

## 主要功能
通过 python main.py 配合参数,可以生成指定的pdf或者txt文件的RAG系统.
同时rag.py提供了接口进行查询.
## 安装与设置
### 克隆代码库:
`git clone https://github.com/madao2333/RAG.git`
### 创建并配置环境:
建议使用 conda 或 venv 创建独立的Python环境。(作者使用的是3.11版本)
`conda create -n rag python=3.11`
`conda activate rag`
### 安装依赖:
在项目根目录下，使用以下命令安装所有必需的库。
`pip install -r requirements.txt`
### 准备数据与模型:
将您的PDF文件（如《温病学》）放置在 config.py 中 pdf_input_dir 指定的目录（默认为 ../pdf）。
下载您需要的基础语言模型（如 text2vec-base-chinese），然后放到models目录下面。
## 使用说明
### 构建RAG
将需要构建RAG的pdf文件或者txt文件放在pdf目录下(名字中不要出现括号),然后在rag目录下就可以执行
`python main.py ../pdf/(文件名) -o (输出的目录名)`
### 在模型推理的时候被调用
提供的接口在rag.py中,初始化的方法有两种:
1.调用get_rag_system_non_interactive,这里调用的rag是在config.py中的DEFAULT_RAG_NAME
2.调用get_rag_system,如果之前没有设置过rag_system,那么用户可以从config中所有AVAILABLE_RAGS中选取想要的RAG(只要之前用python main.py的方法执行之后,会自动添加到参数中)
之后只需要调用rag.py中的simple_rag_query函数,就可以进行查询,(multi.py和inference_LoRA.py是一个在模型推理的时候调用RAG的例子)