# Metric Viewer 使用说明

Metric Viewer 用于查看 Terrabox/Evolution 实验的单实验指标、运行进度，以及两个实验共同完成任务的配对对比。启动入口位于仓库根目录的 `run_metric_viewer.py`，不要直接运行 `metricViewer/app.py`。

## 当前状态

截至 2026-07-11，Metric Viewer 已启动：

- tmux session：`metric_viewer`
- Python PID：`1959038`
- 监听地址：`0.0.0.0:8765`
- 日志：`tmp/metric_viewer.log`

检查当前是否运行：

```bash
tmux ls | rg '^metric_viewer:'
ps -eo pid,ppid,stat,etime,cmd | rg 'run_metric_viewer.py' | rg -v rg
ss -ltnp | rg ':8765\b'
```

## 推荐启动方式

在仓库根目录执行：

```bash
cd /data1/yuhongjie2/terrabox && PYTHONPATH=src:. python run_metric_viewer.py --host 127.0.0.1 --port 8765
```

默认端口是 `8765`。显式使用 `127.0.0.1` 可以避免服务直接暴露到服务器外网，配合 SSH 隧穿访问。

如果端口已被占用，可换用其它端口：

```bash
cd /data1/yuhongjie2/terrabox && PYTHONPATH=src:. python run_metric_viewer.py --host 127.0.0.1 --port 8766
```

`run_metric_viewer.py` 优先使用 FastAPI/uvicorn；如果当前 Python 环境没有安装 `uvicorn`，会自动回退到标准库 HTTP server，因此不需要为了基础查看功能额外安装依赖。`--reload` 仅在安装了 uvicorn 时可用。

## 在 tmux 中后台启动

```bash
tmux new-session -d -s metric_viewer "cd /data1/yuhongjie2/terrabox && PYTHONPATH=src:. python run_metric_viewer.py --host 127.0.0.1 --port 8765 > tmp/metric_viewer.log 2>&1"
```

查看 tmux：

```bash
tmux attach -t metric_viewer
```

实时查看日志：

```bash
tail -f /data1/yuhongjie2/terrabox/tmp/metric_viewer.log
```

如果已经存在同名 tmux，不要重复启动；先用上面的状态检查命令确认旧服务是否仍然健康。

## SSH 隧穿到本地浏览器

在自己的本地电脑上执行：

```bash
ssh -L 8765:127.0.0.1:8765 yuhongjie@<服务器地址>
```

保持该 SSH 连接开启，然后在本地浏览器访问：

```text
http://127.0.0.1:8765
```

如果服务器端使用了 `8766`，对应改成：

```bash
ssh -L 8766:127.0.0.1:8766 yuhongjie@<服务器地址>
```

也可以使用 VS Code Remote SSH 的 Ports 面板转发服务器端口 `8765`。

## 页面功能

- 单实验模式：选择一个实验，查看进度、状态分布、工具指标、分类别 F1、资源消耗等指标。
- 对比模式：选择任意两个实验，只对两边都存在的同一批 `task_id` 做配对比较。
- Static Prompt：查看当前所选实验能够解析到的静态提示词。
- 实验列表来自各 adapter 对 `results/` 目录的扫描；逐任务 JSON 仍是指标计算的权威来源。

## 停止服务

如果服务由 tmux 启动：

```bash
tmux kill-session -t metric_viewer
```

停止后确认端口和进程已释放：

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'run_metric_viewer.py' | rg -v rg
ss -ltnp | rg ':8765\b'
```

不要直接 kill 名称相似但无法确认归属的 Python 进程；先通过命令行、PID 和端口确认属于 Metric Viewer。
