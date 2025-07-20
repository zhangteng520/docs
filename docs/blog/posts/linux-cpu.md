---
date: 2025-07-19
categories:
  - Linux
  - 性能优化
---

# Linux CPU 优化笔记
<!-- more -->
## 1. 查看 CPU 使用情况
- `top`：实时监控系统资源。
- `htop`：更友好的交互式界面（需安装）。
- `mpstat`：多核 CPU 统计。
- `vmstat`：虚拟内存和 CPU 状态。
- `sar`：系统活动报告。

## 2. 查找高 CPU 占用进程
- `ps aux --sort=-%cpu | head`
- `top` 或 `htop` 直接定位高占用进程。

## 3. 优化建议
### 3.1 进程/服务优化
- 停止不必要的服务和进程。
- 使用轻量级替代方案（如 nginx 替代 apache）。
- 合理设置进程优先级（`nice`/`renice`）。

### 3.2 程序优化
- 优化算法，减少无效循环和阻塞。
- 多线程/多进程合理分配。
- 使用异步或并发模型。

### 3.3 系统参数优化
- 调整内核参数（如 `/etc/sysctl.conf`）：
  - `kernel.sched_child_runs_first`、`kernel.sched_min_granularity_ns` 等。
- 合理配置 CPU 亲和性（`taskset`）。

## 4. 监控与报警
- 使用 `top`/`htop`/`glances` 实时监控。
- 配置 `zabbix`、`prometheus`、`grafana` 等监控平台。

## 5. 常用命令速查
```bash
top
htop
mpstat -P ALL 1
ps aux --sort=-%cpu | head
taskset -c 0,1 <command>
nice -n 10 <command>
renice -n 5 <pid>
```

## 6. 其他建议
- 定期升级内核和软件包，修复性能相关 bug。
- 关注硬件瓶颈（如内存、IO），避免 CPU 资源浪费。
- 代码层面避免死循环、资源泄漏。

---
如需深入分析，可结合 `strace`、`perf`、`systemtap` 等工具。
