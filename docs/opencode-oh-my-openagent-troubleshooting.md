# OpenCode + oh-my-openagent 插件排障记录

## 环境信息

| 项目 | 版本/值 |
|------|---------|
| OpenCode | 1.3.15 |
| oh-my-openagent | 3.15.2 |
| Bun (内嵌) | 1.3.11 |
| Node.js | 25.8.2 |
| OS | Linux (服务器) |
| 安装方式 | `npm install -g opencode-ai`（conda 环境内） |

## 问题现象

打开 OpenCode TUI 界面后看不到 Sisyphus agent，插件似乎未加载。启动日志中只显示：

```
INFO  service=plugin path=oh-my-openagent loading plugin
INFO  service=npm pkg=oh-my-openagent@latest installing package
```

之后 ~2.8 秒延迟，没有任何 "loaded" 或 "error" 日志，直接跳到后续初始化。

## 根因分析

### 插件加载机制

OpenCode 的插件加载流程如下：

1. 读取配置中的 `plugin` 列表
2. 对每个插件调用 `Npm.add(pkg)` 安装/解析
3. `Npm.add` 内部使用 `@npmcli/arborist`：
   - 先 `loadVirtual()` 尝试从已有的 `package-lock.json` 解析依赖树
   - 如果成功且找到包 → 直接返回入口点（不需要网络）
   - 如果失败 → `reify()` 从 npm registry 下载安装
4. 解析入口点后 `import()` 加载插件模块
5. 检测插件格式（V1 新格式 或 Legacy 函数格式）

### 关键路径

```
配置文件 (.opencode/opencode.json)
  → plugin: ["oh-my-openagent"]
    → Npm.add("oh-my-openagent@latest")
      → 安装目录: /root/.cache/opencode/packages/oh-my-openagent@latest/
      → arborist.loadVirtual() → 解析 package-lock.json
      → resolveEntryPoint() → dist/index.js
    → import(entrypoint)
    → readV1Plugin(mod, spec, "server", "detect")  // 先尝试 V1
    → getLegacyPlugins(mod)  // fallback 到 Legacy 格式
    → applyPlugin() → 注册工具和权限
```

### 实际问题

问题出在 **npm 依赖安装/缓存损坏**。具体表现为：

1. `/root/.cache/opencode/packages/oh-my-openagent@latest/` 目录中的 `node_modules` 不完整或 `package-lock.json` 损坏
2. `arborist.loadVirtual()` 静默失败（`.catch(() => {})` 吞掉了错误）
3. 随后 `arborist.reify()` 尝试从 npm registry 重新安装，但由于 **代理配置问题**（OpenCode 内嵌的 Bun 运行时不一定继承 shell 的 `http_proxy` 环境变量），网络请求超时
4. `reify()` 的 `.catch()` 抛出 `Npm.InstallFailedError`，被上层的 `error(candidate, _retry, stage, error49, resolved)` 捕获
5. 由于是 `"install"` 阶段的错误，日志输出为 `log12.error("failed to install plugin")`，但在某些情况下这个错误可能被进一步吞掉

## 修复方法

### 核心修复：重建插件的 npm 缓存

```bash
# 1. 清除损坏的缓存
rm -rf /root/.cache/opencode/packages/oh-my-openagent@latest/

# 2. 确保代理环境变量已设置（在启动 opencode 之前）
export http_proxy="http://127.0.0.1:10310"
export https_proxy="http://127.0.0.1:10310"
export all_proxy="socks5://127.0.0.1:10310"

# 3. 重新启动 opencode，让它自动重新安装插件
opencode
```

### 备选修复：手动预安装插件

如果自动安装仍然失败，可以手动在缓存目录中安装：

```bash
# 创建缓存目录
mkdir -p /root/.cache/opencode/packages/oh-my-openagent@latest
cd /root/.cache/opencode/packages/oh-my-openagent@latest

# 初始化并安装
echo '{}' > package.json
npm install oh-my-openagent@latest

# 验证安装
ls node_modules/oh-my-openagent/dist/index.js
```

### 同时确保项目级 .opencode 目录正确

```bash
cd /path/to/your/project/.opencode

# 检查 package.json 是否包含插件依赖
cat package.json
# 应该包含:
# {
#   "dependencies": {
#     "@opencode-ai/plugin": "1.3.15",
#     "oh-my-openagent": "^3.15.2"
#   }
# }

# 如果 node_modules 损坏，重新安装
rm -rf node_modules package-lock.json
npm install
```

## 验证插件是否加载成功

```bash
# 方法 1: 查看启动日志
opencode --print-logs --log-level DEBUG run "hello" 2>&1 | grep -E "call_omo_agent|oh-my|plugin"

# 成功标志:
# service=tool.registry status=started call_omo_agent
# service=tool.registry status=completed ... call_omo_agent

# 方法 2: 在 TUI 界面中确认
# 能看到 Sisyphus agent 和相关工具（task, explore, librarian, oracle 等）
```

## 代理环境下的注意事项

在需要代理访问外网的服务器上：

1. **必须在启动 opencode 之前设置代理环境变量**，因为 OpenCode 内部的 npm 客户端（arborist）和 `fetch()` 调用依赖这些环境变量
2. 建议将代理设置写入 shell profile（如 `~/.bashrc`）：
   ```bash
   export http_proxy="http://127.0.0.1:10310"
   export https_proxy="http://127.0.0.1:10310"
   export all_proxy="socks5://127.0.0.1:10310"
   ```
3. OpenCode 启动时会调用 `fetch("https://registry.npmjs.org/oh-my-openagent")` 检查插件是否有新版本（`Npm.outdated`），如果代理不通会导致额外的启动延迟

## 关键文件路径

| 路径 | 用途 |
|------|------|
| `~/.config/opencode/opencode.json` | 全局配置（provider、plugin 等） |
| `<project>/.opencode/opencode.json` | 项目级配置 |
| `~/.cache/opencode/packages/<pkg>@latest/` | 插件 npm 安装缓存 |
| `<project>/.opencode/node_modules/` | 项目级插件依赖 |
| `~/.config/opencode/node_modules/` | 全局插件依赖 |

## 排障速查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| 看不到 Sisyphus | 插件未加载 | 清除缓存 + 重装 |
| 启动卡顿 2-3 秒 | npm registry 访问慢 | 检查代理配置 |
| `failed to install plugin` | 网络不通 | 设置 http_proxy |
| `plugin incompatible` | 版本不匹配 | 升级 opencode 和 oh-my-openagent |
| `plugin has no server entrypoint` | 插件包损坏 | 删除缓存重装 |
