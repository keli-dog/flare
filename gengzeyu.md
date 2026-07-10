# FLARE tests_seen 四路并行 Eval 记录

服务器路径：`/media/ubuntu/Student/gengzeyu/flare`  
DN（结果标识）：`flare_paper`  
Split：`tests_seen`（共 1533 条，索引 0~1532）

> **推荐直接用 `nohup bash eval.sh ... &` 命令**，不要用脚本（远程环境下脚本易因 conda/nohup 子进程问题秒退）。

---

## 1. 环境准备（每次跑前）

```bash
cd /media/ubuntu/Student/gengzeyu/flare
conda activate flare
export ALFRED_ROOT=/media/ubuntu/Student/gengzeyu/flare/alfred
export FILM=/media/ubuntu/Student/gengzeyu/flare
export DISPLAY=:1
mkdir -p results/shard_logs results/logs results/leaderboard results/analyze_recs
```

Xvfb 若已在 `:1` 运行，**不要重复启动**（会报 `server already running`）。

---

## 2. 全量 tests_seen 四路并行（1533 条）

将 `[0, 1533)` 均分为 4 片（每片约 383 条）。**从 204 续跑**时跳过已完成的前 204 条：

| 片 | 范围 | 条数 |
|----|------|------|
| s1 | 204 ~ 537 | 333 |
| s2 | 537 ~ 870 | 333 |
| s3 | 870 ~ 1203 | 333 |
| s4 | 1203 ~ 1533 | 330 |

### 启动命令

```bash
cd /media/ubuntu/Student/gengzeyu/flare
conda activate flare
export ALFRED_ROOT=/media/ubuntu/Student/gengzeyu/flare/alfred
export FILM=/media/ubuntu/Student/gengzeyu/flare
export DISPLAY=:1
mkdir -p results/shard_logs

CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 204  537 flare_paper > results/shard_logs/s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 537  870 flare_paper > results/shard_logs/s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 870 1203 flare_paper > results/shard_logs/s3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 1203 1533 flare_paper > results/shard_logs/s4.log 2>&1 &
```

### 首次从 0 开始（完整 1533）

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 0   383 flare_paper > results/shard_logs/s1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 383  766 flare_paper > results/shard_logs/s2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 766 1149 flare_paper > results/shard_logs/s3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 1149 1533 flare_paper > results/shard_logs/s4.log 2>&1 &
```

### 实际运行情况（2026-07-08）

- 旧单路 `[0, 1533)` 先跑了约 **204 条**后中断
- 四路 `[204, 1533)` 并行跑完 **s1/s2/s4**
- **s3 `[870, 1203)` 在四路并行时 OOM/子进程崩溃**，只完成 870~873，缺 874~1202（329 条）

---

## 3. s3 补跑四路并行（874 ~ 1203）

| 片 | 范围 | 条数 |
|----|------|------|
| r1 | 874 ~ 957 | 83 |
| r2 | 957 ~ 1040 | 83 |
| r3 | 1040 ~ 1123 | 83 |
| r4 | 1123 ~ 1203 | 80 |

### 启动命令

```bash
cd /media/ubuntu/Student/gengzeyu/flare
conda activate flare
export ALFRED_ROOT=/media/ubuntu/Student/gengzeyu/flare/alfred
export FILM=/media/ubuntu/Student/gengzeyu/flare
export DISPLAY=:1
mkdir -p results/shard_logs

CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 874  957 flare_paper > results/shard_logs/s3_r1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 957 1040 flare_paper > results/shard_logs/s3_r2.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 1040 1123 flare_paper > results/shard_logs/s3_r3.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 1123 1203 flare_paper > results/shard_logs/s3_r4.log 2>&1 &
```

**不用删旧日志**。补跑会写新文件（`s3_r*.log`、`from_874_to_*` 等），与旧 s3 部分结果（870~873）不冲突。

### 若四路秒退（显存不足）→ 改两路

```bash
pkill -f flare_paper; sleep 2
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 874 1087 flare_paper > results/shard_logs/s3_r1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup bash eval.sh tests_seen 1087 1203 flare_paper > results/shard_logs/s3_r2.log 2>&1 &
```

---

## 4. 监控

### 进程 / GPU

```bash
pgrep -af flare_paper
nvidia-smi
```

正常四路：约 4 个 `bash eval.sh` + 4 个 `python main.py`，显存 **20~35GB**，GPU-Util **70%+**。

### 日志

```bash
tail -f results/shard_logs/s1.log          # 全量四路
tail -f results/shard_logs/s3_r1.log       # s3 补跑
```

### 全量进度（0~1532）

```bash
python - <<'PY'
import glob, pickle
eps=set()
for p in glob.glob("results/analyze_recs/tests_seen_anaylsis_recs_from_*_flare_paper.p"):
    for r in pickle.load(open(p,"rb")):
        if r.get("number_of_this_episode") is not None:
            eps.add(int(r["number_of_this_episode"]))
print(f"completed: {len(eps)} / 1533")
PY
```

### 仅 s3 补跑进度（874~1202）

```bash
python - <<'PY'
import glob, pickle, re
LO, HI = 874, 1203
eps=set()
for p in glob.glob("results/analyze_recs/tests_seen_anaylsis_recs_from_*_flare_paper.p"):
    m=re.search(r"from_(\d+)_", p)
    if not m or int(m.group(1)) < LO:
        continue
    for r in pickle.load(open(p,"rb")):
        if r.get("number_of_this_episode") is not None:
            ep=int(r["number_of_this_episode"])
            if LO <= ep < HI:
                eps.add(ep)
print(f"s3 resume: {len(eps)} / {HI - LO}")
PY
```

### 各分片文件明细

```bash
python - <<'PY'
import glob, pickle, re
for p in sorted(glob.glob("results/analyze_recs/tests_seen_anaylsis_recs_from_*_flare_paper.p")):
    recs=pickle.load(open(p,"rb"))
    eps=set()
    for r in recs:
        if r.get("number_of_this_episode") is not None:
            eps.add(int(r["number_of_this_episode"]))
    m=re.search(r"from_(\d+)_to_(\d+)", p)
    span=f"[{m.group(1)},{m.group(2)})" if m else "?"
    print(f"{p.split('/')[-1]}: {len(eps)} unique eps, {len(recs)} records, span {span}")
PY
```

### 还缺哪些 episode

```bash
python - <<'PY'
import glob, pickle
eps=set()
for p in glob.glob("results/analyze_recs/tests_seen_anaylsis_recs_from_*_flare_paper.p"):
    for r in pickle.load(open(p,"rb")):
        if r.get("number_of_this_episode") is not None:
            eps.add(int(r["number_of_this_episode"]))
missing=[i for i in range(1533) if i not in eps]
print(f"missing: {len(missing)}")
if missing:
    print(f"  range: {missing[0]} ~ {missing[-1]}")
PY
```

### actseqs 条数（leaderboard 用）

```bash
python - <<'PY'
import glob, pickle
total=0
for p in sorted(glob.glob("results/leaderboard/actseqs_test_seen_flare_paper_*.p")):
    n=len(pickle.load(open(p,"rb")))
    total+=n
    print(f"{p.split('/')[-1]}: {n}")
print(f"actseqs total: {total} / 1533")
PY
```

### 本地 SR（all_completed，非论文口径）

```bash
python - <<'PY'
import glob, pickle
by_ep={}
for p in glob.glob("results/analyze_recs/tests_seen_anaylsis_recs_from_*_flare_paper.p"):
    for r in pickle.load(open(p,"rb")):
        if r.get("number_of_this_episode") is not None:
            by_ep[int(r["number_of_this_episode"])]=r
succ=sum(1 for r in by_ep.values() if r.get("success"))
print(f"all_completed SR: {succ}/{len(by_ep)} = {100*succ/len(by_ep):.2f}%" if by_ep else "no data")
PY
```

### 停止

```bash
pkill -f flare_paper
```

---

## 5. 结果文件位置

| 类型 | 路径 |
|------|------|
| 手动四路日志 | `results/shard_logs/s*.log`、`s3_r*.log` |
| 正式 eval 日志 | `results/logs/log_tests_seen_from_{from}_to_{to}_flare_paper.txt` |
| 进度 pickle | `results/analyze_recs/tests_seen_anaylsis_recs_from_*_flare_paper.p` |
| Leaderboard 动作序列 | `results/leaderboard/actseqs_test_seen_flare_paper_{from}_{to}.p` |

`tests_seen` **不写** `results/successes/`、`results/fails/`（test 模式）。

---

## 6. 补跑完后：合并 actseqs → 提交官方（拿真实 SR）

### 重要：不用合并日志

| 文件 | 用途 | 是否提交官方 |
|------|------|-------------|
| `results/logs/*.txt`、`shard_logs/*.log` | 本地排查 | ❌ 否 |
| `results/analyze_recs/*.p` | 本地进度 / all_completed | ❌ 否 |
| **`results/leaderboard/actseqs_*.p`** | **low-level 动作序列** | ✅ **合并后提交** |

官方 ALFRED 在服务器上**回放 actseqs**，用 PDDL 判 goal → 这才是论文 Table 1 的 SR。

### 预期有哪些 actseqs 分片（1533 齐了之后）

```
results/leaderboard/actseqs_test_seen_flare_paper_0_1533.p      # 旧单路（约204条）
results/leaderboard/actseqs_test_seen_flare_paper_204_537.p
results/leaderboard/actseqs_test_seen_flare_paper_537_870.p
results/leaderboard/actseqs_test_seen_flare_paper_870_1203.p     # 仅870~873（旧s3）
results/leaderboard/actseqs_test_seen_flare_paper_874_957.p    # s3补跑
results/leaderboard/actseqs_test_seen_flare_paper_957_1040.p
results/leaderboard/actseqs_test_seen_flare_paper_1040_1123.p
results/leaderboard/actseqs_test_seen_flare_paper_1123_1203.p
```

### Step 1：确认 actseqs 齐了

```bash
cd /media/ubuntu/Student/gengzeyu/flare
python - <<'PY'
import glob, pickle
total=0
trials=set()
for p in sorted(glob.glob("results/leaderboard/actseqs_test_seen_flare_paper_*.p")):
    entries=pickle.load(open(p,"rb"))
    for item in entries:
        if item and item!="":
            key=list(item.keys())[0]
            trials.add(key[1] if isinstance(key,tuple) else key)
    total+=len(entries)
    print(f"{p.split('/')[-1]}: {len(entries)}")
print(f"entries={total}, unique trials={len(trials)} / 1533")
PY
```

`unique trials` 必须 **= 1533** 才能提交。

### Step 2：合并成 JSON（推荐，只跑 seen 也能用）

```bash
python scripts/merge_tests_seen_leaderboard.py --dn flare_paper
```

输出：`leaderboard_jsons/tests_actseqs_flare_paper.json`

先检查不写入：

```bash
python scripts/merge_tests_seen_leaderboard.py --dn flare_paper --dry-run
```

> 原仓库 `utils/leaderboard_script.py` 还要求 `tests_unseen` 1529 条，**只跑 seen 会 assert 失败**。用上面的新脚本即可。

### Step 3：拷到本机（服务器不能打开浏览器时）

在**本机 Windows** 执行：

```bash
scp ubuntu@<服务器IP>:/media/ubuntu/Student/gengzeyu/flare/leaderboard_jsons/tests_actseqs_flare_paper.json .
```

或用 `rsync` / WinSCP 均可。

### Step 4：提交 ALFRED 官方 Leaderboard

1. 打开：https://leaderboard.allenai.org/alfred/submissions/public  
2. 登录 / 注册 AI2 账号  
3. 上传 `tests_actseqs_flare_paper.json`  
4. 等官方回放评测（通常几分钟到几小时）  
5. 结果页 **Test Seen Success Rate** = 论文 Table 1 口径  

说明：
- 当前 JSON 里 `tests_unseen` 为空 → 只能拿 **Test Seen** 分数  
- 若要 **Test Unseen**，需另跑 `tests_unseen` 1529 条再合并

### 流程图

```
四路 eval.sh 并行
    ↓
results/leaderboard/actseqs_test_seen_flare_paper_{from}_{to}.p  （多分片）
    ↓
python scripts/merge_tests_seen_leaderboard.py --dn flare_paper
    ↓
leaderboard_jsons/tests_actseqs_flare_paper.json  （一个文件）
    ↓
scp 到本机 → 上传 ALFRED 官网
    ↓
官方回放 → 真实 SR（~40%）
```

---

## 7. 指标口径说明

| 指标 | 来源 | 含义 |
|------|------|------|
| `analyze_recs['success']` | 本地 | `all_completed`：MMP 计划全部高层动作执行完 |
| 论文 Table 1 Test Seen | 官方 leaderboard | PDDL goal 满足，约 ~40% |

本地 `all_completed`（约 53%）**高于**论文 SR 是正常的，口径不同。

---

## 8. Git 提交日志/结果（可选）

```bash
git add results/logs/log_tests_seen_*_flare_paper.txt results/shard_logs/s*.log
git add -f results/analyze_recs/tests_seen_*_flare_paper.p
git commit -m "tests_seen eval logs and results"
git push origin master
```

`results/leaderboard/` 在 `.gitignore` 中，一般不提交。
