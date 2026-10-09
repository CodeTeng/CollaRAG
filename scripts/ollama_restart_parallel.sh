#!/usr/bin/env bash
# Restart ollama with multi-request parallelism for ingestion speed.
#
# 默认 ollama serve 串行处理请求 — AdaGraph / IntentAgent / vanilla ingest
# 在 LLM extractor 阶段会有大量并发 LLM 调用（chunker → extractor），
# 串行模式下并发参数等于摆设。这个脚本：
#   1) 杀掉已存在的 ollama serve（保留模型权重缓存）
#   2) 设 OLLAMA_NUM_PARALLEL=4 让 ollama 同时处理 4 路请求
#   3) 设 OLLAMA_KEEP_ALIVE=30m 让 qwen3:8b 30 分钟内不卸载
#   4) 后台启动，日志输出到 ~/.ollama/logs/server.log
#
# 与之配套的 yaml 设置：configs/ollama_colla_rag_only.yaml 的
#   ingestion.extractor.params.extract_workers: 4
# 两边并发数对齐才有效。

set -e

PARALLEL="${OLLAMA_NUM_PARALLEL:-4}"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

echo ">>> killing existing ollama processes ..."
pkill -f "ollama serve" || true
pkill -f "ollama runner" || true
sleep 1

echo ">>> starting ollama serve with OLLAMA_NUM_PARALLEL=$PARALLEL OLLAMA_KEEP_ALIVE=$KEEP_ALIVE ..."
mkdir -p "$HOME/.ollama/logs"
nohup env \
    OLLAMA_NUM_PARALLEL="$PARALLEL" \
    OLLAMA_KEEP_ALIVE="$KEEP_ALIVE" \
    OLLAMA_FLASH_ATTENTION=1 \
    ollama serve > "$HOME/.ollama/logs/server.log" 2>&1 &

# 等服务起来（健康检查）
for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo ">>> ollama is up (PID $(pgrep -f 'ollama serve' | head -1))"
        echo ">>> models locally available:"
        curl -s http://localhost:11434/api/tags | python3 -c "import sys,json;d=json.load(sys.stdin);[print(f'    - {m[chr(34)+chr(110)+chr(97)+chr(109)+chr(101)+chr(34).strip()]}') for m in d.get('models',[])]" 2>/dev/null || curl -s http://localhost:11434/api/tags
        echo ">>> log: tail -f $HOME/.ollama/logs/server.log"
        exit 0
    fi
done

echo ">>> ERROR: ollama did not become healthy within 10s; check $HOME/.ollama/logs/server.log" >&2
exit 1
