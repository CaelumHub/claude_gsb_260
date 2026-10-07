"""规范化批量任务：后台并行处理，单篇卡住不拖垮整批。

关键设计
--------
* 用 :class:`concurrent.futures.ThreadPoolExecutor` 并行处理多篇文档，
  每篇文档独立 try/except，异常只记为该文档失败。
* **超时不阻塞**：为每个任务记录提交时刻，``timeout`` 内未完成即记为
  失败并从「在途集合」移除，但**不去 join 那个慢线程**。这里刻意不使用
  ``with ThreadPoolExecutor``——其退出时的 ``shutdown(wait=True)`` 会等待
  已经超时的慢任务，反而让一篇卡住拖垮整批。工作线程为 daemon，
  慢任务跑完后结果被丢弃，不影响本批的完成与返回。
* 受控提交（在途任务数 ≈ 2×worker），避免一次性把大量超长文档全部入队。
* 任务状态实时写入 ``normalize_batch`` 分片存储，前端轮询 GET 查进度。
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from typing import Optional

from nlp import get_normalizer
from nlp.normalize import NormalizeConfig

# 轮询间隔
_POLL_INTERVAL = 0.1


def _new_job_id() -> str:
    return "normbatch_" + uuid.uuid4().hex[:12]


def submit_normalize_batch(registry, documents: list[dict],
                           config: NormalizeConfig,
                           max_workers: int = 4,
                           timeout: float = 30.0) -> dict:
    """提交一个批量规范化任务，立即返回初始状态。"""
    store = registry.task("normalize_batch")
    job_id = _new_job_id()
    record = {
        "id": job_id,
        "status": "pending",
        "total": len(documents),
        "done": 0,
        "succeeded": 0,
        "failed": 0,
        "max_workers": max_workers,
        "timeout": timeout,
        "config": config.to_dict(),
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "documents": [
            {"corpus_id": d.get("corpus_id"), "name": d.get("name", ""),
             "ok": None, "error": None, "change_count": 0, "result": None}
            for d in documents
        ],
    }
    store.insert(record)

    thread = threading.Thread(
        target=_run_batch,
        args=(registry, job_id, documents, config, max_workers, timeout),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "total": len(documents), "status": "pending"}


def _run_batch(registry, job_id: str, documents: list[dict],
               config: NormalizeConfig, max_workers: int,
               timeout: float) -> None:
    store = registry.task("normalize_batch")
    lock = threading.Lock()
    normalizer = get_normalizer()
    total = len(documents)
    state = {"done": 0, "succeeded": 0, "failed": 0}

    def _patch(**fields):
        with lock:
            record = store.get(job_id)
            if record:
                record.update(fields)
                store.update(job_id, record)

    def _record(index: int, doc_patch: dict, ok: bool) -> None:
        with lock:
            record = store.get(job_id)
            if record:
                record["documents"][index].update(doc_patch)
                state["done"] += 1
                if ok:
                    state["succeeded"] += 1
                else:
                    state["failed"] += 1
                record["done"] = state["done"]
                record["succeeded"] = state["succeeded"]
                record["failed"] = state["failed"]
                store.update(job_id, record)

    def _work(index: int, text: str) -> tuple[int, dict]:
        result = normalizer.normalize(text, config)
        payload = result.to_dict()
        change_count = sum(len(p["changes"]) for p in payload["paragraphs"])
        return index, {
            "ok": True, "error": None,
            "change_count": change_count, "result": payload,
        }

    _patch(status="running", started_at=time.time())
    workers = max(1, max_workers)

    tasks: "queue.Queue[Optional[tuple[int, float]]]" = queue.Queue()
    for idx, doc in enumerate(documents):
        tasks.put((idx, time.monotonic()))
    for _ in range(workers):
        tasks.put(None)  # 每个 worker 的停止信号

    def _worker() -> None:
        # 守护线程：即使某篇卡死，也不会阻止进程退出，更不阻塞整批收尾。
        while True:
            item = tasks.get()
            if item is None:
                tasks.task_done()
                return
            index, submitted = item
            if time.monotonic() - submitted > timeout:
                _record(index, {
                    "ok": False,
                    "error": f"排队超时（>{timeout}s），已跳过，不影响其它文档",
                    "change_count": 0, "result": None}, False)
                tasks.task_done()
                continue
            try:
                res_idx, patch = _work(index, documents[index].get("text", ""))
                # 若监控方已因超时把该篇记为失败，则丢弃迟到结果，不覆盖
                with lock:
                    already = store.get(job_id)
                    stale = bool(already and already["documents"][index]["ok"] is not None)
                if stale:
                    tasks.task_done()
                    continue
                _record(res_idx, patch, True)
            except Exception as exc:  # noqa: BLE001
                _record(index, {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "change_count": 0, "result": None}, False)
            finally:
                tasks.task_done()

    pool = [threading.Thread(target=_worker, daemon=True) for _ in range(workers)]
    for t in pool:
        t.start()

    # 监控：超时任务直接记失败（其线程若仍在跑，完成后结果被丢弃）。
    deadline_index = {idx: time.monotonic() + timeout for idx in range(total)}
    while state["done"] < total:
        time.sleep(_POLL_INTERVAL)
        with lock:
            record = store.get(job_id)
            if not record:
                break
            for idx, doc in enumerate(record["documents"]):
                if doc["ok"] is None and time.monotonic() > deadline_index[idx]:
                    state["done"] += 1
                    state["failed"] += 1
                    doc.update({
                        "ok": False,
                        "error": f"处理超时（>{timeout}s），已跳过，不影响其它文档",
                        "change_count": 0, "result": None})
            record["done"] = state["done"]
            record["succeeded"] = state["succeeded"]
            record["failed"] = state["failed"]
            store.update(job_id, record)

    _patch(status="finished", finished_at=time.time())
