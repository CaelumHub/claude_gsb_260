"""批量规范化执行器：语料分片存放、长短不一，单篇卡住不能拖垮整批。

与 :class:`pipeline.engine.PipelineEngine.run_batch` 同样的容错思路：

* 每篇文档独立处理，异常被捕获并作为失败项返回；
* 线程池 + 逐篇独立计时轮询，某篇超过 timeout 立即只标记该篇失败
  （worker 为 daemon 线程，池退出时不 join，卡死线程不阻塞整批）；
* 分块调度，避免一次性提交几千篇导致内存 / 文件句柄峰值；
* 超长文本按 ``max_chars`` 截断并标记；
* 支持传入逐篇的人工确认决策（``{doc_id: decisions}``），实现
  「逐处确认 -> 回流 -> 批量落地」的闭环。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .normalize import NormalizationSpec, normalize, apply_decisions

# 单篇文本长度硬上限（字符），超过直接截断并标记，防止恶意/异常输入
MAX_DOC_CHARS = 200_000
DEFAULT_TIMEOUT = 30.0


class _NonBlockingPool(ThreadPoolExecutor):
    """退出时不 join worker 的线程池。

    被超时跳过的任务线程仍在后台运行，若在 ``with`` 退出时 join 会把
    「单篇卡住」重新变成「整批等待」。这里把 worker 创建为 daemon
    线程，并在关闭时 ``wait=False``，真正做到卡死的单篇不拖累整批。
    """

    def _adjust_thread_count(self):  # type: ignore[override]
        import threading
        import weakref
        from concurrent.futures.thread import _threads_queues, _worker

        if self._idle_semaphore.acquire(timeout=0):  # type: ignore[attr-defined]
            return
        num_threads = len(self._threads)  # type: ignore[attr-defined]
        if num_threads >= self._max_workers:
            return
        work_queue = self._work_queue  # type: ignore[attr-defined]

        def weakref_cb(_, q=work_queue):  # 与 CPython 内部回调等价
            q.put(None)

        t = threading.Thread(
            name=f"normpool_{num_threads}", daemon=True, target=_worker,
            args=(weakref.ref(self, weakref_cb),
                  work_queue, self._initializer, self._initargs))  # type: ignore[attr-defined]
        t.start()
        self._threads.add(t)  # type: ignore[attr-defined]
        _threads_queues[t] = work_queue

    def __exit__(self, exc_type, exc, tb):
        self.shutdown(wait=False)


def process_one(text: str,
                spec: Optional[dict] = None,
                decisions: Optional[dict | list] = None,
                *,
                max_chars: int = MAX_DOC_CHARS) -> dict:
    """处理单篇，返回统一结构（永不抛出，异常落在 error 字段）。"""
    truncated = 0
    if len(text) > max_chars:
        truncated = len(text) - max_chars
        text = text[:max_chars]
    try:
        plan = normalize(text, spec or NormalizationSpec().to_dict())
        if decisions:
            final = apply_decisions(plan, decisions)
            result = {
                "ok": True,
                "normalized": final["normalized"],
                "plan": {**plan, "normalized": final["normalized"]},
                "norm_stats": final["stats"],
            }
        else:
            result = {"ok": True, "normalized": plan["normalized"], "plan": plan}
        if truncated:
            result["warning"] = f"文本超出 {max_chars} 字，已截断 {truncated} 字"
            result["truncated"] = truncated
        return result
    except Exception as exc:  # noqa: BLE001 - 批量任务必须隔离单篇异常
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def process_batch(documents: list[str | dict],
                  spec: Optional[dict] = None,
                  decisions_map: Optional[dict] = None,
                  *,
                  max_workers: int = 4,
                  chunk_size: int = 32,
                  timeout: float = DEFAULT_TIMEOUT,
                  max_chars: int = MAX_DOC_CHARS,
                  progress: Optional[Callable[[int, int, dict], None]] = None
                  ) -> list[dict]:
    """批量处理。

    ``documents`` 元素可以是纯文本，也可以是
    ``{"id": ..., "text": ..., "name": ...}``；返回结果保持输入顺序，
    每项形如 ``{"ok", "id", "name", "normalized"/"error", ...}``。
    """
    normalized_items = []
    for i, item in enumerate(documents):
        if isinstance(item, dict):
            normalized_items.append({
                "pos": i, "id": item.get("id") or f"doc_{i}",
                "name": item.get("name", ""), "text": item.get("text", "")})
        else:
            normalized_items.append({
                "pos": i, "id": f"doc_{i}", "name": "", "text": item})

    results: list[Optional[dict]] = [None] * len(normalized_items)
    done = 0
    total = len(normalized_items)

    for start in range(0, total, chunk_size):
        chunk = normalized_items[start:start + chunk_size]
        with _NonBlockingPool(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(
                    process_one, it["text"], spec,
                    (decisions_map or {}).get(it["id"])): it
                for it in chunk
            }
            # 逐篇计时短轮询：任一 future 超过 timeout 即标记失败并取消，
            # 不能用 future.result(timeout=) 串行等待——那会让先完成的任务
            # 消耗后面任务的超时预算。
            pending = dict(future_map)
            started_at = {f: time.monotonic() for f in pending}
            while pending:
                for fut in list(pending):
                    it = pending[fut]
                    if fut.done():
                        try:
                            res = fut.result()
                        except Exception as exc:  # noqa: BLE001
                            res = {"ok": False,
                                   "error": f"{type(exc).__name__}: {exc}"}
                        del pending[fut]
                    elif time.monotonic() - started_at[fut] > timeout:
                        fut.cancel()
                        res = {"ok": False,
                               "error": f"处理超时（>{timeout}s），已跳过"}
                        del pending[fut]
                    else:
                        continue
                    res["id"] = it["id"]
                    res["name"] = it["name"]
                    res["pos"] = it["pos"]
                    results[it["pos"]] = res
                    done += 1
                    if progress:
                        progress(done, total, res)
                if pending:
                    time.sleep(0.02)
    return results  # type: ignore[return-value]
