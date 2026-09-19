"""译文对齐:在「源断句段序列 × 译行文序列」上做单调动态规划。

允许的对齐组(严格单调,必须完整覆盖两侧):
  - 1:1  一个源段 ↔ 一行译文
  - 1:2  一个源段 ↔ 相邻两行译文
  - 2:1  两个相邻源段 ↔ 一行译文

每组代价:
  - 长度比例差: 两侧「去空白字符数 ÷ 各自全文去空白字符总数」的比例之差绝对值;
  - 句末标点不一致: 组末标点类别(句号 / 问号 / 叹号 / 无)两侧不同时加 0.15。

求总成本最小的完整覆盖。同分时依次优先:
  1. 1:1 组数更多;
  2. 路径更早结束(对齐类型按 1:1 → 1:2 → 2:1 的字典序,越早出现小类型越优,
     即同样进度下源侧结束得更早的路径优先)。
"""

from __future__ import annotations

# (源段跨距, 译行跨距, 是否 1:1), 顺序即同分路径的字典序优先级
_MOVES = ((1, 1, True), (1, 2, False), (2, 1, False))
_PUNCT_WEIGHT = 0.15
_EPS = 1e-9

_PERIOD = frozenset("。.")
_QUESTION = frozenset("？?")
_EXCLAIM = frozenset("！!")


def split_translation(text: str) -> list[str]:
    """译文按行拆分: 去除每行首尾空白, 丢弃空行。"""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _no_space_len(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def _punct_category(text: str) -> str | None:
    """句末标点类别: period(。.) / question(？?) / exclaim(！!) / None(无句末标点)。"""
    s = text.rstrip()
    if not s:
        return None
    ch = s[-1]
    if ch in _PERIOD:
        return "period"
    if ch in _QUESTION:
        return "question"
    if ch in _EXCLAIM:
        return "exclaim"
    return None


def _segment_texts(words: list[dict], segments: list[dict]) -> list[tuple[str, int, int]]:
    """解析并校验当前断句方案, 返回每段 (段全文, 首词下标, 末词下标)。

    与术语审校同一口径: 各段按词序连续、不重叠、完整覆盖词序列。
    """
    n_words = len(words)
    parsed: list[tuple[int, int]] = []
    for si, seg in enumerate(segments):
        rng = seg.get("word_range")
        wa, wb = int(rng[0]), int(rng[1])
        if wa < 0 or wb < 0 or wa >= n_words or wb >= n_words:
            raise ValueError(f"第 {si + 1} 段词范围 [{wa}, {wb}] 越界"
                             f"(合法范围 0–{n_words - 1})")
        if wa > wb:
            raise ValueError(f"第 {si + 1} 段词范围 [{wa}, {wb}] 非法:起点大于终点")
        parsed.append((wa, wb))
    if n_words == 0:
        if parsed:
            raise ValueError("词序列为空时不能携带字幕段")
        return []
    if not parsed or parsed[0][0] != 0:
        raise ValueError("断句方案必须从词 0 开始完整覆盖词序列")
    if parsed[-1][1] != n_words - 1:
        raise ValueError("断句方案必须完整覆盖到最后一个词")
    for (a1, b1), (a2, b2) in zip(parsed, parsed[1:]):
        if a2 != b1 + 1:
            raise ValueError(f"断句方案的字幕段不连续:词 {b1} 与词 {a2} 之间存在空隙或重叠")
    return [
        (" ".join(words[k]["text"] for k in range(wa, wb + 1)), wa, wb)
        for wa, wb in parsed
    ]


def align(words: list[dict], segments: list[dict], translation: str) -> dict:
    """对当前断句方案与按行译文执行单调 DP 对齐, 见模块文档字符串。"""
    seg_info = _segment_texts(words, segments)
    lines = split_translation(translation or "")

    m, n = len(seg_info), len(lines)
    if m == 0:
        return {"ok": False, "reason": "empty_source",
                "source_count": 0, "target_count": n,
                "message": "当前断句方案为空,无法对齐。"}
    if n == 0:
        return {"ok": False, "reason": "empty_target",
                "source_count": m, "target_count": 0,
                "message": "请按行粘贴译文。"}

    src_texts = [t for t, _wa, _wb in seg_info]
    src_lens = [_no_space_len(t) for t in src_texts]
    tgt_lens = [_no_space_len(t) for t in lines]
    total_src = sum(src_lens)
    total_tgt = sum(tgt_lens)

    def group_cost(i: int, di: int, j: int, dj: int) -> tuple[float, float, str, str]:
        """返回 (长度比例差代价, 标点代价, 组合并原文, 组合并译文)。"""
        st = " ".join(src_texts[i:i + di])
        tt = " ".join(lines[j:j + dj])
        sr = sum(src_lens[i:i + di]) / total_src if total_src else 0.0
        tr = sum(tgt_lens[j:j + dj]) / total_tgt if total_tgt else 0.0
        c_len = abs(sr - tr)
        cp, tp = _punct_category(st), _punct_category(tt)
        c_punct = 0.0 if cp == tp else _PUNCT_WEIGHT
        return c_len, c_punct, st, tt

    def better(a, b):
        """a / b = (累计代价, 累计 1:1 组数); 代价低者优, 同代价 1:1 多者优。"""
        if a is None:
            return False
        if b is None:
            return True
        if abs(a[0] - b[0]) > _EPS:
            return a[0] < b[0]
        return a[1] > b[1]

    # fwd[i][j]: 从 (0,0) 到 (i,j) 的最优 (代价, 1:1 组数), None 表示不可达
    fwd: list[list[tuple[float, int] | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    fwd[0][0] = (0.0, 0)
    for i in range(m + 1):
        for j in range(n + 1):
            cur = fwd[i][j]
            if cur is None:
                continue
            for di, dj, one in _MOVES:
                ni, nj = i + di, j + dj
                if ni > m or nj > n:
                    continue
                c_len, c_punct, _st, _tt = group_cost(i, di, j, dj)
                cand = (cur[0] + c_len + c_punct, cur[1] + (1 if one else 0))
                if better(cand, fwd[ni][nj]):
                    fwd[ni][nj] = cand

    if fwd[m][n] is None:
        # 无法完整覆盖: 前缀路径能推进到的最远行/列之后即首个未覆盖位置。
        # C 类转移(2:1)会跳过中间行状态, 故按「是否存在 i' > k 的可达状态」判定段 k 可覆盖。
        max_i = max((i for i in range(m + 1)
                     if any(fwd[i][j] is not None for j in range(n + 1))), default=0)
        max_j = max((j for j in range(n + 1)
                     if any(fwd[i][j] is not None for i in range(m + 1))), default=0)
        src_pos = max_i + 1 if max_i < m else None  # 1-based 段号
        tgt_pos = max_j + 1 if max_j < n else None  # 1-based 行号
        return {
            "ok": False,
            "reason": "uncovered",
            "source_count": m,
            "target_count": n,
            "uncovered": {"source": src_pos, "target": tgt_pos},
            "message": _uncovered_message(src_pos, tgt_pos, m, n),
        }

    # suf[i][j]: 从 (i,j) 到 (m,n) 的最优 (代价, 1:1 组数)
    suf: list[list[tuple[float, int] | None]] = [[None] * (n + 1) for _ in range(m + 1)]
    suf[m][n] = (0.0, 0)
    for i in range(m, -1, -1):
        for j in range(n, -1, -1):
            if i == m and j == n:
                continue
            best = None
            for di, dj, one in _MOVES:
                ni, nj = i + di, j + dj
                if ni > m or nj > n or suf[ni][nj] is None:
                    continue
                c_len, c_punct, _st, _tt = group_cost(i, di, j, dj)
                sc, so = suf[ni][nj]
                cand = (c_len + c_punct + sc, (1 if one else 0) + so)
                if better(cand, best):
                    best = cand
            suf[i][j] = best

    # 前向贪心回溯: 每一步按 1:1 → 1:2 → 2:1 选第一个仍能走到全局最优的转移,
    # 由此在全部最优路径中取得「较早结束」(字典序最小)的路径。
    best_cost, best_ones = fwd[m][n]
    groups = []
    i = j = 0
    acc_cost = 0.0
    acc_ones = 0
    gidx = 0
    while i < m or j < n:
        chosen = None
        for di, dj, one in _MOVES:
            ni, nj = i + di, j + dj
            if ni > m or nj > n or fwd[ni][nj] is None or suf[ni][nj] is None:
                continue
            c_len, c_punct, st, tt = group_cost(i, di, j, dj)
            fc, fo = fwd[ni][nj]
            sc, so = suf[ni][nj]
            edge = c_len + c_punct
            if (abs((acc_cost + edge) - fc) <= _EPS
                    and abs(fc + sc - best_cost) <= _EPS
                    and fo == acc_ones + (1 if one else 0)
                    and fo + so == best_ones):
                chosen = (di, dj, one, ni, nj, c_len, c_punct, st, tt)
                break
        if chosen is None:  # 理论上不会到达: fwd[m][n] 可达必有最优后继
            return {"ok": False, "reason": "uncovered",
                    "source_count": m, "target_count": n,
                    "uncovered": {"source": i + 1, "target": j + 1},
                    "message": _uncovered_message(i + 1, j + 1, m, n)}
        di, dj, one, ni, nj, c_len, c_punct, st, tt = chosen
        gidx += 1
        a, b = i, ni - 1
        c, d = j, nj - 1
        groups.append({
            "index": gidx - 1,
            "type": "1:1" if one else ("1:2" if di == 1 else "2:1"),
            "source": {
                "segments": [a + 1, b + 1],
                "word_range": [seg_info[a][1], seg_info[b][2]],
                "text": st,
            },
            "target": {
                "lines": [c + 1, d + 1],
                "text": tt,
            },
            "start": round(words[seg_info[a][1]]["start"], 3),
            "end": round(words[seg_info[b][2]]["end"], 3),
            "cost": {
                "length": round(c_len, 4),
                "punct": round(c_punct, 4),
                "total": round(c_len + c_punct, 4),
            },
        })
        acc_cost += c_len + c_punct
        acc_ones += 1 if one else 0
        i, j = ni, nj

    return {
        "ok": True,
        "total_cost": round(best_cost, 4),
        "source_count": m,
        "target_count": n,
        "groups": groups,
    }


def _uncovered_message(src_pos, tgt_pos, m: int, n: int) -> str:
    parts = []
    if src_pos is not None:
        parts.append(f"源段第 {src_pos} 段(共 {m} 段)起无法被任何对齐组覆盖")
    if tgt_pos is not None:
        parts.append(f"译文第 {tgt_pos} 行(共 {n} 行)起无法被覆盖")
    if not parts:
        return "译文行数与源段数无法在 1:1 / 1:2 / 2:1 规则下完整对齐。"
    return "无法完整覆盖:" + ";".join(parts) + "。输入已保留,请调整译文行数或断句方案。"
