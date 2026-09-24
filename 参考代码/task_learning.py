"""Within-match method memory. Never stores previous task answers as solutions."""
import re
import statistics


class TaskLearning:
    def __init__(self):
        self.records = {}
        self.durations = {}
        self.failures = {}

    def estimate(self, family, default=6, learned=3):
        durations = self.durations.get(family, [])
        if not durations:
            return float(default)
        # Observe real interaction cost, with a small guard against one fast outlier.
        estimate = max(float(learned), statistics.median(durations[-8:]) + 1)
        return min(20.0, estimate + min(3, self.failures.get(family, 0)))

    def record_failure(self, family):
        self.failures[family] = min(5, self.failures.get(family, 0) + 1)

    def learn(self, family, sop, answer, trace, elapsed):
        if isinstance(sop, str):
            sop = sop.strip()[:4000]
        else:
            sop = ''
        # A model may paste its current answer into its proposed SOP.
        if answer and sop and answer in sop:
            sop = sop.replace(answer, '<本题结果，必须重新计算>')
        if not sop or re.search(r'(?:答案|结果)(?:就是|为|是)\s*[:：]?', sop):
            commands = '\n'.join(str(entry.get('command', '')) for entry in trace)
            if any(term in commands for term in ('curl', 'urllib', 'requests')):
                method = '先查看当前任务给出的本地接口说明；按本题参数调用接口；从真实响应提取所需字段'
            elif commands:
                method = '先定位当前任务给出的本地资料；用有界脚本读取并计算；检查真实执行输出'
            else:
                method = '先核对当前任务原文中的已知条件；信息充分时直接求解；不要为简单任务额外探测环境'
            sop = method + '；严格遵守本题答案格式。城市、日期、文件、筛选条件和答案均须重新读取，禁止复用旧值。'
        records = self.records.setdefault(family, [])
        if sop not in records:
            records.append(sop)
            del records[:-4]
        durations = self.durations.setdefault(family, [])
        durations.append(max(1, elapsed))
        del durations[:-8]
        self.failures[family] = max(0, self.failures.get(family, 0) - 1)
        while len(self.records) > 8:
            key = next(iter(self.records))
            self.records.pop(key, None)
            self.durations.pop(key, None)
            self.failures.pop(key, None)
        return sop
