import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from math import comb

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tnq.evaluation.utils import is_correct  # noqa: E402

# ===================== IFEval 配置 =====================
GOOGLE_RESEARCH_PATH = "./google-research"

sys.path.insert(0, GOOGLE_RESEARCH_PATH)

DATASET_DIR = "/home/c30823/work/data"

try:
    from instruction_following_eval.evaluation import (
        InputExample,
        test_instruction_following_strict,
        test_instruction_following_loose,
    )
    IFEVAL_AVAILABLE = True
except ImportError:
    IFEVAL_AVAILABLE = False
# from instruction_following_eval.evaluation_lib import (
#     InputExample,
#     test_instruction_following_strict,
#     test_instruction_following_loose,
# )
# IFEVAL_AVAILABLE = True

# ===================== code_generation_lite 配置 =====================
TIMEOUT = 15  # 单个测试用例超时秒数
RELEASE_VERSION = "release_v6"

DEFAULT_RESULT_DIR = os.path.join(
    os.path.dirname(__file__), "eval_result"
)

precision_list = [
    # 'fp16',
    'vllm',
    'awq4bit_0',
    # "nvfp4_bs16", 
    # "nvfp4_fpquant", 
]

dataset_list = [
    'code_generation_lite', 'IFEval', 'MuSR', 'ZebraLogic', 'MATH-500'
    # 'code_generation_lite',
    # 'IFEval', 
    # 'ZebraLogic',
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从推理结果 JSON 目录中解析答案并计算正确率"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=DEFAULT_RESULT_DIR,
        help="推理结果 JSON 文件所在目录",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="逐题打印详细对比信息",
    )
    return parser.parse_args()


def pass_at_k(n, c, k):
    """无偏 pass@k 估计器 (Chen et al., 2021)"""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ===================== IFEval 评测 =====================
def work_ifeval(json_files):
    """IFEval 数据集专用评测：基于 instruction following 的 strict/loose 判定"""
    if not IFEVAL_AVAILABLE:
        print("[ERROR] IFEval 评测依赖未安装，请执行：")
        print("  pip install datasets numpy immutabledict langdetect nltk")
        print(f"  git clone https://github.com/google-research/google-research.git")
        return None

    from datasets import load_dataset

    ds = load_dataset(DATASET_DIR + "/IFEval", split="train")
    dataset_map = {i: example for i, example in enumerate(ds)}

    total_questions = 0
    total_runs = 0
    total_correct = 0
    total_correct_loose = 0
    total_no_pred = 0
    pass1_correct = 0
    majority_correct = 0

    total_inst = 0
    followed_inst_strict = 0
    followed_inst_loose = 0

    pass1_scores_strict = []

    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = data.get("idx", "?")
        results = data.get("results", [])

        if not results:
            continue
        if idx not in dataset_map:
            print(f"[WARN] index={idx} 不在 IFEval 数据集中，跳过")
            continue

        example = dataset_map[idx]
        prompt = example["prompt"]
        instruction_id_list = example["instruction_id_list"]
        # HuggingFace 数据集会把 kwargs 展开为完整字段（含大量 None），
        # 而 google-research 的 build_description() 不接受多余参数，需要过滤掉 None 值
        kwargs_list = [
            {k: v for k, v in kw.items() if v is not None}
            for kw in example["kwargs"]
        ]

        total_questions += 1
        q_correct = 0
        q_correct_loose = 0
        q_no_pred = 0
        pred_list = []

        for r in results:
            total_runs += 1
            content = r.get("content", "")

            if not content or not content.strip():
                total_no_pred += 1
                q_no_pred += 1
                pred_list.append(False)
                continue

            inp = InputExample(
                key=idx,
                instruction_id_list=instruction_id_list,
                prompt=prompt,
                kwargs=kwargs_list,
            )
            prompt_to_response = {prompt: content}

            out_strict = test_instruction_following_strict(inp, prompt_to_response)
            out_loose = test_instruction_following_loose(inp, prompt_to_response)

            total_inst += len(out_strict.follow_instruction_list)
            followed_inst_strict += sum(out_strict.follow_instruction_list)
            followed_inst_loose += sum(out_loose.follow_instruction_list)

            passed_strict = out_strict.follow_all_instructions
            passed_loose = out_loose.follow_all_instructions

            if passed_strict:
                total_correct += 1
                q_correct += 1
            if passed_loose:
                total_correct_loose += 1
                q_correct_loose += 1

            pred_list.append(passed_strict)

        if pred_list and pred_list[0]:
            pass1_correct += 1

        if sum(pred_list) > len(pred_list) / 2:
            majority_correct += 1

        n = len(pred_list)
        c_strict = sum(pred_list)
        pass1_scores_strict.append(pass_at_k(n, c_strict, 1))

    if total_runs == 0:
        print("未解析到任何推理结果。")
        return None

    print(f"  Strict prompt-level: {total_correct}/{total_runs} = {total_correct / total_runs:.4f}")
    print(f"  Loose  prompt-level: {total_correct_loose}/{total_runs} = {total_correct_loose / total_runs:.4f}")
    if total_inst > 0:
        print(f"  Strict inst-level:   {followed_inst_strict}/{total_inst} = {followed_inst_strict / total_inst:.4f}")
        print(f"  Loose  inst-level:   {followed_inst_loose}/{total_inst} = {followed_inst_loose / total_inst:.4f}")

    return total_correct, total_runs, total_questions, total_correct / total_questions if total_questions > 0 else 0.0, total_no_pred


# ===================== code_generation_lite 评测 =====================
def extract_code(text):
    """从模型输出中提取 Python 代码"""
    if not text:
        return ""
    matches = re.findall(r'```python\s*\n?(.*?)```', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r'```\s*\n?(.*?)```', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return text.strip()


def run_code(code, stdin_input="", timeout=TIMEOUT):
    """在子进程中执行 Python 代码，返回 (stdout, success)"""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, encoding='utf-8'
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "", False
    except Exception:
        return "", False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def check_solution_stdio(code, test_cases, timeout=TIMEOUT):
    """stdin/stdout 模式评测"""
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    for inp_data, expected_data in zip(inputs, outputs):
        if isinstance(inp_data, list):
            stdin_str = "\n".join(str(x) for x in inp_data)
        else:
            stdin_str = str(inp_data)
        if isinstance(expected_data, list):
            expected_str = "\n".join(str(x) for x in expected_data)
        else:
            expected_str = str(expected_data)
        stdout, success = run_code(code, stdin_str, timeout)
        if not success:
            return False
        actual_lines = stdout.strip().splitlines()
        expected_lines = expected_str.strip().splitlines()
        if len(actual_lines) != len(expected_lines):
            return False
        for a, e in zip(actual_lines, expected_lines):
            if a.strip() != e.strip():
                return False
    return True


def check_solution_fn(code, test_cases, timeout=TIMEOUT):
    """函数调用模式评测（LeetCode 风格）"""
    fn_name = test_cases["fn_name"]
    inputs = test_cases["inputs"]
    outputs = test_cases["outputs"]

    test_driver = code + "\n\n"
    test_driver += "import json, sys\n"
    test_driver += f"_fn_name = {json.dumps(fn_name)}\n"
    test_driver += f"_inputs = {json.dumps(inputs)}\n"
    test_driver += f"_outputs = {json.dumps(outputs)}\n"
    test_driver += """
_parts = _fn_name.split(".")
if len(_parts) == 2:
    _cls = globals()[_parts[0]]
    _obj = _cls()
    _fn = getattr(_obj, _parts[1])
else:
    _fn = globals()[_fn_name]

for _inp, _exp in zip(_inputs, _outputs):
    if isinstance(_inp, list):
        _result = _fn(*_inp)
    else:
        _result = _fn(_inp)
    if _result != _exp:
        try:
            if sorted(_result) == sorted(_exp):
                continue
        except (TypeError, AttributeError):
            pass
        print("FAIL")
        sys.exit(1)
print("PASS")
"""
    stdout, success = run_code(test_driver, "", timeout)
    if not success:
        return False
    return stdout.strip() == "PASS"


def check_solution(code, test_cases, timeout=TIMEOUT):
    """自动判断评测模式"""
    if "fn_name" in test_cases:
        return check_solution_fn(code, test_cases, timeout)
    else:
        return check_solution_stdio(code, test_cases, timeout)


# code_generation_lite 各版本对应的 JSONL 文件
CODEGEN_ALLOWED_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}


def load_codegen_dataset_jsonl(release_version=RELEASE_VERSION):
    """通过 JSONL 文件直接加载 code_generation_lite 数据集"""
    dataset_dir = os.path.join(DATASET_DIR, "code_generation_lite")
    allowed = CODEGEN_ALLOWED_FILES.get(release_version)
    if allowed is None:
        raise ValueError(f"Unknown release version: {release_version}, available: {list(CODEGEN_ALLOWED_FILES.keys())}")

    samples = []
    sources = []
    for fname in allowed:
        path = os.path.join(dataset_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            part = []
            for line in f:
                line = line.strip()
                if line:
                    part.append(json.loads(line))
            samples.extend(part)
            sources.append(f"{fname}({len(part)}条)")
    print(f"  JSONL 加载: {' + '.join(sources)} = {len(samples)} 题")
    return samples


def work_livecodebench(json_files, release_version=RELEASE_VERSION):
    """code_generation_lite 数据集专用评测：提取代码并执行测试用例"""
    print(f"正在加载 livecodebench/code_generation_lite ({release_version})...")
    samples = load_codegen_dataset_jsonl(release_version)
    dataset_map = {i: example for i, example in enumerate(samples)}
    print(f"数据集共 {len(samples)} 题")

    total_questions = 0
    total_runs = 0
    total_correct = 0
    total_no_pred = 0
    pass1_correct = 0
    majority_correct = 0
    pass1_scores = []
    difficulty_results = {}

    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = data.get("idx", "?")
        results = data.get("results", [])

        if not results:
            print(f"[WARN] results is empty, skip {fp}")
            continue
        if idx not in dataset_map:
            print(f"[WARN] index={idx} 不在数据集中，跳过")
            continue

        example = dataset_map[idx]

        # public_test_cases 格式: [{"input": "...", "output": "...", "testtype": "stdin"}, ...]
        # 转换为 check_solution 需要的格式: {"inputs": [...], "outputs": [...]}
        pub_raw = example.get("public_test_cases", "[]")
        if isinstance(pub_raw, str):
            pub_cases = json.loads(pub_raw)
        else:
            pub_cases = pub_raw
        test_cases = {
            "inputs": [tc["input"] for tc in pub_cases],
            "outputs": [tc["output"] for tc in pub_cases],
        }
        # 如果有 fn_name (函数调用模式), 从第一个 test case 中取
        if pub_cases and "fn_name" in pub_cases[0]:
            test_cases["fn_name"] = pub_cases[0]["fn_name"]

        difficulty = example.get("difficulty", "unknown")

        total_questions += 1
        q_correct = 0
        q_no_pred = 0
        pred_list = []

        for r in results:
            total_runs += 1
            content = r.get("content", "")

            if not content or not content.strip():
                total_no_pred += 1
                q_no_pred += 1
                pred_list.append(False)
                continue

            code = extract_code(content)
            if not code:
                total_no_pred += 1
                q_no_pred += 1
                pred_list.append(False)
                continue

            try:
                passed = check_solution(code, test_cases)
            except Exception:
                passed = False

            if passed:
                total_correct += 1
                q_correct += 1
            pred_list.append(passed)

        if pred_list and pred_list[0]:
            pass1_correct += 1

        if sum(pred_list) > len(pred_list) / 2:
            majority_correct += 1

        n = len(pred_list)
        c = sum(pred_list)
        p1 = pass_at_k(n, c, 1)
        pass1_scores.append(p1)

        if difficulty not in difficulty_results:
            difficulty_results[difficulty] = []
        difficulty_results[difficulty].append(p1)

        if total_questions % 20 == 0:
            print(f"  已评测 {total_questions} 题 ...")

    if total_runs == 0:
        print("未解析到任何推理结果。")
        return None

    print(f"  pass@1 (greedy):   {pass1_correct / total_questions:.4f}")
    print(f"  pass@1 (unbiased): {np.mean(pass1_scores):.4f}")
    print(f"  majority vote:     {majority_correct / total_questions:.4f}")
    print(f"  总正确率:           {total_correct}/{total_runs} = {total_correct / total_runs:.4f}")
    for diff in sorted(difficulty_results.keys()):
        scores = difficulty_results[diff]
        print(f"    {diff:10s}: {len(scores):4d} 题, pass@1 = {np.mean(scores):.4f}")

    return total_correct, total_runs, total_questions, float(np.mean(pass1_scores)), total_no_pred


# ===================== ZebraLogic 评测 =====================
def extract_json_table(text):
    """从模型输出中提取 JSON 表格"""
    if not text:
        return None

    # 1. 尝试提取 ```json ... ``` 代码块
    json_blocks = re.findall(r'```json\s*\n?(.*?)```', text, re.DOTALL)
    if json_blocks:
        candidate = json_blocks[-1].strip()
    else:
        # 2. 尝试提取 ``` ... ``` 代码块
        code_blocks = re.findall(r'```\s*\n?(.*?)```', text, re.DOTALL)
        if code_blocks:
            candidate = code_blocks[-1].strip()
        else:
            # 3. 使用括号匹配：从最后一个 } 往前找匹配的 {
            last_close = text.rfind('}')
            if last_close == -1:
                return None

            # 从 last_close 往前匹配括号
            depth = 1  # 已经有一个 }
            pos = last_close - 1
            while pos >= 0 and depth > 0:
                if text[pos] == '}':
                    depth += 1
                elif text[pos] == '{':
                    depth -= 1
                pos -= 1

            if depth == 0:
                # 找到匹配的 {，位置在 pos+1
                candidate = text[pos+1:last_close+1]
            else:
                return None

    # 尝试解析 JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def normalize_table(table_json):
    """
    归一化表格为统一格式: {"header": [...], "rows": [[...], ...]}
    支持多种输入格式:
    - {"header": [...], "rows": [...]}  (标准格式)
    - {"1": {...}, "2": {...}}  (house number as key)
    - {"house1": {...}, "house2": {...}}  (house prefix)
    - {"houses": [{"number": 1, ...}, {"number": 2, ...}]}  (数组格式)
    - {"solution": [{"house": 1, ...}, ...]}  (solution 包装的数组)
    - [{"house": 1, ...}, ...]  (直接数组格式)
    """
    # 处理直接数组格式
    if isinstance(table_json, list):
        array_data = table_json
    elif not table_json or not isinstance(table_json, dict):
        return None
    # 格式1: 已经是标准格式
    elif "header" in table_json and "rows" in table_json:
        return table_json
    # 格式4/5: {"houses": [...]} 或 {"solution": [...]} 数组格式
    elif ("houses" in table_json and isinstance(table_json["houses"], list)) or \
         ("solution" in table_json and isinstance(table_json["solution"], list)):
        array_data = table_json.get("houses") or table_json.get("solution")
    else:
        array_data = None

    # 处理数组格式数据
    if array_data is not None:
        houses = {}
        for idx, house_obj in enumerate(array_data):
            if not isinstance(house_obj, dict):
                continue
            # 提取 house number (支持多种字段名)
            house_num = (house_obj.get("number") or
                        house_obj.get("house_number") or
                        house_obj.get("house") or
                        house_obj.get("id"))

            # 如果没有找到 house number，使用数组索引 + 1
            if house_num is None:
                house_num = idx + 1

            # 复制对象，移除 house number 相关字段（因为它会成为 House 列）
            house_data = {k: v for k, v in house_obj.items()
                         if k not in ("number", "house_number", "house", "id")}
            houses[int(house_num)] = house_data

        if houses:
            # 跳转到统一的构建逻辑
            pass
        else:
            return None
    else:
        # 格式2/3: {"1": {...}, "2": {...}} 或 {"house1": {...}, "house_1": {...}, ...}
        # 提取所有 house 数据
        houses = {}
        for key, value in table_json.items():
            if not isinstance(value, dict):
                continue
            # 提取 house number
            house_num = None
            if key.isdigit():
                house_num = int(key)
            elif key.startswith("house"):
                # 支持 house1, house_1 等格式
                suffix = key[5:]  # 去掉 "house" 前缀
                if suffix.startswith("_"):
                    suffix = suffix[1:]  # 去掉下划线
                if suffix.isdigit():
                    house_num = int(suffix)

            if house_num is not None:
                houses[house_num] = value

        if not houses:
            return None

    # 构建标准格式
    # 从第一个 house 推断 header
    first_house = houses[min(houses.keys())]
    # 归一化 key 名称
    key_mapping = {
        "name": "Name",
        "nationality": "Nationality",
        "book": "BookGenre", "book_genre": "BookGenre", "bookgenre": "BookGenre",
        "lunch": "Food", "food": "Food",
        "color": "Color",
        "animal": "Animal",
        "occupation": "Occupation",
        "phonemodel": "PhoneModel", "phone_model": "PhoneModel", "phone": "PhoneModel",
        "child": "Children", "children": "Children",
        "music": "MusicGenre", "musicgenre": "MusicGenre", "music_genre": "MusicGenre",
        "height": "Height",
        "pet": "Pet",
        "hair": "HairColor", "hair_color": "HairColor", "haircolor": "HairColor",
        "mother": "Mother", "mother_name": "Mother",
        "birthday": "Birthday", "birthday_month": "Birthday",
        "sport": "FavoriteSport", "favorite_sport": "FavoriteSport", "favoritesport": "FavoriteSport",
        "preference": "Vacation", "vacation": "Vacation",
        "cigar": "Cigar",
        "education": "Education",
        "flower": "Flower",
        "car": "CarModel", "car_model": "CarModel", "carmodel": "CarModel",
        "house_style": "HouseStyle", "housestyle": "HouseStyle",
    }

    # 构建 header (House + 其他属性)
    attrs = set()
    for house_data in houses.values():
        for k in house_data.keys():
            normalized_k = key_mapping.get(k.lower(), k)
            attrs.add(normalized_k)

    header = ["House"] + sorted(attrs)

    # 构建 rows
    rows = []
    for house_num in sorted(houses.keys()):
        house_data = houses[house_num]
        row = [str(house_num)]
        for attr in header[1:]:
            # 反向查找原始 key
            value = None
            for orig_k, v in house_data.items():
                if key_mapping.get(orig_k.lower(), orig_k) == attr:
                    value = v
                    break
            row.append(str(value) if value is not None else "")
        rows.append(row)

    return {"header": header, "rows": rows}


def compare_tables(pred_table, gold_table):
    """
    比较两个表格。

    返回:
      - puzzle_correct: 维持现有口径, 仅基于已对齐列判断整表是否全对
      - cell_correct:   对齐后且值正确的 cell 数
      - cell_total:     现有 cell-level accuracy 使用的分母, 即已对齐 cell 数
      - gold_total:     gold 表中的总 cell 数, 用于 recall
      - pred_total:     pred 表中的总 cell 数, 用于 precision
    """
    if pred_table is None or gold_table is None:
        return False, 0, 0, 0, 0

    pred_header = pred_table.get("header", [])
    pred_rows = pred_table.get("rows", [])
    gold_header = gold_table.get("header", [])
    gold_rows = gold_table.get("rows", [])
    gold_total = len(gold_rows) * len(gold_header)
    pred_total = len(pred_rows) * len(pred_header)

    if len(pred_rows) != len(gold_rows):
        return False, 0, gold_total, gold_total, pred_total

    # 建立 header 映射 (pred -> gold)
    header_map = {}
    for i, ph in enumerate(pred_header):
        for j, gh in enumerate(gold_header):
            if ph.lower() == gh.lower():
                header_map[i] = j
                break

    # 逐 cell 比较
    cell_correct = 0
    cell_total = 0
    all_match = True

    for pred_row, gold_row in zip(pred_rows, gold_rows):
        for pred_idx, pred_val in enumerate(pred_row):
            gold_idx = header_map.get(pred_idx)
            if gold_idx is None:
                continue

            cell_total += 1
            # 归一化比较 (小写 + 去空格)
            pred_norm = str(pred_val).lower().strip()
            gold_norm = str(gold_row[gold_idx]).lower().strip()

            if pred_norm == gold_norm:
                cell_correct += 1
            else:
                all_match = False

    return all_match, cell_correct, cell_total, gold_total, pred_total


def work_zebralogic(json_files):
    """ZebraLogic 数据集专用评测：提取 JSON 表格并逐格比较"""
    total_questions = 0
    total_runs = 0
    total_correct = 0  # puzzle-level: 整张表全对
    total_no_pred = 0
    pass1_correct = 0
    majority_correct = 0

    # cell-level 统计
    total_cells = 0
    correct_cells = 0
    total_gold_cells = 0
    total_pred_cells = 0

    pass1_scores = []

    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = data.get("idx", "?")
        results = data.get("results", [])
        gold_answer_raw = data.get("gold_answer")

        if not results or not gold_answer_raw:
            continue

        # 解析 gold_answer (可能是 JSON 字符串)
        if isinstance(gold_answer_raw, str):
            # 替换单引号为双引号 (Python dict 格式)
            gold_answer_raw = gold_answer_raw.replace("'", '"')
            try:
                gold_table = json.loads(gold_answer_raw)
            except json.JSONDecodeError:
                print(f"[WARN] idx={idx} gold_answer 解析失败")
                continue
        else:
            gold_table = gold_answer_raw

        total_questions += 1
        q_correct = 0
        q_no_pred = 0
        pred_list = []

        for ri, r in enumerate(results):
            total_runs += 1
            content = r.get("content", "")

            if not content or not content.strip():
                total_no_pred += 1
                q_no_pred += 1
                pred_list.append(False)
                continue

            # 提取并归一化表格
            pred_json = extract_json_table(content)
            if pred_json is not None and 'solution' in pred_json:
                pred_json = pred_json['solution']
            pred_table = normalize_table(pred_json)

            if pred_table is None:
                total_no_pred += 1
                q_no_pred += 1
                pred_list.append(False)
                # print(f"[WARN] pred_table is None, skip {fp} {idx} {ri}")
                # print(f"pred_json {pred_json}")
                # print(f"gold_table {gold_table}")
                continue

            # 比较表格
            puzzle_match, cell_match, cell_count, gold_cell_count, pred_cell_count = compare_tables(pred_table, gold_table)

            if puzzle_match:
                total_correct += 1
                q_correct += 1

            correct_cells += cell_match
            total_cells += cell_count
            total_gold_cells += gold_cell_count
            total_pred_cells += pred_cell_count
            pred_list.append(puzzle_match)

        # pass@1 (greedy)
        if pred_list and pred_list[0]:
            pass1_correct += 1

        # majority vote
        if sum(pred_list) > len(pred_list) / 2:
            majority_correct += 1

        # 无偏 pass@k
        n = len(pred_list)
        c = sum(pred_list)
        pass1_scores.append(pass_at_k(n, c, 1))

    if total_runs == 0:
        print("未解析到任何推理结果。")
        return None

    cell_accuracy = correct_cells / total_cells if total_cells else 0.0
    cell_precision = correct_cells / total_pred_cells if total_pred_cells else 0.0
    cell_recall = correct_cells / total_gold_cells if total_gold_cells else 0.0
    if cell_precision + cell_recall == 0:
        cell_f1 = 0.0
    else:
        cell_f1 = 2 * cell_precision * cell_recall / (cell_precision + cell_recall)

    print(f"  Puzzle-level accuracy: {total_correct}/{total_runs} = {total_correct / total_runs:.4f}")
    print(f"  Cell-level accuracy:   {correct_cells}/{total_cells} = {cell_accuracy:.4f}")
    print(f"  Cell precision:        {correct_cells}/{total_pred_cells} = {cell_precision:.4f}")
    print(f"  Cell recall:           {correct_cells}/{total_gold_cells} = {cell_recall:.4f}")
    print(f"  Cell F1:               {cell_f1:.4f}")
    print(f"  pass@1 (greedy):       {pass1_correct / total_questions:.4f}")
    print(f"  pass@1 (unbiased):     {np.mean(pass1_scores):.4f}")
    print(f"  majority vote:         {majority_correct / total_questions:.4f}")
    print(f"  未抽取表格:             {total_no_pred}")

    return total_correct, total_runs, total_questions, float(np.mean(pass1_scores)), total_no_pred


# ===================== 通用评测（MuSR / MATH-500 等） =====================
def work(json_files):
    total_questions = 0
    total_runs = 0
    total_correct = 0
    total_no_pred = 0
    pass1_correct = 0
    majority_correct = 0

    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = data.get("index", "?")
        gold = data.get("gold_answer")
        results = data.get("results", [])

        if not results:
            continue

        total_questions += 1
        q_correct = 0
        q_no_pred = 0
        pred_list = []

        for r in results:
            total_runs += 1
            pred = r.get("extracted_answer")

            if pred is None:
                matched = False
                q_no_pred += 1
            else:
                matched = is_correct(str(pred), str(gold)) if gold is not None else False
                pred_list.append(str(pred))

            if matched:
                q_correct += 1

            # if args.verbose:
            #     repeat_idx = r.get("repeat_index", "?")
            #     status = "✓" if matched else "✗"
            #     print(f"  [{status}] #{idx} repeat={repeat_idx}  pred={pred!s:<20s}  gold={gold}")

        total_correct += q_correct
        total_no_pred += q_no_pred

        # pass@1: 至少一次正确
        if q_correct > 0:
            pass1_correct += 1

        # majority vote: 出现次数最多的答案是否正确
        if pred_list and gold is not None:
            counter = Counter(pred_list)
            most_common_answer = counter.most_common(1)[0][0]
            if is_correct(most_common_answer, str(gold)):
                majority_correct += 1

        # if args.verbose and len(results) > 1:
        #     print(f"  -> 问题 {idx}: {q_correct}/{len(results)} 正确")

    if total_runs == 0:
        print("未解析到任何推理结果。")
        return

    accuracy = total_correct / total_runs
    q_pass_rate = sum(
        1 for fp in json_files
        if (lambda d: any(
            is_correct(str(r.get("extracted_answer", "")), str(d.get("gold_answer", "")))
            for r in d.get("results", [])
            if r.get("extracted_answer") is not None and d.get("gold_answer") is not None
        ))(json.load(open(fp, "r", encoding="utf-8")))
    ) / total_questions if total_questions > 0 else 0.0

    # print()
    # print("=" * 60)
    # # print(f"结果目录: {args.result_dir}")
    # print(f"问题数:      {total_questions}")
    # print(f"总推理次数:  {total_runs}")
    # print(f"正确次数:    {total_correct}")
    # print(f"未抽取答案:  {total_no_pred}")
    # print(f"整体准确率:  {accuracy:.2%}  ({total_correct}/{total_runs})")
    # if total_runs > total_questions:
    #     print(f"pass@1 (至少一次正确的问题比例): {q_pass_rate:.2%}")
    # print("=" * 60)

    return total_correct, total_runs, total_questions, q_pass_rate, total_no_pred


def main():
    args = parse_args()

    for dataset in dataset_list:
        for precision in precision_list:
            json_files = sorted(glob.glob(os.path.join(args.result_dir, precision, dataset, "*.json")))
            if not json_files:
                print(f"在 {args.result_dir}/{precision}/{dataset} 下未找到任何 JSON 文件。")
                continue

            print(f"\n[{dataset}] [{precision}] 共 {len(json_files)} 个文件")

            if dataset == "IFEval":
                result = work_ifeval(json_files)
            elif dataset == "code_generation_lite":
                result = work_livecodebench(json_files)
            elif dataset == "ZebraLogic":
                result = work_zebralogic(json_files)
            else:
                result = work(json_files)

            if result is None:
                print(f"[{dataset}] [{precision}] 评测失败或无结果")
            else:
                total_correct, total_runs, total_questions, q_pass_rate, total_no_pred = result
                print(f"[{dataset}] [{precision}] 正确率: {total_correct}/{total_runs}  {total_correct/total_runs:.2%}")
                print(f"[{dataset}] [{precision}] pass@1: {q_pass_rate}")
                print(f"[{dataset}] [{precision}] 未抽取答案: {total_no_pred}")
            print("=" * 60)

def cmp_two_results(result_dir1, result_dir2):
    json_files1 = sorted(glob.glob(os.path.join(result_dir1, "*.json")))
    json_files2 = sorted(glob.glob(os.path.join(result_dir2, "*.json")))
    if not json_files1 or not json_files2:
        print(f"在 {result_dir1} 和 {result_dir2} 下未找到任何 JSON 文件。")
        return
    for fp1, fp2 in zip(json_files1, json_files2):
        with open(fp1, "r", encoding="utf-8") as f1, open(fp2, "r", encoding="utf-8") as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)

            correct1 = data1.get("gold_answer") == data1.get("results")[0].get("extracted_answer")
            correct2 = data2.get("gold_answer") == data2.get("results")[0].get("extracted_answer")
            
            if data1.get('index') == 2:
                continue
                
            if correct1 != correct2:
                print(f"问题 {data1.get('index')} 在 {result_dir1} 和 {result_dir2} 中的正确答案不同。")
                print(f"gold: {data1.get('gold_answer')}")
                print(f"pred1: {data1.get('results')[0].get('extracted_answer')}")
                print(f"pred2: {data2.get('results')[0].get('extracted_answer')}")

                thinking1 = data1.get("results")[0].get("thinking_content")
                thinking2 = data2.get("results")[0].get("thinking_content")
                content1 = data1.get("results")[0].get("content")
                content2 = data2.get("results")[0].get("content")
                print("===================thinking1===================")
                print(thinking1)
                print("===================thinking2===================")
                print(thinking2)
                print("===================content1===================")
                print(content1)
                print("===================content2===================")
                print(content2)

                return 
            

if __name__ == "__main__":
    main()
    # cmp_two_results(
    #     "eval_results/claude_math_nvfp4_bs16_repeat1_0224",
    #     "eval_results/claude_math_fp16_repeat1_0224"
    # )




"""
In an adrenaline inducing bungee jumping site, Mack's thrill-seeking adventure came to a gruesome end by a nunchaku; now, it's up to Detective Winston to unravel the deadly secrets between Mackenzie and Ana.\n\n

Winston took a gulp of his black coffee, staring at the notes sprawled across his desk. A murder case at a bungee jumping site was definitely out of the ordinary. Today's victim was a young man named Mack, loud mouthed and cocky by all accounts. \n\n

Mack was bungee jumping the day he was killed. Oddly enough, according to the records, no one else was documented at the bungee jumping site that day, making this case even more peculiar. The first stop for the day was to visit one of Mack's housemates, a woman named Ana. They were seen leaving in the same vehicle from their shared housing complex the morning of the murder, and it was time for Winston to dig deeper. \n\n

As he pulled into the shared housing driveway, a nondescript car came into sight. He learned from neighbours that it was frequently used by multiple residents, but Ana had a peculiar interest in it. She would insist on driving whenever with a group of friends, later meticulously cleaning the car after each use. An idiosyncrasy of hers maybe, but a part of the puzzle nonetheless.\n\nWinston knocked on the door, Ana opened it warily, twiddling a cleaning cloth and spray in her hands and greeted him with a nervous nod. Ana gets nervous and fidgets with the cleaner and cloth when questioned. Winston could sense palpable unease as he started asking her questions.\n\n\"Ana, did you not join Mack and the others for bungee jumping today?\" Winston questioned, to which she responded, \"I signed up to jump. But I didn't end up going through with it.\"\n\n\"

Any particular reason you didn't join the others, Ana?\" Winston proceeded. \n\nAna took a deep breath, \"Well sir, my faith doesn't really permit bungee jumping. Truth be told, I was persuaded strongly by Mack. I had even signed up out of peer pressure but couldn't push myself.\"\n\nIt was true – Mack was insisting that everyone in the group should bungee jump. Mack had reportedly also been vocal about ridiculing Ana’s faith, even encouraging others to join him in doing so. It was a significant factor in their relationship.\n\n\"Ana, did you and Mack leave in the same car for the bungee jumping event this morning?\" Winston gently pushed further.\n\n\"Yes. Yes, we did. We always carpool.\" She responded while anxiously using the cleaner and cloth on her car’s dashboard. Her eyes flickered nervously back to Winston, expecting the next question.\n\nWinston took a deep breath, standing up to leave, \"Alright Ana, that should cover everything for now. We'll be in touch.\"\n\nAna nervously nodded without looking up from her cleaning, wringing the cloth repeatedly as Winston walked away, left again with another piece to the enigmatic puzzle of Mack's murder.\n\nThe day was getting older and Winston was getting more tired, but the case was fresh, and he wasn't one to back down. He tugged on his coat as he approached the bashful teen waiting for him by the police station.\n\n\"Mackenzie, it is?\" he asked, extending his hand.\n\n\"Yeah, that's right.\" The slight lisp, overlaid with blanket anxiety, confirmed what the school reports suggested.\n\n\"You were at the site when Mack... erm... you know,\" Winston's voice was methodical, calm -- almost robotic. The suspicion on Mackenzie was not unfounded - the security cameras showed him buying nunchaku a week before. \n\nMackenzie shifted on his feet, looking away before answering, \"Yeah, I was there.\"\n\nWinston pulled out a small notebook, \"What were you doing there, Mackenzie?”\n\n“Bungee jumping, like Mack… Then I left. I didn't... I didn't do anything…” Mackenzie replied.\n\nInternally, Winston sighed at the never-ending waterfall of teenage angst this case was turning into. \n\n“Martial arts, huh?” Winston segued, gesturing to a bruise on Mackenzie’s knuckles. “Nunchaku particularly, I see? Training does include the use of those, correct?” \n\nThe change in Mackenzie’s demeanor mirrored the bitterness in the last month’s weather – dark eyes replaced with ice-cold ones. “Yeah,” he admitted, shrinking slightly.\n\nMackenzie always took pride in being the best at everything. So when Mack got everything he wanted - the promotion to team captain, the respect, the attention - it was a hard pill for Mackenzie to swallow. Winston remembered the team talk, Mackenzie was indeed the top candidate but it had gone to Mack instead. \n\nWhat clinched it was Mackenzie’s remarks about Mack, echoing whispers of dispute and bickering, lost in the crowded lunchroom. There were also multiple witness reports of the two seen arguing at the bungee jumping site previously. Mackenzie had indeed said disparaging, almost emotional things about Mack – all stemming from a potent brew of jealousy, Winston inferred.\n\nShifting later through the detritus of Mackenzie's life, Winston discovered the nunchaku that matched the forensics report. They were tucked away, but the layer of dust suggested they weren't a favored possession anymore. It wasn’t hidden, it was misplaced – discarded in the throes of developing maturity.\n\nAs the sun started to set, Winston could see witnesses, scattered across the park, repeatedly pointing to the bungee jumping scaffolding. It occurred to him, then, the narrative of the past days. Mackenzie, jealous and wronged, over and over, at the same sight. It was quite a sight. \n\nWinston, shuffling back to the station, was left with one thought - Looks like Mackenzie had quite an eventful week.\n\nWho is the most likely murderer?",
  

"""
