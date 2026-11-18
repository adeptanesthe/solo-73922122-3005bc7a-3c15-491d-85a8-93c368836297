"""Ready-made payloads demonstrating the tricky cases required by the spec."""

SAMPLES = [
    {
        "id": "duplicate-tokens",
        "title": "重复 token（同一输入/目标多次出现，且跨微批重复）",
        "note": (
            "输入 4 号与目标 4 号在多处重复，测试 index_add 的梯度累加与"
            "共享 W 的输入/输出双重角色。"
        ),
        "payload": {
            "seed": 20240517,
            "scale": 1.0,
            "vocab_size": 256,
            "microbatches": [
                {
                    "name": "dup-A",
                    "tokens": [[4, 4], [4, 4], [7, 4], [4, 7]],
                },
                {
                    "name": "dup-B",
                    "tokens": [[4, 4], [9, 9], [9, -100], [4, 9]],
                },
            ],
        },
    },
    {
        "id": "uneven-valid",
        "title": "不等有效长度（含整批被忽略的微批）",
        "note": (
            "第一个微批仅 1 个有效目标，第二个全部 -100（应当跳过且不参与损失），"
            "第三个有 6 个有效目标；全局均值必须按“全部有效 token”加权。"
        ),
        "payload": {
            "seed": 991,
            "scale": 0.5,
            "vocab_size": 300,
            "microbatches": [
                {
                    "name": "short",
                    "tokens": [[1, 2]],
                },
                {
                    "name": "all-ignored",
                    "tokens": [[3, -100], [4, -100], [5, -100]],
                },
                {
                    "name": "long",
                    "tokens": [[6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, -100]],
                },
            ],
        },
    },
    {
        "id": "large-scores",
        "title": "大分数（scale=100，词表 4096）",
        "note": (
            "W 乘以 100 后 logits 可达数万，验证分块 logsumexp / 重算后向的"
            "数值稳定性和 float64 精度。"
        ),
        "payload": {
            "seed": 424242,
            "scale": 100.0,
            "vocab_size": 4096,
            "microbatches": [
                {
                    "name": "big-A",
                    "tokens": [[1, 4095], [2048, 0], [4095, 4095], [17, 33]],
                },
                {
                    "name": "big-B",
                    "tokens": [[255, 256], [256, 255], [1024, -100]],
                },
                {
                    "name": "big-C",
                    "tokens": [[0, 0], [4095, 1], [2047, 2048]],
                },
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Contribution-analysis samples: same training payloads plus 1..32 validation
# token pairs (input, target; -100 ignored).  These deliberately include a
# harmful micro-batch (negative Shapley contribution) and an all-ignored one.
# ---------------------------------------------------------------------------
CONTRIBUTION_SAMPLES = [
    {
        "id": "beneficial-vs-harmful",
        "title": "有益 vs 有害微批（2 个微批的精确 Shapley）",
        "note": (
            "align 微批让 4/7/9 号输入对自身目标过度自信，在含未见组合的"
            "验证集上反而抬高平均交叉熵（Shapley 贡献为负、有害）；"
            "conflict 微批把同样的输入推向别的目标，起到抑制过度自信的作用，"
            "贡献为正（有益）。两批完整训练的总改善仍为正。"
        ),
        "payload": {
            "seed": 20240517,
            "scale": 1.0,
            "vocab_size": 256,
            "microbatches": [
                {
                    "name": "align",
                    "tokens": [[4, 4], [7, 7], [9, 9], [4, 4]],
                },
                {
                    "name": "conflict",
                    "tokens": [[4, 8], [7, 8], [9, 3], [4, 3]],
                },
            ],
            "validation_tokens": [
                [4, 4], [7, 7], [9, 9], [4, 7], [7, 4],
            ],
        },
    },
    {
        "id": "ignored-batch-neutral",
        "title": "含全忽略微批（Shapley 贡献应为 0）",
        "note": (
            "ignored 微批的目标全部为 -100：任何包含它的子集都保持原始 W，"
            "因此它的精确 Shapley 贡献恰好为 0；另外两批一正一负。"
        ),
        "payload": {
            "seed": 991,
            "scale": 0.5,
            "vocab_size": 300,
            "microbatches": [
                {
                    "name": "helpful",
                    "tokens": [[1, 2], [6, 7], [8, 9], [10, 11]],
                },
                {
                    "name": "ignored",
                    "tokens": [[3, -100], [4, -100], [5, -100]],
                },
                {
                    "name": "opposing",
                    "tokens": [[1, 10], [6, 20], [8, 30], [10, 40]],
                },
            ],
            "validation_tokens": [
                [1, 2], [6, 7], [8, 9], [10, 11],
                [2, 2], [7, 7], [11, -100],
            ],
        },
    },
    {
        "id": "large-vocab-contribution",
        "title": "大词表 4 微批（4096 词表，16 种加入顺序）",
        "note": (
            "4 个微批共 16 个 token，穷举 2^4=16 个候选 W 与 4!=24 种"
            "加入顺序，验证贡献之和与完整训练总改善一致。"
        ),
        "payload": {
            "seed": 424242,
            "scale": 10.0,
            "vocab_size": 4096,
            "microbatches": [
                {
                    "name": "c-A",
                    "tokens": [[1, 4095], [2048, 0], [4095, 4095], [17, 33]],
                },
                {
                    "name": "c-B",
                    "tokens": [[255, 256], [256, 255], [1024, 1024]],
                },
                {
                    "name": "c-C",
                    "tokens": [[0, 0], [4095, 1], [2047, 2048]],
                },
                {
                    "name": "c-D",
                    "tokens": [[7, 9], [9, 7], [33, 17], [17, -100]],
                },
            ],
            "validation_tokens": [
                [1, 4095], [2048, 0], [255, 256], [0, 0],
                [4095, 4095], [7, 9], [9, 7], [33, 17],
            ],
        },
    },
]
