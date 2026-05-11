"""验证 MMR 打散逻辑和流水线模块导入"""
import sys
sys.path.insert(0, "src")
from ecom_rec.pipeline.rerank import mmr_rerank

candidates = ["i1","i2","i3","i4","i5"]
scores = {"i1":0.9,"i2":0.85,"i3":0.8,"i4":0.75,"i5":0.7}
cats = {"i1":"skincare","i2":"skincare","i3":"makeup","i4":"makeup","i5":"haircare"}

result = mmr_rerank(candidates, scores, cats, k=3, lambda_=0.5)
assert len(result) == 3
assert result[0] == "i1"  # 最高分首选
# lambda=0.5 时，i3(makeup)应优于i2(skincare)，因为 i3 类目更多样
assert "i3" in result or "i4" in result  # 有 makeup 类目入选
print(f"MMR 结果：{result}")

# 验证 pipeline __init__ 导入
from ecom_rec.pipeline import MultiRecall, mmr_rerank, Recommender
print("流水线模块导入成功")
print("STATUS: DONE")
