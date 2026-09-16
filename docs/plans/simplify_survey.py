"""Generate the short questionnaire without modifying the frozen model."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE / "shanghai_survey_v2"
NEW = HERE / "shanghai_survey_v3"
NEW.mkdir(exist_ok=True)
VERSION = "shanghai-survey-v3.1"
main = HERE / "Shanghai_Travel_Intention_Survey.md"
backup = OLD / "Shanghai_Travel_Intention_Survey_v2_full.md"
if not backup.exists():
    backup.write_bytes(main.read_bytes())

def save(name, data):
    (NEW/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")

for name in ("cards.json","forms.json"):
    data = json.loads((OLD/name).read_text(encoding="utf-8"))
    data["version"] = VERSION
    save(name,data)
cards = json.loads((NEW/"cards.json").read_text(encoding="utf-8"))["cards"]
mapping = json.loads((OLD/"field_mapping.json").read_text(encoding="utf-8"))
mapping["version"] = VERSION
mapping["other_question_codes"] = {"S01":["yes","no"],"S02":["yes","no"]}
mapping["persona_fields"]["household_size"] = {"question":"P04","map":{str(i):i for i in range(1,11)},"unsupported":["11_plus"]}
mapping["availability"] = {"car":"P06 == yes AND P07 == yes","bike":"P09 == yes","pt":True,"walk":True}
for key in ("unable_reason_codes","shift_statuses","shift_sign","shift_directions"):
    mapping.pop(key,None)
mapping["shift_measurement"] = "Not collected. Evaluate mode choice only; frozen departure output is not validated by this survey."
mapping["choice_statuses"] = ["selected","unable","skipped"]
mapping["raw_encoding"] = "Use symbolic codes in question_options.json. Blank means skipped. All questions single choice."
save("field_mapping.json",mapping)

questions = [
("S01","您是否已满18周岁，并愿意参加本次匿名调查？",[("yes","是"),("no","否")]),
("S02","过去30天，您是否在上海出行过？",[("yes","是"),("no","否")]),
("P01","您的年龄是？",[(x,x+"岁" if x!="65+" else "65岁及以上") for x in ("18-24","25-34","35-44","45-64","65+")]),
("P02","您目前主要从事什么工作？",[("student","学生"),("office_worker","办公室工作"),("service_worker","服务业工作"),("manual_worker","生产、施工或体力劳动"),("retired","退休"),("unemployed","待业"),("homemaker","料理家务"),("other","其他")]),
("P03","您的个人月收入大约是多少？",[("below_5000","5000元以下"),("from_5000_to_9999","5000—9999元"),("from_10000_to_19999","10000—19999元"),("at_least_20000","20000元及以上"),("refused","不方便回答")]),
("P04","您家通常一起生活的有几人？（包括自己）",[(str(i),str(i)+"人") for i in range(1,11)]+[("11_plus","11人及以上")]),
("P05","您家是否有未满18岁的孩子？",[("yes","有"),("no","没有")]),
("P06","您家是否有可供您使用的小汽车？",[("yes","有"),("no","没有")]),
("P07","您是否持有有效的小汽车驾驶证？",[("yes","有"),("no","没有")]),
("P09","您是否有自己的普通自行车？（不含电动车和共享单车）",[("yes","有"),("no","没有")]),
("P11","您是否持有目前有效的公交或地铁定期乘车套餐？（不含普通交通卡、乘车码）",[("yes","有"),("no","没有")]),
("P12","您平时最常用哪种方式出行？",[("car_driver","自己开车"),("car_passenger","乘坐家人或朋友的车"),("public_transport","公交或地铁"),("pedal_bike","普通自行车"),("ridehail","出租车或网约车"),("ebike","电动自行车"),("walk","步行"),("mixed","多种方式差不多"),("other","其他")]),
("P13","您平时的出发时间一般能调整多少？",[("up_to_5","基本固定，最多调整5分钟"),("from_6_to_30","可以调整6—30分钟"),("over_30","可以调整30分钟以上"),("no_stable_schedule","没有固定出发时间")]),
("P14","您的身体状况会影响步行或骑自行车吗？",[("none","基本不影响"),("mild","有些影响"),("significant","影响较大")]),
]
save("question_options.json",{
    "version":VERSION,
    "questions":[{"id":q,"text":t,"options":[{"code":c,"label":l} for c,l in opts]} for q,t,opts in questions],
    "mode_options":[{"code":c,"label":l} for c,l in [("car","自己开车"),("pt","公交或地铁"),("bike","普通自行车"),("walk","步行"),("unable","都不适合")]]
})
parts = ["# 上海出行选择问卷\n","本问卷用于了解不同出行条件下的交通选择，匿名填写，仅用于学术研究。所有题目均为单选，不想回答的题目可以跳过。\n","## 一、基本情况\n"]
for index,(qid,title,opts) in enumerate(questions,1):
    parts.append(f"**{index}. {title}**\n\n<!-- question_id={qid} -->\n\n"+"　".join("○ "+label for _,label in opts)+"\n")
    if qid in ("S01","S02"):
        parts.append("选择“否”，问卷结束。\n")
parts += ["## 二、出行选择\n",
"请设想：您要从家里去 **6公里外的商店取一件已买好的商品**，原计划 **早上8:00出发，9:00前到达**。下面是同一次出行遇到的10种情况，每种情况单独选择。\n",
"表中是从家到商店的**全程时间和总费用**，已包含步行、等车、换乘和停车。您只需选出自己最可能采取的做法，不用计算。\n",
"如果您家有可用的小汽车且您有驾照，就可以选择自己开车；如果您有自己的普通自行车，就可以选择骑车。\n"]
notes = {"B0":"晴天，交通正常。","W1":"下雨，出行时间和费用如下。","D1":"晴天，公交或地铁发生延误，等车比平时多15分钟。","WD1":"下雨，公交或地铁发生延误，等车比平时多15分钟。","F1":"晴天，公交或地铁票价涨到6元。","P1":"晴天，停车费涨到30元，开车总费用为35元。","R1":"晴天，道路临时受阻，开车比平时多花20分钟。","A_WALK":"晴天，公交或地铁站点较远，进站和出站各需步行12分钟。","A_WAIT":"晴天，公交或地铁班次较少，首次等车需要15分钟。","A_TRANSFER":"晴天，公交或地铁需要换乘1次，换乘步行和等车共需10分钟。"}
labels = {"car":"自己开车","pt":"公交或地铁","bike":"普通自行车","walk":"步行"}
for i,card in enumerate(cards,1):
    cid = card["card_id"]
    pt = card["alternatives"][1]
    parts.append(f"## 情景卡 {i:02d}\n\n<!-- card_id={cid}，后台按分配顺序展示 -->\n\n**{notes[cid]}**\n")
    parts.append("| 出行方式 | 全程时间（分钟） | 总费用（元） |\n|---|---:|---:|")
    for alt in card["alternatives"]:
        parts.append(f"| {labels[alt['mode']]} | {alt['travel_time_min']} | {alt['monetary_cost']} |")
    transfer = "无需换乘" if pt["transfers"] == 0 else f"换乘{pt['transfers']}次（{pt['transfer_time_min']}分钟）"
    parts.append(f"\n公交或地铁：进出站步行共{pt['access_time_min']+pt['egress_time_min']}分钟，首次等车{pt['wait_time_min']}分钟，{transfer}。以上均已计入全程时间。\n")
    parts.append("**您会选择哪种方式？**\n\n○ 自己开车　○ 公交或地铁　○ 普通自行车　○ 步行　○ 都不适合\n")
parts.append("---\n\n问卷结束，感谢您的参与！\n")
main.write_text("\n".join(parts),encoding="utf-8")
adapter = (OLD/"survey_adapter.py").read_text(encoding="utf-8")
adapter = adapter.replace('("P06", "P07", "P08")','("P06", "P07")').replace('("P09", "P10")','("P09",)')
adapter = adapter.replace('    if str(raw.get("C00", "")).strip() != "confirmed":\n        errors.append("offered_modes:not_confirmed")\n','')
adapter = adapter.replace("outputs/shanghai_survey_v2","outputs/shanghai_survey_v3")
(NEW/"survey_adapter.py").write_text(adapter,encoding="utf-8")
(NEW/"protected_files_before.json").write_bytes((OLD/"protected_files_before.json").read_bytes())
def header(name, fields):
    with (NEW/name).open("w",encoding="utf-8-sig",newline="") as f:
        csv.writer(f).writerow(fields)
header("respondents_template.csv",["respondent_id","survey_version","form_id","recruitment_channel","completed_at"]+[q[0] for q in questions])
header("sp_responses_template.csv",["respondent_id","survey_version","form_id","card_id","task_order","car_available","bike_available","choice_status","chosen_mode"])
(NEW/"IMPLEMENTATION.md").write_text("""# 简版实施说明（仅研究者阅读）

当前问卷为上一级 Shanghai_Travel_Intention_Survey.md，版本 shanghai-survey-v3.1。
14道基本情况单选，10张情景卡各1道交通方式单选，共24道单选。不设置理解测试、确认页、原因追问、详细出行日记或出发时间选择题。耗时通过试填实测。

## 平台设置

- 只展示问卷正文，不展示变量代码及研究者说明。
- S01或S02选“否”结束；其余题可跳过，不弹补答确认。
- 每人随机且均衡分配 forms.json 中一个顺序，展示全部10张卡。记录实际顺序，卡片编号按展示顺序生成。
- P06、P07均为“有”时显示开车选项及表格行；P09为“有”时显示自行车选项及表格行。公交或地铁、步行始终显示。纸质版保留正文的选项适用说明。
- 家庭人数可用单选下拉框。每张卡选完交通方式后直接下一卡，不追问原因。
- 按 question_options.json 导出代码；未答留空；“都不适合”记 choice_status=unable、chosen_mode留空，正常方式记selected。
- 删除旧版C00/C01/C02、P08/P10、S03、R系列、E系列和精确分钟字段，旧数据不能直接标记为v3。

## 冻结模型对应

模型、词表、归一化及原始推理代码保持冻结。10张卡数值和顺序方案不变，数据以 cards.json 为准。survey_adapter.py 仅转换问卷数据，不调用参考流水线重新估算时间费用。

交通套餐不包括普通交通卡；家庭车辆、自有普通自行车按题目定义编码。网约车、电动车、搭乘他人汽车等不受支持的习惯方式不强行归类。缺失必要输入和家庭人数11人及以上保留原始记录，单列覆盖率，不伪造模型输入。

本版不采集情景下的出发时间调整，只评估交通方式选择，不验证冻结模型的出发时间输出。“都不适合”单列比例。情景中的计划出发和到达时间仍作为固定模型输入保留；基本情况P13为模型需要的日常时间灵活性输入，不是情景出发时间结果题。

本版用于假设情景选择比较，不包含详细实际出行验证。旧版存放于 shanghai_survey_v2；使用本目录的编码及空表头。

## 重建与验证

项目根目录运行 .venv/Scripts/python.exe -B docs/plans/simplify_survey.py 重建。
运行本目录 validate_package.py 检查显示数值、冻结输入接口及受保护文件哈希；结果写入 validation_report.json。
""",encoding="utf-8")
qa = (OLD/"validate_package.py").read_text(encoding="utf-8")
for old,new in [("自驾小汽车 / Driving","自己开车"),("公共交通 / Public transport","公交或地铁"),("普通人力自行车 / Conventional pedal bicycle","普通自行车"),("步行 / Walking","步行")]:
    qa = qa.replace(old,new)
start = qa.index('        formula = ')
end = qa.index('    checks.append("ten_card',start)
qa = qa[:start]+'''        transfer = "无需换乘" if pt["transfers"] == 0 else f"换乘{pt['transfers']}次（{pt['transfer_time_min']}分钟）"
        formula = f"进出站步行共{pt['access_time_min']+pt['egress_time_min']}分钟，首次等车{pt['wait_time_min']}分钟，{transfer}。"
        checked(formula in fragment, f"{cid}: displayed PT components differ")
'''+qa[end:]
qa = qa.replace(', C00="confirmed"','').replace(', P08="yes",',',').replace(', P10="yes",',',')
qa = qa.replace(', P08="yes" if car else "not_applicable"','').replace(', P10="yes" if bike else "not_applicable"','').replace(', P08="not_applicable"','')
qa = qa.replace('raw = dict(raw_base, P08="unknown", P10="unknown")','raw = dict(raw_base, P06="no", P09="no")')
qa = qa.replace('    checks.append("ten_card_content','''    checked("C00" not in text and "确认理解" not in text, "Old confirmation remains")
    checked("**您的出发时间会怎么安排？**" not in text, "Removed departure question remains")
    checked(text.count("**您会选择哪种方式？**") == 10, "Mode question count")
    checked("shift_directions" not in MAPPING, "Removed departure response schema remains")
    checks.append("ten_card_content''')
(NEW/"validate_package.py").write_text(qa,encoding="utf-8")
print(f"Built questionnaire: {len(main.read_text(encoding='utf-8').splitlines())} lines.")
