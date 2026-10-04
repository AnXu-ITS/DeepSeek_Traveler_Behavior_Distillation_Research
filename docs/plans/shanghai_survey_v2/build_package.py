"""Build questionnaire and data-only survey assets. Never edits models or production code."""
from __future__ import annotations
import copy
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VERSION = "shanghai-survey-v2.0"

def dump(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def alternative(mode, time, cost, exposure):
    return dict(mode=mode, available=True, travel_time_min=time, monetary_cost=cost,
                access_time_min=0, transfers=0, reliability_delay_min=0,
                weather_exposure=exposure, pt_feasible=0, egress_time_min=0,
                wait_time_min=0, in_vehicle_time_min=0, transfer_time_min=0,
                coverage_ratio=0)

def main():
    car = alternative("car", 20, 15, .05)
    pt = alternative("pt", 35, 4, .4)
    pt.update(access_time_min=7, egress_time_min=7, wait_time_min=5,
              in_vehicle_time_min=16, pt_feasible=1, coverage_ratio=.8)
    bike = alternative("bike", 25, 0, .9)
    walk = alternative("walk", 80, 0, 1)
    base = dict(
        card_id="B0",
        trip=dict(purpose="shopping", origin_type="home", destination_type="shop",
                  distance_km=6, desired_departure_min=480, desired_arrival_min=540,
                  time_constraint="hard"),
        context=dict(context_id="SP_B0", weather=dict(condition="clear", intensity=0),
                     road_congestion=.3, transit_delay_min=0, transit_disruption=False,
                     road_disruption=False, fare_multiplier=1., parking_cost_multiplier=1.,
                     congestion_charge=0.),
        alternatives=[car, pt, bike, walk],
        parking_cost_rmb=10, other_car_cost_rmb=5,
        note_zh="晴天。公共交通按本题时刻正常运行，道路正常通行。",
        note_en="Clear weather. Public transport operates as scheduled in this situation and roads are open.",
    )
    ids = ["B0", "W1", "D1", "WD1", "F1", "P1", "R1", "A_WALK", "A_WAIT", "A_TRANSFER"]
    cards = []
    for cid in ids:
        card = copy.deepcopy(base)
        card["card_id"] = cid
        card["context"]["context_id"] = "SP_" + cid
        cp = card["context"]
        pt = card["alternatives"][1]
        if cid in ("W1", "WD1"):
            cp["weather"] = dict(condition="rain", intensity=.75)
            card["note_zh"] = "暴雨。表中总时间已按本情景给定；除注明的情况外，没有其他服务变化。"
            card["note_en"] = "Heavy rain. The displayed totals apply to this situation; there are no other service changes unless stated."
        if cid in ("D1", "WD1"):
            cp["transit_delay_min"] = 15
            pt.update(travel_time_min=50, wait_time_min=20, reliability_delay_min=15)
            prefix_zh = "暴雨。" if cid == "WD1" else "晴天。"
            prefix_en = "Heavy rain. " if cid == "WD1" else "Clear weather. "
            card["note_zh"] = prefix_zh + "公共交通预计额外延误15分钟，体现在首次候车中，已计入20分钟候车和50分钟总时间；无需再加算。"
            card["note_en"] = prefix_en + "Public transport has an expected extra 15-minute delay before first boarding. It is already included in the 20-minute wait and 50-minute total; do not add it again."
        if cid == "F1":
            cp["fare_multiplier"] = 1.5
            pt["monetary_cost"] = 6
            card["note_zh"] = "晴天。本次公共交通票价由通常4元调整为6元，为通常票价的1.5倍。其余条件见表。"
            card["note_en"] = "Clear weather. The public transport fare is RMB 6 instead of the usual RMB 4, or 1.5 times the usual fare. Other conditions are shown below."
        if cid == "P1":
            cp["parking_cost_multiplier"] = 3
            card["parking_cost_rmb"] = 30
            card["alternatives"][0]["monetary_cost"] = 35
            card["note_zh"] = "晴天。本次停车费由通常10元调整为30元，为通常停车费的3倍；自驾总费用为35元，已包含停车。"
            card["note_en"] = "Clear weather. Parking costs RMB 30 instead of the usual RMB 10, three times the usual parking charge. Total driving cost is RMB 35, including parking."
        if cid == "R1":
            cp["road_disruption"] = True
            card["alternatives"][0].update(travel_time_min=40, reliability_delay_min=20)
            card["note_zh"] = "晴天。道路临时中断，需要绕行；自驾总时间为40分钟，已包含额外20分钟。给定公共交通路线、骑行路线和步行路线不受影响。"
            card["note_en"] = "Clear weather. A temporary road disruption requires a detour. Driving takes 40 minutes in total, including 20 extra minutes. The specified public transport, cycling and walking routes are unaffected."
        if cid == "A_WALK":
            pt.update(travel_time_min=45, access_time_min=12, egress_time_min=12)
            card["note_zh"] = "晴天。公共交通站外接驳路径较长，进站与离站步行各12分钟；公共交通门到门共45分钟。"
            card["note_en"] = "Clear weather. Public transport access and egress paths require 12 minutes of walking each. The door-to-door transit total is 45 minutes."
        if cid == "A_WAIT":
            pt.update(travel_time_min=45, wait_time_min=15)
            card["note_zh"] = "晴天。按本题给定的正常班次安排，首次候车为15分钟，并非突发故障；公共交通门到门共45分钟。"
            card["note_en"] = "Clear weather. Under the scheduled service in this situation, waiting before first boarding takes 15 minutes; this is not an unexpected fault. The door-to-door transit total is 45 minutes."
        if cid == "A_TRANSFER":
            pt.update(travel_time_min=45, transfers=1, transfer_time_min=10)
            card["note_zh"] = "晴天。公共交通需换乘1次，换乘步行及等待合计10分钟，车内时间合计16分钟；公共交通门到门共45分钟。"
            card["note_en"] = "Clear weather. Public transport requires one transfer, with 10 minutes of transfer walking and waiting. Time aboard vehicles totals 16 minutes; the door-to-door total is 45 minutes."
        cards.append(card)

    contrasts = [
        dict(id="rain", terms={"W1":1,"B0":-1}, target_mode="pt"),
        dict(id="delay", terms={"D1":1,"B0":-1}, target_mode="pt"),
        dict(id="rain_delay_interaction", terms={"WD1":1,"W1":-1,"D1":-1,"B0":1}, target_mode="pt"),
        dict(id="fare", terms={"F1":1,"B0":-1}, target_mode="pt"),
        dict(id="parking", terms={"P1":1,"B0":-1}, target_mode="car"),
        dict(id="road", terms={"R1":1,"B0":-1}, target_mode="car"),
        dict(id="walk_vs_wait_equal_total", terms={"A_WALK":1,"A_WAIT":-1}, target_mode="pt"),
        dict(id="transfer_vs_wait_equal_total", terms={"A_TRANSFER":1,"A_WAIT":-1}, target_mode="pt"),
    ]
    dump("cards.json", dict(version=VERSION, status="pilot_design_not_empirically_validated",
                           supported_modes=["car","pt","bike","walk"], cards=cards,
                           planned_contrasts=contrasts))
    sequence = [0,1,9,2,8,3,7,4,6,5]
    forms = [dict(form_id=f"F{k+1:02d}", card_order=[ids[(x+k)%10] for x in sequence]) for k in range(10)]
    dump("forms.json", dict(version=VERSION, design="even-size Williams order design",
                           allocation="uniform_random_at_respondent_level_before_any_SP_answers",
                           forms=forms))
    labels = dict(car=("自驾小汽车","Driving"), pt=("公共交通","Public transport"),
                  bike=("普通人力自行车","Conventional pedal bicycle"), walk=("步行","Walking"))
    parts = [(HERE / "participant_sections.source.txt").read_text(encoding="utf-8")]
    for idx, c in enumerate(cards, 1):
        parts.append(f"\n## 情景卡 {idx:02d} / Situation card {idx:02d}\n\n<!-- card_id={c['card_id']}，显示序号由所分配题序决定，勿向受访者显示研究代号。 -->\n")
        parts.append(c["note_zh"] + "\n\n" + c["note_en"] + "\n")
        parts.append("本题仍是约6公里的到店取货，原计划08:00出发，最晚09:00到达。\nThe same 6 km collection trip applies: planned departure 08:00, arrival by 09:00.\n")
        parts.append("| 方式 / Mode | 门到门总时间 / Total minutes | 本人新增费用 / Additional RMB |\n|---|---:|---:|")
        for a in c["alternatives"]:
            zh, en = labels[a["mode"]]
            parts.append(f"| {zh} / {en} | {a['travel_time_min']} | {a['monetary_cost']} |")
        parts.append(f"\n自驾费用分解：油电等费用{c['other_car_cost_rmb']}元＋停车{c['parking_cost_rmb']}元，已含在表内。\nDriving cost includes RMB {c['other_car_cost_rmb']} for fuel/energy and other running costs plus RMB {c['parking_cost_rmb']} parking; already included above.\n")
        pt = c["alternatives"][1]
        parts.append("**公共交通时间分解，已计入总时间 / Public transport components, already included**\n")
        parts.append(f"进站步行{pt['access_time_min']}＋首次候车{pt['wait_time_min']}＋车内{pt['in_vehicle_time_min']}＋换乘步行及等待{pt['transfer_time_min']}＋离站步行{pt['egress_time_min']}＝{pt['travel_time_min']}分钟；换乘{pt['transfers']}次。\n")
        parts.append(f"Access walk {pt['access_time_min']} + initial wait {pt['wait_time_min']} + in vehicle {pt['in_vehicle_time_min']} + transfer walking/waiting {pt['transfer_time_min']} + egress walk {pt['egress_time_min']} = {pt['travel_time_min']} minutes; {pt['transfers']} transfer(s).\n")
        parts.append(f"""<!-- question_id={c['card_id']}-M -->
**您会选择哪种方式？ / Which mode would you choose?**

- [ ] 自驾小汽车，仅可用时显示可选 / Driving, selectable only if available
- [ ] 公共交通 / Public transport
- [ ] 普通人力自行车，仅可用时显示可选 / Conventional bicycle, selectable only if available
- [ ] 步行 / Walking
- [ ] 无法在给定方式中选择 / Unable to choose among the offered modes
- [ ] 不愿回答 / Prefer not to answer

<!-- question_id={c['card_id']}-T -->
**相对于08:00，您会如何安排出发？ / How would you depart relative to 08:00?**

- [ ] 不调整，08:00出发 / No change, leave at 08:00
- [ ] 提前 / Earlier → 选择分钟数 / Select minutes
- [ ] 延后 / Later → 选择分钟数 / Select minutes
- [ ] 无法判断 / Unable to judge
- [ ] 不愿回答 / Prefer not to answer

分钟选项：1–120的每个整数、超过120分钟、无法估计。平台同时显示换算后的时刻。
Minute options: every integer from 1 to 120, more than 120, or unable to estimate. The survey also displays the resulting clock time.

<!-- M为无法选择或不愿答时跳过T；跳过记结构缺失，不记0。 -->

<!-- question_id={c['card_id']}-U -->
**仅当无法选择时显示 / Shown only if unable to choose**

最主要原因是什么？ / What is the main reason?

- [ ] 这些方式均不适合完成本次出行 / None of these modes suits this trip
- [ ] 身体状况使可用方式都无法完成本次出行 / Physical limitations prevent completing the trip with the offered modes
- [ ] 无法理解或想象题目条件 / Unable to understand or imagine the situation
- [ ] 其他原因 / Another reason
- [ ] 不愿说明 / Prefer not to explain

---
""")
    parts.append("""## 五、最后两个问题 / 5. Two final questions

### E01 如果在现实中遇到这些情况，您是否还会考虑题目没有提供的方式或安排？可多选
### In real life, would you consider any options not offered in these situations? Select all that apply

- [ ] 出租车或网约车 / Taxi or ride-hailing
- [ ] 电动或电助力自行车 / Electric or electrically assisted bicycle
- [ ] 共享单车 / Shared bicycle
- [ ] 亲友接送 / A lift from friends or family
- [ ] 改为配送或请人代领 / Delivery or collection by someone else
- [ ] 取消出行或改日领取 / Cancel the trip or collect on another day
- [ ] 通常不会考虑其他方式 / Usually no other option
- [ ] 无法判断或不愿回答 / Unable to judge or prefer not to answer

### E02 总体而言，您觉得这些情景容易理解吗？
### Overall, how easy were the situations to understand?

- [ ] 容易理解 / Easy
- [ ] 大部分能理解，少数地方不清楚 / Mostly clear, with a few unclear points
- [ ] 有多处不清楚 / Several unclear points
- [ ] 很难理解 / Very difficult
- [ ] 不愿回答 / Prefer not to answer

感谢您的参与。/ Thank you for participating.
""")
    (ROOT / "docs/plans/Shanghai_Travel_Intention_Survey.md").write_text("\n".join(parts), encoding="utf-8")
    headers = {
        "respondents_template.csv": ["respondent_id","survey_version","form_id","language","recruitment_channel","completed_at","S01","S02","S03"] + [f"P{i:02d}" for i in range(1,15)] + ["C00","C01_first","C01_second","C02_first","C02_second","E01","E02"],
        "sp_responses_template.csv": ["respondent_id","survey_version","form_id","card_id","task_order","car_available","bike_available","choice_status","chosen_mode","unable_reason","shift_status","shift_direction","shift_minutes","departure_clock","page_seconds"],
        "rp_trips_template.csv": ["respondent_id","reference_date","R01","R02","R03_departure","R03_arrival","R03_arrival_day","R04_departure_status","R04_departure","R04_arrival_status","R04_arrival","R04_arrival_day","R05","R06_origin_district","R06_origin_landmark","R06_dest_district","R06_dest_landmark","R07_distance_km","R08_main_mode","R09_access_modes","R10_cost_rmb","R11_access_min","R11_wait_min","R11_in_vehicle_min","R11_transfer_min","R11_egress_min","R11_transfers","R11_nonwalk_access"],
        "canonical_personas_template.csv": ["persona_id","age_group","income_group","occupation","household_size","has_children","car_ownership","driving_license","bike_ownership","transit_pass","habitual_mode","schedule_flexibility","mobility_limitation","sp_car_available","sp_bike_available"],
    }
    for name, cols in headers.items():
        with (HERE / name).open("w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerow(cols)
    print(f"Built {len(cards)} cards, {len(forms)} order forms, questionnaire and {len(headers)} empty templates.")
if __name__ == "__main__":
    main()
