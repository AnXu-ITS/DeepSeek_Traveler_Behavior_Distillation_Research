"""Person-trip denominator and walk/bike physical-speed sensitivity, frozen inputs."""
from __future__ import annotations
import argparse,shutil,time
from pathlib import Path
from helsinki_execution import ROOT,OUT,CARDS,read,write,writejl,sha,csvout,events,summarize_people,prepare_java,run_matsim

def singapore():
 rows=[]
 for card in ['C0_baseline','C1_heavy_rain','C2_fare_increase','C3_transit_delay','C4_road_disruption','C5_joint_rain_delay']:
  src=ROOT/'outputs/singapore_phase_c_s9'/card;dst=OUT/'audits/singapore'/card
  if (dst/'summary.json').exists():rows.append(read(dst/'summary.json'));continue
  man=read(src/'adapter_manifest.json')['decisions'];ids={r['persona_id'] for r in man}
  print('Singapore person-trip audit',card,flush=True)
  ev=src/'output/ITERS/it.0/0.events.xml.zst';people,types=events(ev,ids)
  writejl(dst/'person_trip_events.jsonl',[dict(person_id=k,**v) for k,v in people.items()])
  old=read(src/'phase_c_result.json');summary=dict(card=card,**summarize_people(people),historical_reported_mean_trip_time_min=old['metrics']['mean_trip_time_min'],historical_failed_trips=old['metrics']['failed_trips'],event_sha256=sha(ev),interpretation='Historical mean_trip_time_min counts completed legs, not complete door-to-door person trips. Failed legs are excluded; horizon statistic retains departed people who fail to complete, without treating failure as zero.')
  write(dst/'summary.json',summary);rows.append(summary);csvout(OUT/'audits/singapore_summary.csv',rows)
 write(OUT/'audits/singapore_summary.json',rows)

def speed_config(out):
 """Official MATSim modeVehicleTypesFromVehiclesData, fixed maximum velocities."""
 xml=['<?xml version="1.0" encoding="UTF-8"?>','<vehicleDefinitions xmlns="http://www.matsim.org/files/dtd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.matsim.org/files/dtd http://www.matsim.org/files/dtd/vehicleDefinitions_v1.0.xsd">']
 for mode,speed in [('car',1000000),('bike',4.17),('walk',1.34)]:
  xml.append(f'<vehicleType id="{mode}"><length meter="7.5"/><width meter="1.0"/><maximumVelocity meterPerSecond="{speed}"/><accessTime secondsPerPerson="1.0"/><egressTime secondsPerPerson="1.0"/><doorOperation mode="serial"/><passengerCarEquivalents pce="1.0"/></vehicleType>')
 xml.append('</vehicleDefinitions>');(out/'mode_vehicles.xml').write_text('\n'.join(xml),encoding='utf-8')
 cfg=(out/'config.xml').read_text(encoding='utf-8');cfg=cfg.replace('<module name="qsim">','<module name="qsim">\n<param name="vehiclesSource" value="modeVehicleTypesFromVehiclesData" />')
 cfg=cfg.replace('</config>','<module name="vehicles"><param name="vehiclesFile" value="mode_vehicles.xml" /></module>\n</config>')
 cfg=cfg.replace('deleteDirectoryIfExists','failIfDirectoryExists');(out/'config.xml').write_text(cfg,encoding='utf-8')

def shanghai():
 java=prepare_java();rows=[]
 for card in ['B0','D1']:
  src=ROOT/'outputs/shanghai_survey_matsim_321x10_v1'/card
  man=read(src/'adapter_manifest.json')['decisions'];ids={r['person_id'] for r in man}
  for setting in ['historical_link_speed','corrected_mode_max_speed']:
   dst=OUT/'audits/shanghai'/f'{card}_{setting}';sp=dst/'summary.json'
   if sp.exists():rows.append(read(sp));continue
   dst.mkdir(parents=True,exist_ok=True);print('Shanghai speed audit',card,setting,flush=True)
   if setting=='historical_link_speed':ev=src/'output/ITERS/it.0/0.events.xml.zst';code=0;runtime=0.
   else:
    if (dst/'output').exists():raise RuntimeError(f'Preserve unfinished run: {dst}')
    for name in ['population.xml','config.xml']:shutil.copy2(src/name,dst/name)
    speed_config(dst);started=time.time();code,tail=run_matsim(dst,java);runtime=time.time()-started
    if code:write(sp,dict(card=card,setting=setting,exit_code=code,tail=tail));raise RuntimeError(f'MATSim failure {card}')
    ev=dst/'output/ITERS/it.0/0.events.xml.zst'
   people,types=events(ev,ids);writejl(dst/'person_trip_events.jsonl',[dict(person_id=k,**v) for k,v in people.items()])
   summary=dict(card=card,setting=setting,exit_code=code,runtime_s=runtime,**summarize_people(people),population_unchanged=(setting=='historical_link_speed' or sha(src/'population.xml')==sha(dst/'population.xml')),event_sha256=sha(ev),source_population_sha256=sha(src/'population.xml'),speed_ms={'walk':1.34,'bike':4.17} if setting=='corrected_mode_max_speed' else 'road link freespeed',interpretation='Identical assigned modes, departure times, routes and physical timetable; only road-mode maximum velocities capped. Any new missed service/failure is retained; this isolates execution speed and does not calibrate behavior.')
   write(sp,summary);rows.append(summary);csvout(OUT/'audits/shanghai_speed_summary.csv',rows)
 write(OUT/'audits/shanghai_speed_summary.json',rows)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('audit',choices=['singapore','shanghai']);a=ap.parse_args()
 {'singapore':singapore,'shanghai':shanghai}[a.audit]()
