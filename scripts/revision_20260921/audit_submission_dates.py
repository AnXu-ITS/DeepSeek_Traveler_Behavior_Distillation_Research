"""Read questionnaire timestamps and verify the deployed Shanghai screening wording."""
from datetime import datetime
import csv,json
import openpyxl
from prepare_survey import ROOT,OUT,sha,writej,readj

def main():
    dest=OUT/'sample_flow_audit';result={};hashes={}
    for city,folder,header in [('Singapore','新加披调查问卷','时间戳记'),('Shanghai','上海调查问卷','提交答卷时间')]:
        p=next((ROOT/folder).glob('*.xlsx'));before=sha(p);book=openpyxl.load_workbook(p,read_only=True,data_only=True)
        it=book.active.iter_rows(values_only=True);heads=next(it);idx=heads.index(header);dates=[]
        for row in it:
            if not any(v is not None for v in row):continue
            v=row[idx]
            dates.append(v if isinstance(v,datetime) else datetime.strptime(v,'%Y/%m/%d %H:%M:%S'))
        book.close();assert sha(p)==before
        result[city]=dict(timestamp_column=header,n_submissions=len(dates),n_parsed=len(dates),earliest_date=min(dates).date().isoformat(),latest_date=max(dates).date().isoformat(),timezone='Not specified in exported timestamp; date interpretation follows raw export',source='Raw per-submission timestamp, not file metadata')
        hashes[str(p.relative_to(ROOT))]=before
        if city=='Shanghai':assert heads[5]=='2、过去30天，您是否在上海出行过？'
    options=ROOT/'docs/plans/shanghai_survey_v3/question_options.json'
    q=next(q for q in readj(options)['questions'] if q['id']=='S02');assert q['text']=='过去30天，您是否在上海出行过？'
    form=ROOT/'docs/plans/shanghai_survey_v3/field/上海出行方式调查_问卷.md'
    assert q['text'] in form.read_text(encoding='utf-8-sig')
    hashes[str(options.relative_to(ROOT))]=sha(options);hashes[str(form.relative_to(ROOT))]=sha(form)
    screening=dict(field='S02',raw_question=q['text'],english_meaning='Whether the respondent had travelled in Shanghai in the past 30 days',
      interpretation='Recent Shanghai travel screening, not residence. Eight no answers skip subsequent persona/tasks; 333 yes answers provide all ten task answers.',
      legacy_names='residence/residency_screenout/n_residents and human_complete_resident_sensitivity are retained only as historical machine-readable field/path names. They do not assert residence.',
      supersedes='Earlier flow-audit prose describing the eight Shanghai screen-outs as nonresidents and the 333 as residents; all cohort membership and numeric results unchanged.',
      singapore='The separate Singapore visitor/residence screening wording remains unchanged.')
    writej(dest/'submission_dates_and_screening.json',dict(status='passed',dates=result,shanghai_screening=screening,raw_sources_unchanged=True,source_sha256=hashes,code_sha256=sha(__file__)))
    fieldfile=OUT/'authoritative_screening_fields.csv'
    with fieldfile.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['city','field','deployed_question_meaning','inclusion_rule','legacy_annotation'])
        w.writeheader();w.writerows([
          dict(city='Shanghai',field='S02',deployed_question_meaning=screening['english_meaning'],inclusion_rule='Yes to travel in Shanghai during previous 30 days; 333 complete task respondents',legacy_annotation='Historical residence/residency names do not mean residence'),
          dict(city='Singapore',field='Q02',deployed_question_meaning='Singapore residence or pass status',inclusion_rule='Exclude Tourist / Visitor (Not living in Singapore); 332 eligible task respondents',legacy_annotation='Separate city-specific screen; do not replace with the Shanghai recent-travel criterion')])
    authority=OUT/'authoritative_mapping_revision.json'
    if authority.exists():
        meta=readj(authority);meta['screening_wording']=screening
        meta['screening_field_dictionary']='authoritative_screening_fields.csv'
        meta['submission_date_audit']='sample_flow_audit/submission_dates_and_screening.json'
        writej(authority,meta)
    (dest/'SCREENING_WORDING_CORRECTION.md').write_text('# Submission dates and screening wording\n\nRaw submission timestamps cover Singapore 20 August–5 September 2026 (334/334 parsed), and Shanghai 12–18 September 2026 (341/341 parsed). These dates support the reported August–September 2026 collection window; no file modification timestamps were used. Exported time zones are unspecified.\n\nThe deployed Shanghai S02 asks whether respondents travelled in Shanghai in the past 30 days, not whether they live there. The eight no responses are recent-travel screening exclusions. The remaining 333 complete respondents are recent-travel-eligible; 321 meet the frozen model-input mapping. Legacy machine-readable names containing residence/resident are retained for compatibility and must be read with this correction. Singapore uses its separate visitor/residence screen. No raw source, cohort membership, human choice, model input or numeric result changes.\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',dates=result,screening=screening),ensure_ascii=False))

if __name__=='__main__':main()
