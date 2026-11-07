#!/usr/bin/env python3
"""Batch ISY agricultural parser with schema checks and auditable failures."""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def load_texts(path):
    data=json.loads(path.read_text(encoding='utf-8'))
    payload=data.get('res',data)
    return [str(x) for x in payload.get('overall_ocr_res',{}).get('rec_texts',[])]


def check_schema(texts):
    compact=' '.join(texts).casefold()
    area=any(x in compact for x in ('superficie', 'area'))
    production='production' in compact
    yield_=any(x in compact for x in ('rendement','yield'))
    dates=sum(t.strip()=='1925' for t in texts)
    if not (area and production and yield_):
        return False,'missing area/production/yield header'
    if dates<3:
        return False,f'only {dates} exact 1925 headers (expected 3)'
    return True,'agricultural headers detected'


def run_command(args, log_path):
    result=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
    log_path.write_text(result.stdout,encoding='utf-8')
    return result.returncode, result.stdout[-700:].replace('\n',' | ')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ocr-root',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--target-period',default='1925')
    p.add_argument('--python',default=sys.executable)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    v1script=a.repo/'paddleocr/extraction/parse_isy_table.py'
    v2script=a.repo/'paddleocr/extraction/parse_isy_table_v2.py'
    for path in [v1script,v2script]:
        if not path.exists(): raise SystemExit(f'Missing parser: {path}')
    jobs=sorted(a.ocr_root.glob('page_*/region_*_table/result_*/*_table_res.json'))
    print('Recognized table JSON files:',len(jobs))
    if not jobs:
        raise SystemExit('No OCR JSONs found. Check OCR job outputs.')
    summaries=[]
    records=[]
    for input_json in jobs:
        page=input_json.parents[2].name
        region=input_json.parents[1].name
        # For path .../page_XXXX/region_YYY_table/result_01/foo_res.json
        if not page.startswith('page_') or not region.startswith('region_'):
            page=input_json.parents[2].name;region=input_json.parents[1].name
        out=a.output_dir/page/region
        out.mkdir(parents=True,exist_ok=True)
        summary={'page':page,'region':region,'status':'','reason':'','rows':0,'countries':0,'regions':0,'numeric_fields':0,'ocr_json':str(input_json)}
        try:
            texts=load_texts(input_json)
            ok,reason=check_schema(texts)
            if not ok:
                summary.update(status='SKIPPED_SCHEMA',reason=reason)
            else:
                v1out=out/'v1';v2out=out/'v2'
                cmd1=[a.python,str(v1script),'--input-json',str(input_json),'--output-dir',str(v1out),'--target-period',a.target_period]
                rc1,msg1=run_command(cmd1,out/'v1.log')
                if rc1:
                    summary.update(status='FAILED_V1',reason=msg1)
                else:
                    cmd2=[a.python,str(v2script),'--input-json',str(v1out/'isy_target_period.json'),'--output-dir',str(v2out)]
                    rc2,msg2=run_command(cmd2,out/'v2.log')
                    if rc2:
                        summary.update(status='FAILED_V2',reason=msg2)
                    else:
                        final_csv=v2out/'isy_target_period.csv'
                        with final_csv.open(newline='',encoding='utf-8') as f:
                            rows=list(csv.DictReader(f))
                        count_vals=sum(bool((r.get(k) or '').strip()) for r in rows for k in ('area_1925_raw','production_1925_raw','yield_1925_raw'))
                        summary.update(status='PARSED',reason='automatic extraction; not manually validated',rows=len(rows),countries=sum(r.get('row_type')=='country' for r in rows),regions=sum(r.get('row_type') in ('region_total','region_header') for r in rows),numeric_fields=count_vals)
                        for r in rows:
                            records.append({
                                'page':page,'region':region,'row_type':r.get('row_type',''),
                                'country_en_raw':r.get('country_en_raw',''),
                                'country_fr_raw':r.get('country_fr_raw',''),
                                'parent_region':r.get('parent_region',''),
                                'area_1925_raw':r.get('area_1925_raw',''),
                                'production_1925_raw':r.get('production_1925_raw',''),
                                'yield_1925_raw':r.get('yield_1925_raw',''),
                                'entity_qualifier':r.get('entity_qualifier',''),
                                'validation_status':'UNVERIFIED',
                            })
        except Exception as ex:
            summary.update(status='ERROR',reason=repr(ex))
        summaries.append(summary)
        print(f"{page}/{region}: {summary['status']} | {summary['rows']} rows | {summary['numeric_fields']} values | {summary['reason'][:100]}",flush=True)
    sm_fields=['page','region','status','reason','rows','countries','regions','numeric_fields','ocr_json']
    with (a.output_dir/'run_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=sm_fields);w.writeheader();w.writerows(summaries)
    ex_fields=['page','region','row_type','country_en_raw','country_fr_raw','parent_region','area_1925_raw','production_1925_raw','yield_1925_raw','entity_qualifier','validation_status']
    with (a.output_dir/'all_extracted_rows_UNVERIFIED.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=ex_fields);w.writeheader();w.writerows(records)
    print('\nTOTAL',len(summaries),'tables;',sum(s['status']=='PARSED' for s in summaries),'parsed;',len(records),'candidate rows;',sum(s['numeric_fields'] for s in summaries),'numeric fields (UNVERIFIED)')
    print('Summary:',a.output_dir/'run_summary.csv')
    print('Extracted rows:',a.output_dir/'all_extracted_rows_UNVERIFIED.csv')

if __name__=='__main__':main()
