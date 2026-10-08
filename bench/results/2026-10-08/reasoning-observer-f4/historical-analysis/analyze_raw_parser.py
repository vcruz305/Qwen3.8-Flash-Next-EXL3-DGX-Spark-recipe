"""CPU replay of saved raw observer output against the actual Qwen channel parser."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path('/home/vcruz/src/qwen-overnight-20261008/tabbyapi-agent')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from endpoints.OAI.utils.stream_parser import Qwen3CoderStreamParser,REASONING,CONTENT,TOOL
from endpoints.OAI.utils.tools import get_toolcall_tags
from inspect_prompt_and_sse import observed

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    source_head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    assert source_head=='f4fb6b73a4adbf5f4faa4a3d9b90b3455c8c7fd6'
    source_changes=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
    assert not source_changes
    root=HERE/'live-observer';result=json.loads((root/'result.json').read_text())
    assert result['state']=='completed' and result['observer_traces']['completed']==6
    start,end=get_toolcall_tags('qwen3_5')
    rows=[];checks=0
    for i,item in enumerate(result['observer_traces']['traces']):
        path=root/'raw'/item['file'];assert sha(path)==item['sha256'];trace=json.loads(path.read_text())
        report_path=root/f'auto{i//2+1}.json';report=json.loads(report_path.read_text())
        name='auto_reasoning_stream' if trace['streaming_mode'] else 'auto_reasoning_nonstream'
        case=next(c for c in report['cases'] if c['name']==name)
        entry=report['requests'][case['request_indices'][0]]
        public=observed(entry);raw=trace['raw_finish']['text']
        partitions=[]
        for width in (1,7,31,max(1,len(raw))):
            parser=Qwen3CoderStreamParser(reasoning_start='<think>',reasoning_end='</think>',
                tool_start=start,tool_end=end,start_in_reasoning=True,tool_calls_in_reasoning=True)
            events=[]
            for offset in range(0,len(raw),width):events.extend(parser.feed(raw[offset:offset+width]))
            events.extend(parser.finish())
            routed={channel:'' for channel in (REASONING,CONTENT,TOOL)}
            for channel,text in events:routed[channel]+=text
            assert routed[REASONING]==public['reasoning']
            assert routed[CONTENT]==public['content']
            assert routed[TOOL]==''
            partitions.append({'width':width,'matches_api_channels':True});checks+=1
        forced=[e for e in trace['events'] if e['kind']=='forced_token_sampled']
        requested=[e for e in trace['events'] if e['kind']=='injection_requested']
        marker=raw.find('</think>')
        rows.append({'trace':item['file'],'trace_sha256':sha(path),'api_report':report_path.name,
            'api_report_sha256':sha(report_path),'case':name,'case_status':case['status'],
            'request_id':trace['request_id'],'forced_positions':[e['new_tokens_before'] for e in forced],
            'forced_ids':[e['token_id'] for e in forced],
            'queued_results_at_injection':[e['state']['queued_results'] for e in requested],
            'raw_closing_tag_count':raw.count('</think>'),
            'before_first_real_closing_tag':raw[:marker] if marker>=0 else None,
            'after_first_real_closing_tag':raw[marker+len('</think>'):] if marker>=0 else None,
            'raw_text':raw,'observed_reasoning':public['reasoning'],'observed_content':public['content'],
            'parser_partitions':partitions,'finish_reason':public['finish'],'stream_done':public['done']})
    positions=[row['forced_positions'][0] for row in rows]
    output=HERE/'raw-parser-analysis.json'
    report={'scope':'Saved synthetic observer data; CPU parser replay only. No new inference. Instrumented timings are not performance measurements.',
        'source_commit':source_head,'source_tracked_changes':source_changes,
        'stream_parser_sha256':sha(ROOT/'endpoints/OAI/utils/stream_parser.py'),
        'script_sha256':sha(Path(__file__)),'controller_result_sha256':sha(root/'result.json'),
        'completed_traces':6,'parser_replays':checks,'all_channel_matches':True,
        'observed_forced_positions':positions,'producer_boundary_varied':len(set(positions))>1,
        'cases':rows,'interpretation':'The failing raw response contains the quotation continuation after a real reasoning closing marker; replaying the unchanged parser exactly matches every saved API channel. No routing mismatch was found. The same loaded model forced its24-token budget at producer position24 in five successful traces and28 in the failed trace. This establishes variable cutoff placement and an observed association with the failure; it does not guarantee that every response at position24 will be semantically correct.'}
    with output.open('x') as file:json.dump(report,file,indent=2,ensure_ascii=False);file.write('\n')
    print(json.dumps({'path':str(output),'sha256':sha(output),'source':source_head,'traces':6,'parser_replays':checks,'all_channel_matches':True,'positions':positions,'case_statuses':[r['case_status'] for r in rows]}))
if __name__=='__main__':main()
