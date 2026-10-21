"""Check final Word metrics, rare labels and errors against the fresh C3 run."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile

from src.models.distilbert_study import read_json, write_json
from src.datasets.goemotions import sha256
from scripts.verification.reproduce_report import verify_word_tables
from scripts.distilbert.reproduce_c3 import SEEDS, require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    root=args.run_root.resolve()
    data=read_json(root/'report_data/report_data.json')
    historical=read_json(root/'historical_verification/verification.json')
    histories=[row for row in historical['histories'] if row['architecture']!='distilbert']
    histories.extend({'architecture':'distilbert','seed':s,'selected_epoch':read_json(root/f'seed_{s}/run.json')['best_epoch']}
                     for s in SEEDS)
    result=verify_word_tables(args.report,{'records':data['all_runs'],'averages':data['mean_std'],'histories':histories})
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    with zipfile.ZipFile(args.report) as archive:
        doc=ET.fromstring(archive.read('word/document.xml'))
        relationships=ET.fromstring(archive.read('word/_rels/document.xml.rels'))
        image_digests={name:sha256(root/'report_data'/name)
                       for name in ('learning_curves.png','test_f1.png')}
        import hashlib
        # Word can renumber relationship IDs when updating the TOC. Match image
        # content and verify actual main-text and appendix references instead.
        image_relationships={node.attrib['Id']:hashlib.sha256(archive.read('word/'+node.attrib['Target'])).hexdigest()
                             for node in relationships if node.attrib['Type'].endswith('/image')}
        drawing_ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
                    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
        used=[node.attrib['{'+drawing_ns['r']+'}embed'] for node in doc.findall('.//a:blip',drawing_ns)]
        for name,expected in image_digests.items():
            require(sum(image_relationships.get(rid)==expected for rid in used)==2,
                    f'The report must use the measured {name} in the body and appendix')
    tables=[[[ ''.join(t.text or '' for t in c.findall('.//w:t',ns)) for c in r.findall('w:tc',ns)]
             for r in table.findall('w:tr',ns)] for table in doc.findall('.//w:tbl',ns)]
    checked=0

    def eq(actual,expected):
        nonlocal checked
        require(actual==str(expected),f'Report differs: {actual!r} != {expected!r}')
        checked+=1

    per=[]
    for seed in SEEDS:
        for metric in read_json(root/f'seed_{seed}/test_results.json')['results']:
            per.extend(dict(row,seed=seed,mode=metric['threshold_mode']) for row in metric['per_label'])
    for row,label in enumerate(data['rare_labels'],1):
        values=[]
        for col,mode in ((1,'fixed'),(2,'tuned')):
            array=[r['f1'] for r in per if r['label']==label and r['mode']==mode]
            mean,std=statistics.mean(array),statistics.stdev(array)
            eq(tables[15][row][col],'0.0000' if mean==std==0 else f'{mean:.4f} ± {std:.4f}')
            values.append(mean)
        delta=values[1]-values[0]
        eq(tables[15][row][3],f'{delta:+.4f}' if delta else '0.0000')
    for i,category in enumerate(('partial_multi_label','rare_false_negative','missed_extra_pair'),1):
        for col,arch in enumerate(('bert','roberta','distilbert'),1):
            row=next(r for r in data['error_counts'] if r['architecture']==arch and r['category']==category)
            eq(tables[16][i][col],row['error_samples'])
    ids=list(dict.fromkeys(r['id'] for r in data['report_examples']))
    for table,example_id in zip((17,18,19),ids):
        for row,arch in enumerate(('bert','roberta','distilbert'),1):
            item=next(r for r in data['report_examples'] if r['id']==example_id and r['architecture']==arch)
            eq(tables[table][row][1],', '.join(item['predicted_labels']) or 'Không nhãn')
            eq(tables[table][row][2], 'FN: '+(', '.join(item['missed_labels']) or '—')+'; FP: '+(', '.join(item['extra_labels']) or '—'))
    for row,pair in enumerate(data['top_pairs'],1):
        for col,key in enumerate(('missed_label','extra_label','bert','roberta','distilbert')):
            eq(tables[20][row][col],int(pair[key]) if col>=2 else pair[key])
    result.update(additional_rare_and_error_cells=checked,figures_sha256=image_digests,
                  checked_at_utc=datetime.now(timezone.utc).isoformat(),passed=True,
                  source_report_data_sha256=sha256(root/'report_data/report_data.json'))
    write_json(root/'report_data/word_verification.json',result)
    print(f'PASS: 111 metric cells plus {checked} rare/error cells and both measured figures.')


if __name__=='__main__':
    main()
