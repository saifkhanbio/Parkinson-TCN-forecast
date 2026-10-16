"""Sample host and GPU use while the two approved study queues run."""
import csv
import argparse
import json
from pathlib import Path
import subprocess
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'work/secondary-resources'
out.mkdir(parents=True,exist_ok=True)
path=out/'samples.csv'
parser=argparse.ArgumentParser()
parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
if path.exists() and not args.resume:
    raise FileExistsError('Preserve previous resource measurements')
rows=[]
if args.resume:
    with path.open() as stream:
        for row in csv.DictReader(stream):
            rows.append({key:value if key=='utc' else float(value) for key,value in row.items()})
    (out/'monitor_restart.json').write_text(json.dumps(dict(reason='Initial telemetry process ended with signal 15; model queues continued',
        prior_last_sample_utc=rows[-1]['utc'],restart_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
psutil.cpu_percent()
with path.open('a' if args.resume else 'w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=['utc','cpu_percent','available_gib','gpu_percent','gpu_memory_mib','gpu_power_w','gpu_temperature_c'])
    if not args.resume:
        writer.writeheader()
    while True:
        try:
            command=['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits']
            values=subprocess.check_output(command,text=True,timeout=8).strip().splitlines()[0].split(',')
            row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cpu_percent=psutil.cpu_percent(),
                     available_gib=psutil.virtual_memory().available/2**30,gpu_percent=float(values[0]),
                     gpu_memory_mib=float(values[1]),gpu_power_w=float(values[2]),gpu_temperature_c=float(values[3]))
            rows.append(row);writer.writerow(row);stream.flush()
            if len(rows)%15==0:
                print(f"Resources: CPU {row['cpu_percent']:.1f}%, GPU {row['gpu_percent']:.1f}%, RAM available {row['available_gib']:.1f} GiB",flush=True)
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            print(f'Resource sample unavailable: {exc}',flush=True)
        states=[]
        for run in ['secondary_v1','donor_comparisons_gpu_v1']:
            manifest=ROOT/'results'/run/'run_manifest.json'
            try:
                states.append(json.loads(manifest.read_text())['status']=='complete')
            except (OSError,ValueError):
                states.append(False)
        if all(states) or (out/'stop').exists():
            break
        time.sleep(2)
if rows:
    summary=dict(samples=len(rows),first_utc=rows[0]['utc'],last_utc=rows[-1]['utc'],
                 mean_cpu_percent=sum(r['cpu_percent'] for r in rows)/len(rows),
                 max_cpu_percent=max(r['cpu_percent'] for r in rows),
                 mean_gpu_percent=sum(r['gpu_percent'] for r in rows)/len(rows),
                 max_gpu_percent=max(r['gpu_percent'] for r in rows),
                 max_gpu_memory_mib=max(r['gpu_memory_mib'] for r in rows),
                 minimum_available_gib=min(r['available_gib'] for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
