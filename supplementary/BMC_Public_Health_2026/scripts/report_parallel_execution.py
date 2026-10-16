"""Render recorded CPU/GPU utilization without treating gaps as measured time."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gbd_park_matplotlib')
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_local_baselines import sha
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    source=ROOT/'work/secondary-resources/samples.csv'
    frame=pd.read_csv(source)
    frame['time']=pd.to_datetime(frame.utc,utc=True)
    assert frame.time.is_monotonic_increasing and not frame.time.duplicated().any()
    assert frame[['cpu_percent','gpu_percent']].ge(0).all().all() and frame[['cpu_percent','gpu_percent']].le(100).all().all()
    assert np.isfinite(frame[['cpu_percent','gpu_percent','available_gib','gpu_memory_mib']]).all().all()
    elapsed=(frame.time-frame.time.iloc[0]).dt.total_seconds()/60
    gaps=frame.time.diff().dt.total_seconds().gt(15)
    plot=frame.copy()
    plot.loc[gaps,['cpu_percent','gpu_percent','available_gib','gpu_memory_mib']]=np.nan
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    axes[0].plot(elapsed,plot.cpu_percent,color='#2676a6',lw=1,label='Host CPU')
    axes[0].plot(elapsed,plot.gpu_percent,color='#d58a27',lw=1,label='GPU (NVIDIA-reported)')
    axes[0].set(ylabel='Recorded utilization (%)',ylim=(0,105))
    axes[0].legend(loc='lower left',ncol=2)
    axes[1].plot(elapsed,plot.available_gib,color='#347d51',label='Available host RAM (GiB)')
    axes[1].plot(elapsed,plot.gpu_memory_mib/1024,color='#915fa1',label='Used GPU memory (GiB)')
    axes[1].set(ylabel='Memory (GiB)',xlabel='Minutes since first recorded sample')
    axes[1].legend(loc='best',ncol=2)
    for axis in axes:
        axis.grid(alpha=.2)
        for index in frame.index[gaps]:
            axis.axvspan(elapsed.iloc[index-1],elapsed.iloc[index],color='gray',alpha=.15)
    fig.suptitle('Parallel study execution: 12 CPU workers and 4 GPU workers\nShaded gaps were not measured',fontsize=12)
    out=ROOT/'reports/parallel_execution_v1';out.mkdir(exist_ok=True)
    for suffix in ['png','svg','pdf']:
        fig.savefig(out/f'utilization.{suffix}',dpi=200)
    plt.close(fig)
    validation=dict(passed=True,samples=len(frame),first_sample=str(frame.time.iloc[0]),last_sample=str(frame.time.iloc[-1]),
                    sampling_gaps=int(gaps.sum()),mean_sampled_cpu_percent=float(frame.cpu_percent.mean()),
                    mean_sampled_gpu_percent=float(frame.gpu_percent.mean()),max_cpu_percent=float(frame.cpu_percent.max()),
                    max_gpu_percent=float(frame.gpu_percent.max()),minimum_available_ram_gib=float(frame.available_gib.min()),
                    max_used_gpu_memory_mib=float(frame.gpu_memory_mib.max()),input_sha256=sha(source),script_sha256=sha(Path(__file__)))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    report=out/'report.md'
    text=report.read_text().split('## Production measurements')[0].rstrip()
    text+='\n\n## Production measurements\n\n'
    text+=f"Across {len(frame):,} recorded samples, mean host CPU utilization was {frame.cpu_percent.mean():.1f}% and mean GPU utilization was {frame.gpu_percent.mean():.1f}%. Peaks were {frame.cpu_percent.max():.1f}% and {frame.gpu_percent.max():.1f}%, respectively. Minimum available RAM was {frame.available_gib.min():.2f} GiB; peak GPU memory use was {frame.gpu_memory_mib.max()/1024:.2f} GiB. There were {int(gaps.sum())} gaps longer than fifteen seconds. These are sample averages across training and orchestration phases, not uninterrupted or training-only averages.\n\n"
    text+='![Recorded processor use and memory](utilization.png)\n\n'
    text+='The [validation record](validation.json) identifies the exact sample file and report script. Timing and utilization describe this machine and workload; they are not statistical or clinical study results.\n'
    report.write_text(text)
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    main()
