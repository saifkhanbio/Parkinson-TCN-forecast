"""Descriptive figures for the new export; no fitted forecast results."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gbd_park_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors={'Male':'#236A91','Female':'#BC5478'}
rates=pd.read_csv(ROOT/'saudi_age_rates_2023.csv')
counts=pd.read_csv(ROOT/'saudi_age_counts_2023.csv')
pop=pd.read_csv(ROOT/'un_vs_gbd_population_comparison.csv')
pop=pop[pop.year.eq(2023)]
fig=plt.figure(figsize=(13.5,9.5),layout='constrained')
grid=fig.add_gridspec(2,2,height_ratios=[1,1.1])
a=fig.add_subplot(grid[0,0])
for outcome,color in [('prevalence','#236A91'),('incidence','#B77614')]:
    d=rates[rates.outcome.eq(outcome)].sort_values('age_start')
    a.plot(np.arange(len(d)),d.male_female_rate_ratio,marker='o',color=color,label=outcome.capitalize())
a.set_xticks(np.arange(len(d)),d.age_group,rotation=45,ha='right')
a.set_ylabel('Male:female rate ratio')
a.set_xlabel('Age group')
a.set_title('A. Saudi age-specific sex differences, 2023',loc='left',fontweight='bold')
a.legend(frameon=False)
a.grid(axis='y',alpha=.18)
b=fig.add_subplot(grid[0,1])
age_bands=['45-64','65-79','80+']
for offset,sex in [(-.18,'Male'),(.18,'Female')]:
    d=counts[counts.outcome.eq('prevalence') & counts.sex_name.eq(sex)].set_index('age_band').reindex(age_bands)
    bars=b.bar(np.arange(3)+offset,d.val,width=.34,label=sex,color=colors[sex])
    b.bar_label(bars,labels=[f'{x:,.0f}' for x in d.val],padding=3,fontsize=9)
b.set_xticks(np.arange(3),age_bands)
b.set_ylim(0,9500)
b.set_xlabel('Age group')
b.set_ylabel('GBD-estimated prevalent persons')
b.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
b.set_title('B. Saudi prevalence counts, ages 45+, 2023',loc='left',fontweight='bold')
b.legend(frameon=False)
b.grid(axis='y',alpha=.18)
c=fig.add_subplot(grid[1,:])
locations=['Saudi Arabia','Bahrain','Kuwait','Oman','Qatar','United Arab Emirates']
for offset,sex in [(-.17,'Male'),(.17,'Female')]:
    d=pop[pop.sex_name.eq(sex)].set_index('location_name').reindex(locations)
    values=100*(d.un_based_to_native_gbd_case_ratio-1)
    bars=c.barh(np.arange(6)+offset,values,height=.30,label=sex,color=colors[sex])
    for bar,value in zip(bars,values):
        c.text(value+(1.1 if value>=0 else -1.1),bar.get_y()+bar.get_height()/2,
               f'{value:+.1f}%',va='center',ha='left' if value>=0 else 'right',fontsize=10)
c.set_yticks(np.arange(6),locations)
c.invert_yaxis()
c.axvline(0,color='#333333',lw=.8)
c.set_xlim(-57,33)
c.set_xlabel('Change from native GBD case total when the SAME age-specific rates are applied to UN populations (%)')
c.set_title('C. Population-source sensitivity of 2023 prevalence counts, ages 45+',loc='left',fontweight='bold')
c.legend(frameon=False,loc='upper left',ncol=2)
c.grid(axis='x',alpha=.18)
fig.suptitle('New GBD data: age–sex detail and population assumptions',fontsize=17,fontweight='bold')
fig.supxlabel('Exploratory point estimates; no significance or forecast-accuracy claims. C is a same-year sensitivity calculation, not a forecast.\nSources: supplied GBD 2023 export and prepared UN WPP 2024 age–sex population extract.',fontsize=9)
fig.savefig(ROOT/'new_data_findings.png',dpi=180)
fig.savefig(ROOT/'new_data_findings.svg')
print('Saved new_data_findings.png and .svg')
