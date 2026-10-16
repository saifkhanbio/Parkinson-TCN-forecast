"""Eight JDR figures regenerated from retained numerical evidence."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gbd_jdr_matplotlib')
from pathlib import Path
import importlib.util
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/JDR_GBD_PARK'
FIG=OUT/'figures'
TYPOGRAPHY=[]
SELECTED=set(range(1,9))


def load(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def enlarge_forecast_text(fig,number):
 # At the manuscript's 6.55-inch display width, 16 source points become 8.32
 # points. Reflow dense legends and notes instead of shrinking their lettering.
 for text in fig.findobj(match=Text):text.set_fontsize(max(16,text.get_fontsize()))
 for ax in fig.axes:
  ax.tick_params(labelsize=16,pad=8)
  ax.xaxis.label.set_fontsize(17);ax.yaxis.label.set_fontsize(17)
  ax.set_title(ax.get_title(loc='left'),loc='left',fontsize=19)
 if number==6:
  fig.set_size_inches(12.6,10.4)
  fig.axes[0].get_subplotspec().get_gridspec().update(left=.10,right=.965,top=.70,bottom=.16,wspace=.38,hspace=.58)
  fig.texts[0].set_position((.075,.963));fig.texts[0].set_fontsize(22)
  fig.texts[1].set_position((.075,.923))
  for legend in list(fig.legends):legend.remove()
  forecast=load('plot_standalone_forecast_comparison')
  fig.legend(handles=[Line2D([],[],color=c,ls=ls,marker=m,mfc='white',ms=7,lw=2.2,label=label)
       for _,label,c,ls,m in forecast.SCENARIOS],loc='upper left',bbox_to_anchor=(.075,.893),
       ncol=1,frameon=False,fontsize=16,handlelength=2.5,labelspacing=.5,borderaxespad=0)
  for i,ax in enumerate(fig.axes):
   ax.set_xticks([2024,2026,2028]);ax.set_xlim(2023.80,2029.85)
   ax.tick_params(axis='x',pad=12)
   if i%3==0:
    ax.set_ylabel('Prevalent cases\n(thousands)' if i<3 else 'Annual incident\ncases (thousands)',fontsize=17,labelpad=12)
   callouts=ax.lines[-3:]
   ymax=ax.get_ylim()[1]
   positions=forecast.separate_labels([line.get_ydata()[0] for line in callouts],ymax*.135,ymax*.95)
   for text,line,y in zip(ax.texts,callouts,positions):
    line.set_ydata([line.get_ydata()[0],y,y]);text.set_y(y)
  fig.texts[2].set_position((.53,.091));fig.texts[2].set_fontsize(17)
  fig.texts[3].set_position((.075,.061));fig.texts[3].set_va('top')
  fig.texts[3].set_text('National scenario: 2024 anchor advanced with UN growth;\nGBD-aligned allocation within ages 80+.')
 elif number==7:
  fig.set_size_inches(12.6,9.3)
  fig.axes[0].get_subplotspec().get_gridspec().update(left=.10,right=.97,top=.75,bottom=.37,wspace=.28)
  fig.texts[0].set_position((.075,.958));fig.texts[0].set_fontsize(22)
  fig.texts[1].set_position((.075,.907))
  fig.legends[0].set_bbox_to_anchor((.073,.866))
  for i,ax in enumerate(fig.axes):
   ax.set_xticks([2023,2025,2028]);ax.set_xlim(2022.80,2030.65)
   if i==2:ax.set_title('C   Both sexes\n     combined',loc='left',fontsize=19)
   x=ax.get_position().x0
   header,prevalence,incidence=fig.texts[2+i*3:5+i*3]
   header.set_position((x,.263));header.set_text('2023 → 2028 counts')
   for text,y in [(prevalence,.219),(incidence,.143)]:
    label,values=text.get_text().split(': ',1)
    text.set_text(label+'\n'+values);text.set_position((x,y));text.set_va('top')
  fig.texts[-2].set_position((.53,.311));fig.texts[-2].set_fontsize(17)
  fig.texts[-1].set_position((.075,.069));fig.texts[-1].set_va('top')
  fig.texts[-1].set_text('GBD 2023 population baseline + UN growth. Point forecasts;\ncount changes include population growth and ageing.')


def save(fig,number):
 if number not in SELECTED:
  plt.close(fig);return
 # Allow the wider bold labels the same clear margins as the reference figures.
 if number==2:
  fig.subplots_adjust(left=.19,wspace=.43)
 if number==4:
  for ax in fig.axes[:2]:ax.set_ylabel('Population aged 80+\n(thousands)')
  for ax in fig.axes[2:]:ax.set_xlabel('80+ share of sex-specific\n45+ burden (%)')
 if number in [6,7,8]:enlarge_forecast_text(fig,number)
 # Match the example manuscript's bold lettering, including labels created by
 # imported plotting functions and tick labels materialized during drawing.
 fig.canvas.draw()
 for text in fig.findobj(match=Text):
  text.set_fontweight('bold')
 fig.canvas.draw()
 renderer=fig.canvas.get_renderer()
 # Locators retain unused ticks beyond axis limits; those labels are not drawn.
 unused_tick_labels=set()
 for ax in fig.axes:
  for axis in [ax.xaxis,ax.yaxis]:
   low,high=sorted(axis.get_view_interval())
   for tick in axis.get_major_ticks()+axis.get_minor_ticks():
    if tick.get_loc()<low-1e-8 or tick.get_loc()>high+1e-8:
     unused_tick_labels.update([tick.label1,tick.label2])
 visible=[t for t in fig.findobj(match=Text)
          if t.get_visible() and t.get_text().strip() and t not in unused_tick_labels]
 assert all(t.get_fontweight()=='bold' for t in visible)
 outside=[]
 for text in visible:
  bbox=text.get_window_extent(renderer)
  if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width+1 or bbox.y1 > fig.bbox.height+1:
   outside.append(text.get_text())
 TYPOGRAPHY.append({'figure':number,'visible_text_elements':len(visible),
   'font_weights':sorted({t.get_fontweight() for t in visible}),
   'font_families':sorted({f for t in visible for f in t.get_fontfamily()}),
   'font_size_range_pt':[min(t.get_fontsize() for t in visible),max(t.get_fontsize() for t in visible)],
   'minimum_font_size_at_manuscript_width_pt':round(min(t.get_fontsize() for t in visible)*6.55/fig.get_figwidth(),2),
   'text_outside_canvas':outside})
 assert not outside,(number,'Text extends outside figure canvas',outside)
 for ext in ['pdf','svg','png','tiff']:
  options={'dpi':600 if ext=='tiff' else 300,'facecolor':'white'}
  if ext=='tiff':options['pil_kwargs']={'compression':'tiff_lzw'}
  fig.savefig(FIG/f'figure_{number}.{ext}',**options)
 fig.savefig(FIG/f'figure_{number}_preview.png',dpi=100,facecolor='white')
 plt.close(fig)


def panel(ax,title,xlabel):
 ax.set_title(title,loc='left',weight='bold',fontsize=13,pad=13)
 ax.set_xlabel(xlabel,fontsize=10.5)
 ax.spines[['top','right','left']].set_visible(False)
 ax.tick_params(length=0,pad=5)
 ax.grid(axis='x',color='#E4E9ED',linewidth=.6)
 ax.set_axisbelow(True)


def disability():
 endpoint=pd.read_csv(ROOT/'reports/supporting_v1/endpoint_tcn_comparisons.csv')
 rates=pd.read_csv(ROOT/'reports/supporting_v1/rate_interval_five_origin.csv')
 point=pd.read_csv(ROOT/'reports/supporting_v1/burden_endpoint_points.csv')
 intervals=pd.read_csv(ROOT/'reports/supporting_v1/burden_interval_five_origin.csv')
 keys=[(o,s) for o in ['ylds','ylls','dalys','deaths'] for s in ['Male','Female']]
 names={'ylds':'YLDs','ylls':'YLLs','dalys':'DALYs','deaths':'Deaths'}
 colors={'Male':'#0072B2','Female':'#C55300'}
 fig,axs=plt.subplots(2,2,figsize=(12.6,10.4))
 fig.subplots_adjust(left=.13,right=.97,top=.88,bottom=.10,wspace=.36,hspace=.39)
 fig.text(.08,.961,'Forecasting disability and mortality burden',fontsize=19,weight='bold')
 fig.text(.08,.925,'Saudi Arabia · ages 45+ · five-year forecasts · separate evaluation by sex',fontsize=11,color='#536371')
 exported=[]
 for i,(o,s) in enumerate(keys):
  a=endpoint.loc[endpoint.target.eq('Saudi Arabia')&endpoint.outcome.eq(o)&endpoint.sex.eq(s)].iloc[0]
  for off,field,color,marker in [(-.2,'local_champion','#777777','s'),(0,'nonneural_champion','#009E73','^'),(.2,'tcn_adapted','#222222','o')]:
   axs[0,0].scatter(a[field],i+off,c=color,marker=marker,s=35)
  r=rates.loc[rates.target.eq('Saudi Arabia')&rates.outcome.eq(o)&rates.sex.eq(s)
        &rates.procedure.eq('direct')&rates.family.eq('tcn_adapted')&rates.horizon.eq(5)
        &rates.age_band.eq('45+')&rates.scale.eq('rate')&rates.level.eq(.8)].iloc[0]
  axs[0,1].barh(i,100*r.coverage,color=colors[s],height=.64)
  axs[0,1].text(100*r.coverage+2,i,f'{100*r.coverage:.1f}',va='center',fontsize=9)
  p=point.loc[point.target.eq('Saudi Arabia')&point.outcome.eq(o)&point.sex.eq(s)
       &point.procedure.eq('direct')&point.family.eq('tcn_adapted')&point.horizon.eq(5)
       &point.population_method.eq('log_trend_last8')&point.node.eq(s+'__80+_within_45+')].iloc[0]
  axs[1,0].plot([p.value,p.observed],[i,i],color='#BBBBBB',lw=1.3)
  axs[1,0].scatter(p.value,i,c=colors[s],marker='o',s=38)
  axs[1,0].scatter(p.observed,i,c='#222222',marker='D',s=28)
  it=intervals.loc[intervals.target.eq('Saudi Arabia')&intervals.outcome.eq(o)&intervals.sex.eq(s)
       &intervals.procedure.eq('direct')&intervals.family.eq('tcn_adapted')&intervals.horizon.eq(5)
       &intervals.population_method.eq('log_trend_last8')&intervals.node.eq(s+'__80+_within_45+')
       &intervals.level.eq(.8)].iloc[0]
  axs[1,1].scatter(100*it.coverage,i,c=colors[s],s=50)
  axs[1,1].text(100*it.coverage+3,i,f'{round(5*it.coverage)}/5',va='center',fontsize=10)
  exported.append(dict(outcome=o,sex=s,local_error=a.local_champion,nonneural_error=a.nonneural_champion,
      tcn_error=a.tcn_adapted,rate_80_coverage_percent=100*r.coverage,
      forecast_80plus_share=p.value,gbd_80plus_share=p.observed,share_80_coverage_percent=100*it.coverage))
 for ax in axs.flat:
  ax.set_yticks(range(8),[names[o]+' · '+s for o,s in keys],fontsize=10)
  ax.set_ylim(7.6,-.6)
 panel(axs[0,0],'A  Endpoint rate accuracy','Mean absolute log error')
 axs[0,0].set_xlim(left=0)
 panel(axs[0,1],'B  Rate interval coverage','Empirical coverage of nominal 80% intervals (%)')
 panel(axs[1,0],'C  Burden at ages 80+','80+ share within sex-specific 45+ burden (%)')
 axs[1,0].set_xlim(0,57)
 panel(axs[1,1],'D  Age-share interval coverage','Empirical coverage of nominal 80% intervals (%)')
 for ax in [axs[0,1],axs[1,1]]:
  ax.axvline(80,color='#555555',ls=':',lw=1.3);ax.set_xlim(-2,105);ax.set_xticks([0,20,40,60,80,100])
 axs[0,0].legend(handles=[Line2D([],[],ls='',marker=m,color=c,label=l) for m,c,l in
      [('s','#777777','Local'),('^','#009E73','Non-neural'),('o','#222222','TCN')]],
      loc='upper right',frameon=False,fontsize=9)
 axs[1,0].legend(handles=[Line2D([],[],ls='',marker=m,color=c,label=l) for m,c,l in
      [('o','#0072B2','Male forecast'),('o','#C55300','Female forecast'),('D','#222222','GBD estimate')]],
      loc='upper right',frameon=False,fontsize=9)
 fig.text(.08,.035,'YLDs: years lived with disability. YLLs: years of life lost. DALYs: disability-adjusted life-years.',fontsize=10,color='#536371')
 pd.DataFrame(exported).to_csv(OUT/'analysis/disability_reporting_summary.csv',index=False)
 save(fig,5)


def decomposition():
 d=pd.read_csv(OUT/'analysis/forecast_growth_decomposition.csv')
 d=d.loc[d.year.eq(2028)&d.scenario.eq('gbd_2023_aligned_un_growth')]
 components=[('population_size','#0072B2','Population size\n(45+)'),('composition','#E69F00','Age / age–sex\ncomposition'),('rates','#009E73','Age-specific\ndisease rates')]
 fig,axs=plt.subplots(1,2,figsize=(12.6,9.5))
 fig.subplots_adjust(left=.14,right=.97,top=.70,bottom=.30,wspace=.38)
 fig.text(.07,.963,'What contributes to the projected increase?',fontsize=22,weight='bold')
 fig.text(.07,.911,'Saudi Arabia · 2023–2028 · GBD baseline + UN growth\nExploratory arithmetic decomposition',fontsize=16,color='#536371',va='top')
 fig.legend(handles=[plt.Rectangle((0,0),1,1,color=c,label=l) for _,c,l in components],
            loc='upper left',bbox_to_anchor=(.066,.818),ncol=3,frameon=False,fontsize=16)
 for j,o in enumerate(['prevalence','incidence']):
  ax=axs[j]
  for i,s in enumerate(['Male','Female','Both']):
   r=d.loc[d.outcome.eq(o)&d.sex.eq(s)].iloc[0]
   positive=0.;negative=0.
   for key,color,_ in components:
    value=r[key+'_percentage_point_contribution'];left=positive if value>=0 else negative
    ax.barh(i,value,left=left,color=color,height=.53,hatch='///' if value<0 else None,edgecolor='white',lw=.5)
    if value>=0:positive+=value
    else:negative+=value
    if abs(value)>=3:
     label_y=i
     label_x=left+value/2
     label_color='white' if key!='composition' else '#222222'
     label_align='center'
     if key=='composition' and abs(value)<4:
      label_y=i-.40
      ax.plot([left+value/2]*2,[i-.275,i-.32],color='#222222',lw=.7)
     if key=='rates' and 0<value<6:
      label_x=left+value+.6;label_color='#007C5D';label_align='left'
     ax.text(label_x,label_y,f'{value:+.1f}',ha=label_align,va='center',color=label_color,fontsize=16,weight='bold')
   ax.plot(r.total_change_percent,i+.35,'D',ms=5,color='#222222')
   ax.text(r.total_change_percent,i+.56,f'Total +{r.total_change_percent:.2f}%',ha='center',va='center',fontsize=16)
  ax.axvline(0,color='#555555',lw=.8);ax.set_xlim(-4,48);ax.set_ylim(2.95,-.7)
  ax.set_yticks(range(3),['Male','Female','Both sexes'])
  panel(ax,f'{"AB"[j]}  {o.capitalize()}','Contribution to change from 2023\n(percentage points)')
 fig.text(.07,.157,'Contributions average all six replacement orders and sum exactly\nto the percentage change in case counts.',fontsize=16,color='#536371',va='top')
 fig.text(.07,.082,'Female prevalence: +26.86 points from population size,\n+2.15 from age composition and −1.59 from rates.',fontsize=16,color='#536371',va='top')
 save(fig,8)


def main(numbers=None):
 global SELECTED
 SELECTED=set(numbers or range(1,9))
 FIG.mkdir(parents=True,exist_ok=True)
 TYPOGRAPHY.clear()
 if (OUT/'figure_typography.json').exists():
  TYPOGRAPHY.extend(item for item in json.loads((OUT/'figure_typography.json').read_text())
                    if item['figure'] not in SELECTED)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'font.weight':'bold',
  'axes.labelweight':'bold','axes.titleweight':'bold','pdf.fonttype':42,'svg.fonttype':'none'})
 old=load('build_ije_manuscript');old.FIG=FIG;old.save_figure=save
 paths=['results/primary_v1/primary_age_contrasts.csv','reports/secondary_v1/endpoint_contrasts.csv',
 'reports/donor_comparisons_gpu_v1/endpoint_percent_changes_vs_gpu_reference.csv','results/learning_curves_v1/summary.csv',
 'reports/distribution_mixture_v1/rate_comparison.csv','reports/distribution_mixture_v1/burden_comparison.csv',
 'reports/population_sensitivity_v1/saudi_population_source_comparison.csv','reports/population_sensitivity_v1/saudi_80plus_share_endpoint.csv']
 old.figures(*[pd.read_csv(ROOT/p) for p in paths])
 disability()
 forecast=load('plot_standalone_forecast_comparison');forecast.setup_style()
 counts=pd.read_csv(ROOT/'forecast.csv')
 counts=counts.loc[counts.record_type.eq('current_study_count_forecast')]
 # The original export label is read explicitly to prevent silent empty plots.
 if counts.empty:
  all_rows=pd.read_csv(ROOT/'forecast.csv')
  counts=all_rows.loc[all_rows.metric.eq('count_45plus')]
 if counts.empty:
  all_rows=pd.read_csv(ROOT/'forecast.csv');print(all_rows.record_type.unique(),all_rows.metric.unique());raise ValueError('Count rows not found')
 counts=counts.loc[~counts.population_scenario.str.startswith('national',na=False)|counts.within_80plus_allocation.eq('gbd_aligned_within_80plus')]
 fig=plt.figure(figsize=(12.6,8.5))
 fig.text(.068,.96,'Projected Parkinson’s disease case counts, 2024–2028',fontsize=18,weight='bold')
 fig.text(.068,.921,'Saudi Arabia · ages 45+ · adapted TCN · alternative population scenarios',fontsize=11,color='#536371')
 forecast.forecast_panels(fig,counts,top=.79,bottom=.13)
 fig.text(.5,.069,'Year',ha='center',fontsize=11)
 fig.text(.068,.028,'National scenario: 2024 anchor advanced with UN growth; GBD-aligned allocation within ages 80+.',fontsize=9.5,color='#536371')
 save(fig,6)
 change=load('plot_forecast_percentage_change');change.setup_style();change.export=lambda fig,stem:save(fig,7)
 change.main_figure(pd.read_csv(ROOT/'forecast_percentage_change.csv'))
 decomposition()
 thumbs=[]
 for i in range(1,9):
  img=Image.open(FIG/f'figure_{i}_preview.png').convert('RGB');img.thumbnail((650,650))
  canvas=Image.new('RGB',(690,700),'white');canvas.paste(img,((690-img.width)//2,35))
  ImageDraw.Draw(canvas).text((20,10),f'Figure {i}',fill='black');thumbs.append(canvas)
 sheet=Image.new('RGB',(1380,2800),'#EEEEEE')
 for i,img in enumerate(thumbs):sheet.paste(img,((i%2)*690,(i//2)*700))
 sheet.save(OUT/'figure_contact_sheet.png')
 TYPOGRAPHY.sort(key=lambda item:item['figure'])
 (OUT/'figure_typography.json').write_text(json.dumps(TYPOGRAPHY,indent=2)+'\n')
 print(f'Figures {sorted(SELECTED)} exported as vector PDF/SVG, 300 dpi PNG and 600 dpi TIFF.')


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--figures',type=int,nargs='+',choices=range(1,9),help='Regenerate only selected figures.')
 main(parser.parse_args().figures)
