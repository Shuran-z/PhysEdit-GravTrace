"""Render saved development gravity profiles without rerunning fits."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    data=json.load(open('docs/phyediting/gravity_profile15.json'))
    rows=data['records']
    fig,axes=plt.subplots(5,4,figsize=(13,13),constrained_layout=True)
    for ax,r in zip(axes.flat,rows):
        g=[c['gravity']/r['saved_gravity'] for c in r['curve']]
        minimum=min(c['cost'] for c in r['curve'])
        delta=[c['cost']-minimum for c in r['curve']]
        ax.plot(g,delta,color='#2667a8',lw=1.2)
        ax.axvline(1,color='#2667a8',ls='--',lw=1,label='Saved estimate')
        ax.axvline(r['gravity_target']/r['saved_gravity'],color='#bd442d',lw=1,label='Target (diagnosis only)')
        ax.set_title(r['id'].split('__')[0]+f" | {r['saved_error_pct']:.1f}% error",fontsize=9)
        ax.set_xlabel('g / saved g',fontsize=8);ax.set_ylabel('Cost minus grid minimum',fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(rows):]:ax.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower right',fontsize=9)
    fig.suptitle('15 fps development diagnosis: 14 error-selected cases + 5 fixed-hash controls\nFixed g, reoptimized unknown velocity; grid relative to saved estimate',fontsize=12)
    fig.savefig('docs/phyediting/gravity_profile15.svg')
    fig.savefig('/tmp/gravtrace-profile15.png',dpi=100)

if __name__=='__main__':main()
