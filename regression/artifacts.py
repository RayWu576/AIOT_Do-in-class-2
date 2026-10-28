"""Render diagnostics without assuming a dataset or meaningful time order."""
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def diagnostic_png(model,kind):
    diagnostics=model['diagnostics']
    fig,ax=plt.subplots(figsize=(7,4.5))
    if kind=='linearity':
        d=diagnostics['linearity'];ax.scatter(d['fitted'],d['residuals'],s=13,alpha=.6)
        ax.axhline(0,color='#6551e8',linestyle='--');ax.set(xlabel='Training fitted value',ylabel='Residual',title='Linearity: residuals vs fitted')
    elif kind=='qq':
        points=diagnostics['qq'];x=[p['theoretical'] for p in points];y=[p['observed'] for p in points]
        ax.scatter(x,y,s=13,alpha=.6)
        ax.set(xlabel='Normal theoretical quantile',ylabel='Training residual quantile',title='Normality: Q-Q plot')
    else:raise ValueError('Unknown diagnostic plot')
    fig.tight_layout();stream=BytesIO();fig.savefig(stream,format='png',dpi=150);plt.close(fig)
    return stream.getvalue()
