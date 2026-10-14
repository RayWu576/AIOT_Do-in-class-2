(() => {
'use strict';
const D=window.DASHBOARD_DATA,$=id=>document.getElementById(id);
if(!D){document.querySelector('main').innerHTML='<section class="panel"><h2>尚未產生資料</h2><p>請先執行 python main.py，再重新開啟網站。</p></section>';return;}
const methods=['Linear Regression','Ridge','Lasso','Elastic Net','Random Forest'],colors=['#5585ef','#09ad87','#e7a43c','#9c68db','#e16a87'];
const desc=['一般最小平方法','L2 正則化迴歸','L1 正則化迴歸','L1 + L2 正則化','非線性決策樹集成'],icons=['↗','≋','λ','⋈','⌘'];
const names={'R&D Spend':'研發投入','Marketing Spend':'行銷投入','Administration':'行政支出','State_Florida':'Florida','State_New York':'New York'},fields=['R&D Spend','Marketing Spend','Administration'];
const num=(v,d=4)=>Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}),money=v=>'$'+num(v,2),pv=v=>v<.0001?'< 0.0001':num(v);
const recommended=D.comparison.find(r=>r.Model===D.recommended);
const s={method:D.recommended,k:recommended.Features,compare:true,values:{'R&D Spend':100000,'Marketing Spend':150000,'Administration':85000},location:'California',sort:'abs',scenarios:[]};
const model=()=>D.models.find(m=>m.method===s.method&&m.k===s.k),best=Math.min(...D.models.map(m=>m.rmse)),isBest=m=>Math.abs(m.rmse-best)<1e-6;
const storage={get:k=>{try{return localStorage.getItem(k)}catch{return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch{}}};
if(storage.get('startup-theme')==='dark')document.body.classList.add('dark');
$('theme').onclick=()=>{document.body.classList.toggle('dark');storage.set('startup-theme',document.body.classList.contains('dark')?'dark':'light');};
function toast(t){$('toast').textContent=t;$('toast').classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.remove('show'),2500);}
function choose(method,k){s.method=method;s.k=k;render();}
function controls(){
 $('methods').innerHTML=methods.map((m,i)=>`<button class="method-button ${m===s.method?'active':''}" data-method="${m}" aria-pressed="${m===s.method}"><span class="method-icon">${icons[i]}</span><strong>${m}</strong><small>${desc[i]}</small></button>`).join('');
 $('methods').querySelectorAll('button').forEach(b=>b.onclick=()=>choose(b.dataset.method,s.k));
 $('counts').innerHTML=[1,2,3,4,5].map(k=>`<button data-k="${k}" aria-pressed="${k===s.k}" class="${k===s.k?'active':''}">k = ${k}${isBest(D.models.find(m=>m.method===s.method&&m.k===k))?'<small>★</small>':''}</button>`).join('');
 $('counts').querySelectorAll('button').forEach(b=>b.onclick=()=>choose(s.method,+b.dataset.k));
 const notes=['一般線性迴歸，直接以最小平方法擬合與預測。','標準化後調整 L2 正則化 alpha；直接以 Ridge 模型預測。','標準化後調整 L1 正則化 alpha；直接以 Lasso 模型預測，部分係數可為零。','標準化後同時調整 alpha 與 l1_ratio；直接使用 Elastic Net 預測。','以 60 棵決策樹的平均值預測；訓練內交叉驗證選擇樹深度，葉節點至少 2 筆。'];
 $('method-note').textContent=notes[methods.indexOf(s.method)];$('compare').checked=s.compare;
}
function metrics(m){
 const base=D.models.find(x=>x.method===s.method&&x.k===5),delta=(base.rmse-m.rmse)/base.rmse*100;
 const items=[['TEST RMSE',money(m.rmse),isBest(m)?'★ 固定 k 比較中的最低測試誤差':'越低越好 · 保留測試集','↘'],['R² SCORE',num(m.r2),`測試集解釋力 · ${num(m.r2*100,2)}%`,'↗'],['MEAN ABSOLUTE ERROR',money(m.mae),'測試集平均絕對誤差','±'],['TUNING CV RMSE',money(m.cvRmse),'訓練內 4 折調參分數 · 非無偏估計','⌘'],['TRAIN RMSE',money(m.trainRmse),'與測試誤差對照觀察泛化落差','ƒ']];
 $('metrics').innerHTML=items.map(([label,v,note,icon])=>`<article class="metric"><div class="metric-label">${label}<span class="metric-symbol">${icon}</span></div><strong>${v}</strong><p>${note}</p></article>`).join('');
 $('active-features').innerHTML=m.features.map(f=>`<span class="feature-chip">${f}</span>`).join('');
 $('insight').textContent=s.k===5?'完整 5 特徵模型 · 與精簡模型比較後再判斷':`相較完整模型，測試 RMSE ${delta>=0?'降低':'增加'} ${num(Math.abs(delta),1)}%`;
 $('equation').innerHTML=m.kind==='forest'?`Random Forest = <b>${m.trees.length} 棵決策樹預測的平均值</b> · 無單一線性公式`:`Profit = <b>${num(m.intercept,2)}</b> `+m.features.map((f,i)=>`${m.weights[i]>=0?'+':'−'} <b>${num(Math.abs(m.weights[i]),5)}</b> × (${f})`).join(' ');
}
function chart(id,metric){
 const visible=s.compare?methods:[s.method],data=D.models.filter(m=>visible.includes(m.method)),vals=data.map(m=>m[metric]);
 const min=Math.min(...vals),max=Math.max(...vals),pad=(max-min)*.16||1,lo=min-pad,hi=max+pad,W=560,H=265,L=55,R=15,T=15,B=36,x=k=>L+(k-1)*(W-L-R)/4,y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${metric} 隨特徵數變化">`;
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;svg+=`<line class="grid-line" x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"/><text x="${L-8}" y="${y(v)+3}" text-anchor="end">${metric==='rmse'?num(v,0):num(v,3)}</text>`;}
 for(let k=1;k<=5;k++)svg+=`<line class="grid-line" x1="${x(k)}" y1="${T}" x2="${x(k)}" y2="${H-B}" opacity=".5"/><text x="${x(k)}" y="${H-15}" text-anchor="middle">${k}</text>`;
 svg+=`<line x1="${x(s.k)}" y1="${T}" x2="${x(s.k)}" y2="${H-B}" stroke="var(--purple)" opacity=".25" stroke-dasharray="4 4"/>`;
 visible.filter(v=>v!==s.method).concat(s.method).forEach(method=>{
  const points=D.models.filter(m=>m.method===method),c=colors[methods.indexOf(method)],active=method===s.method;
  svg+=`<path d="${points.map((m,i)=>`${i?'L':'M'}${x(m.k)} ${y(m[metric])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${active?3:1.8}" opacity="1"/>`;
  points.forEach(m=>{const label=`${method} · k=${m.k} · ${metric==='rmse'?money(m.rmse):num(m.r2)}`;svg+=`<circle cx="${x(m.k)}" cy="${y(m[metric])}" r="${m.k===s.k&&active?6:3.5}" fill="${c}" stroke="var(--surface)" stroke-width="2" tabindex="0" role="button" data-method="${method}" data-k="${m.k}" aria-label="${label}"><title>${label}</title></circle>`;});
 });
 svg+=`<text x="${(W+L-R)/2}" y="${H-1}" text-anchor="middle" style="font-size:9px">Number of features (k)</text></svg>`;
 $(id).innerHTML=`<div class="legend">${visible.map(m=>`<button data-method="${m}" class="${m===s.method?'selected':''}"><i style="background:${colors[methods.indexOf(m)]}"></i>${m}</button>`).join('')}</div>`+svg+'<div class="chart-tooltip">點選圖例或資料點，切換目前模型</div>';
 $(id).querySelectorAll('.legend button').forEach(b=>b.onclick=()=>choose(b.dataset.method,s.k));
 $(id).querySelectorAll('circle').forEach(dot=>{const activate=()=>choose(dot.dataset.method,+dot.dataset.k);dot.onclick=activate;dot.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};dot.onmouseenter=()=>{$(id).querySelector('.chart-tooltip').textContent=dot.getAttribute('aria-label');};});
}
function coefficients(m){
 const forest=m.kind==='forest',table=$('coefficients').closest('table');
 $('stats-title').textContent=forest?'Random Forest 特徵重要性':'模型係數與可用統計';
 $('stats-subtitle').textContent=forest?'以樹的 impurity reduction 計算；不是因果效果或顯著性檢定':'原始金額單位係數 · 僅 Linear Regression 提供 OLS 推論';
 $('stats-note').textContent=forest?'重要性總和為 1，反映樹分裂的相對貢獻；資料只有 40 筆訓練樣本，重要性可能不穩定。':m.method==='Linear Regression'?'OLS 的 p 值與信賴區間未校正特徵篩選效應，僅供探索。VIF ≥ 5 提示共線性。':'正則化模型的 OLS 標準誤、t 值、p 值及信賴區間不適用，因此以 — 顯示。VIF 描述輸入的共線性。';
 table.querySelector('thead').innerHTML=forest?'<tr><th>預測變數</th><th>重要性</th><th>比例</th></tr>':'<tr><th>預測變數</th><th>係數 β</th><th>標準誤</th><th>t 值</th><th>p 值</th><th>95% CI</th><th>VIF</th></tr>';
 const safe=(v,d=4)=>v===null?'—':num(v,d);
 $('coefficients').innerHTML=forest?m.features.map((f,i)=>`<tr><td>${f}</td><td>${num(m.importances[i])}</td><td>${num(m.importances[i]*100,2)}%</td></tr>`).join(''):m.coefficients.map(c=>`<tr><td>${c.name}</td><td><strong>${num(c.estimate)}</strong></td><td>${safe(c.se)}</td><td>${safe(c.t,2)}</td><td>${c.p===null?'—':`<span class="pill ${c.p<.05?'green-pill':'purple-pill'}">${pv(c.p)}</span>`}</td><td>${c.low===null?'—':`[${num(c.low,3)}, ${num(c.high,3)}]`}</td><td>${safe(c.vif,2)}</td></tr>`).join('');
 const effects=forest?m.importances:m.features.map((f,i)=>Math.abs(m.weights[i]*(f.startsWith('State_')?1:Math.max(1,D.ranges[f].max-D.ranges[f].min)))),largest=Math.max(...effects,1e-12);
 $('coefficient-bars').innerHTML=`<div class="eyebrow purple">${forest?'FEATURE IMPORTANCE · 樹模型相對重要性':'FEATURE EFFECT · 訓練範圍內的利潤變化幅度'}</div>`+m.features.map((f,i)=>`<div class="bar-row"><span>${f}</span><div class="bar-track"><div class="bar-fill" style="width:${effects[i]/largest*100}%"></div></div><span>${forest?num(effects[i]*100,1)+'%':num(effects[i]/1000,1)+'k'}</span></div>`).join('');
 $('parameters').textContent='訓練內交叉驗證參數：'+JSON.stringify(m.parameters);
}
function inputs(m){
 $('inputs').innerHTML=fields.map((f,i)=>`<div class="${m.features.includes(f)?'':'inactive'}"><label class="field-label" for="number-${i}">${names[f]}<span>${m.features.includes(f)?'使用中':'未納入目前模型'}</span></label><div class="input-pair"><input id="number-${i}" type="number" min="0" max="1000000000" step="1000" value="${s.values[f]}" aria-label="${names[f]}金額"><input id="range-${i}" type="range" min="0" max="${Math.ceil(Math.max(D.ranges[f].max,s.values[f],100000)/50000)*50000}" step="1000" value="${s.values[f]}" aria-label="${names[f]}滑桿"></div></div>`).join('');
 fields.forEach((f,i)=>{const n=$('number-'+i),r=$('range-'+i);n.oninput=()=>{if(n.value===''||!Number.isFinite(n.valueAsNumber)||n.valueAsNumber<0||n.valueAsNumber>1e9)return;s.values[f]=n.valueAsNumber;r.max=Math.max(+r.max,s.values[f]);r.value=s.values[f];predict();};n.onchange=()=>{n.value=s.values[f];};r.oninput=()=>{s.values[f]=+r.value;n.value=r.value;predict();};});
 $('state').value=s.location;$('state-status').textContent=m.features.some(f=>f.startsWith('State_'))?'地區特徵使用中':'地區未納入目前模型';predict();
}
// Exact browser inference for the fitted Python models, including RF trees.
window.startupPredict=(m,v)=>{
 if(m.kind!=='forest')return m.intercept+m.weights.reduce((total,w,i)=>total+w*v[m.features[i]],0);
 const values=m.features.map(f=>Math.fround(v[f]));
 return m.trees.reduce((sum,t)=>{let node=0;while(t.left[node]!==-1)node=values[t.feature[node]]<=t.threshold[node]?t.left[node]:t.right[node];return sum+t.value[node];},0)/m.trees.length;
};
function predict(){
 const m=model(),v={...s.values,'State_Florida':+(s.location==='Florida'),'State_New York':+(s.location==='New York')},profit=window.startupPredict(m,v);
 $('profit').textContent=money(profit);
 $('contributions').textContent=m.kind==='forest'?`${m.trees.length} 棵決策樹平均預測 · 輸入 ${m.features.join('、')}`:m.features.map((f,i)=>`${names[f]||f} ${m.weights[i]*v[f]>=0?'+':'−'}${money(Math.abs(m.weights[i]*v[f]))}`).join(' · ');
 const outside=m.features.filter(f=>fields.includes(f)&&(v[f]<D.ranges[f].min||v[f]>D.ranges[f].max));
 $('extrapolation').textContent=outside.length?`提示：${outside.map(f=>names[f]).join('、')}超出訓練資料範圍，屬於外插情境。`:'直接使用目前模型預測；零係數或未納入的特徵不影響結果。預測不保證實際收益。';return profit;
}
function scenarios(){
 $('scenarios').innerHTML=s.scenarios.map((v,i)=>`<div class="scenario"><span>${i+1}. ${v.method} · k=${v.k}<br><span class="muted">${fields.map(f=>`${names[f]} ${num(v.values[f],0)}`).join(' / ')} · ${v.location}</span></span><strong>${money(v.profit)}</strong><button data-index="${i}" aria-label="移除情境 ${i+1}">×</button></div>`).join('');
 $('scenarios').querySelectorAll('button').forEach(b=>b.onclick=()=>{s.scenarios.splice(+b.dataset.index,1);scenarios();});
}
function residuals(m){
 const rows=[...m.residuals].sort((a,b)=>s.sort==='id'?a.id-b.id:s.sort==='percent'?b.percent-a.percent:Math.abs(b.residual)-Math.abs(a.residual));
 $('residuals').innerHTML=rows.map(r=>`<tr><td>Startup #${r.id}</td><td>${money(r.actual)}</td><td>${money(r.predicted)}</td><td class="${r.residual>=0?'positive':'negative'}">${r.residual>=0?'+':'−'}${money(Math.abs(r.residual))}</td><td>${money(Math.abs(r.residual))}</td><td>${num(r.percent,2)}%</td><td><span class="pill ${Math.abs(r.residual)>m.rmse?'purple-pill':'green-pill'}">${Math.abs(r.residual)>m.rmse?'高殘差':'範圍內'}</span></td></tr>`).join('');
 const W=900,H=190,L=68,R=24,T=20,B=28,all=m.residuals.flatMap(r=>[r.actual,r.predicted]),lo=Math.min(...all)*.9,hi=Math.max(...all)*1.05,x=v=>L+(v-lo)/(hi-lo)*(W-L-R),y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="真實與預測利潤散佈圖，虛線為完美預測"><line x1="${x(lo)}" y1="${y(lo)}" x2="${x(hi)}" y2="${y(hi)}" stroke="var(--muted)" opacity=".35" stroke-dasharray="4 4"/>`;
 for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;svg+=`<line class="grid-line" x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"/><text x="${L-8}" y="${y(v)+3}" text-anchor="end">${num(v/1000,0)}k</text><text x="${x(v)}" y="${H-8}" text-anchor="middle">${num(v/1000,0)}k</text>`;}
 svg+=m.residuals.map(r=>`<circle cx="${x(r.actual)}" cy="${y(r.predicted)}" r="5" fill="${Math.abs(r.residual)>m.rmse?'#9c68db':'#09ad87'}" opacity=".85"><title>Startup #${r.id} · 真實 ${money(r.actual)} · 預測 ${money(r.predicted)}</title></circle>`).join('');
 $('scatter').innerHTML=svg+`<text x="${W-R}" y="${T}" text-anchor="end">X 真實利潤 / Y 預測利潤 · 虛線 = 完美預測</text></svg>`;
}
function render(){const m=model();controls();metrics(m);chart('rmse-chart','rmse');chart('r2-chart','r2');coefficients(m);inputs(m);residuals(m);}
$('compare').onchange=e=>{s.compare=e.target.checked;chart('rmse-chart','rmse');chart('r2-chart','r2');};$('state').onchange=e=>{s.location=e.target.value;predict();};$('sort').onchange=e=>{s.sort=e.target.value;residuals(model());};
$('reset').onclick=()=>{Object.assign(s,{method:D.recommended,k:recommended.Features,compare:true,location:'California',sort:'abs',values:{'R&D Spend':100000,'Marketing Spend':150000,'Administration':85000},scenarios:[]});$('sort').value='abs';render();scenarios();toast('已重設模型、輸入與情境');};
$('save-scenario').onclick=()=>{if(s.scenarios.length>=5){toast('最多保留 5 個情境，請先移除一個');return;}s.scenarios.push({method:s.method,k:s.k,values:{...s.values},location:s.location,profit:predict()});scenarios();toast('已加入情境比較');};
$('download').onclick=()=>{const m=model(),q=v=>'"'+String(v).replaceAll('"','""')+'"',rows=[['Method','Feature count','Test RMSE','Test R2','Adjusted R2','MAE','Selected features'],[m.method,m.k,m.rmse,m.r2,m.adjustedR2,m.mae,m.features.join(', ')],[],['Feature','Forest importance'],...m.features.map((f,i)=>[f,m.importances[i]??'']),[],['Predictor','Estimate','SE','t','p','CI low','CI high','VIF'],...m.coefficients.map(c=>[c.name,c.estimate,c.se,c.t,c.p,c.low,c.high,c.vif??'']),[],['Sample ID','Actual profit','Predicted profit','Residual','Error %'],...m.residuals.map(r=>[r.id,r.actual,r.predicted,r.residual,r.percent])],url=URL.createObjectURL(new Blob(['\ufeff'+rows.map(r=>r.map(q).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`startup-${m.method.replaceAll(' ','-')}-k${m.k}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已匯出目前模型與測試結果');};
$('ranking').innerHTML=D.comparison.map(r=>`<tr><td><span class="feature-chip">${r.Rank===1?'★ ':''}${r.Rank}</span></td><td>${r.Model}</td><td>${r.Features}</td><td><strong>${money(r['CV RMSE'])}</strong></td><td>± ${num(r['CV Std'],2)}</td><td>${money(r['Test RMSE'])}</td><td>${num(r['Test R2'])}</td></tr>`).join('');
$('split-tag').textContent=`${D.trainSize} / ${D.testSize} 訓練與測試`;$('generated').textContent='資料產生時間：'+new Date(D.generatedAt).toLocaleString('zh-TW',{timeZone:'Asia/Taipei'})+'（台灣時間）。重新執行 main.py 後，重新整理頁面即可載入更新數據。';render();
})();
