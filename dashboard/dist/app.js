(() => {
'use strict';
const D=window.DASHBOARD_DATA,$=id=>document.getElementById(id);
if(!D){document.querySelector('main').innerHTML='<section class="panel"><h2>尚未產生資料</h2><p>請先執行 python main.py，再重新開啟網站。</p></section>';return;}
const methods=['Correlation','Mutual Information','RFE','RFECV','LASSO'],colors=['#5585ef','#09ad87','#e7a43c','#9c68db','#e16a87'];
const desc=['Pearson 相關係數排名','互資訊 · 非線性關聯','遞迴消除特徵','交叉驗證選擇數量','L1 正則化 · 係數排名'],icons=['↗','⋈','✂','⌘','λ'];
const names={'R&D Spend':'研發投入','Marketing Spend':'行銷投入','Administration':'行政支出','State_Florida':'Florida','State_New York':'New York'},fields=['R&D Spend','Marketing Spend','Administration'];
const num=(v,d=4)=>Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}),money=v=>'$'+num(v,2),pv=v=>v<.0001?'< 0.0001':num(v);
const s={method:'LASSO',k:2,compare:true,values:{'R&D Spend':100000,'Marketing Spend':150000,'Administration':85000},location:'California',sort:'abs',scenarios:[]};
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
 const notes=['依訓練資料的絕對 Pearson 相關係數選取前 k 個特徵。','依訓練資料的互資訊選取前 k 個特徵；State 虛擬變數按離散特徵處理。','標準化後，依線性迴歸係數遞迴移除特徵，保留 k 個。','固定 k 曲線使用 RFECV 的 RFE 消除路徑；自動選取結果請見下方完整工作流程比較。','固定 k 時以標準化 LassoCV 係數絕對值排名；零係數同分按原欄位順序。自動選取結果另列於下方。'];
 $('method-note').textContent=notes[methods.indexOf(s.method)];$('compare').checked=s.compare;
}
function metrics(m){
 const base=D.models.find(x=>x.method===s.method&&x.k===5),delta=(base.rmse-m.rmse)/base.rmse*100;
 const items=[['TEST RMSE',money(m.rmse),isBest(m)?'★ 固定 k 比較中的最低誤差':'越低越好 · 保留測試集','↘'],['R² SCORE',num(m.r2),`解釋 ${num(m.r2*100,2)}% 的測試集變異`,'↗'],['ADJUSTED R²',num(m.adjustedR2),'測試集 · 校正特徵數量','≋'],['MEAN ABSOLUTE ERROR',money(m.mae),'測試集平均絕對誤差','±'],['F-STATISTIC',num(m.f,1),`訓練集 OLS · p ${pv(m.fP)}`,'ƒ']];
 $('metrics').innerHTML=items.map(([label,v,note,icon])=>`<article class="metric"><div class="metric-label">${label}<span class="metric-symbol">${icon}</span></div><strong>${v}</strong><p>${note}</p></article>`).join('');
 $('active-features').innerHTML=m.features.map(f=>`<span class="feature-chip">${f}</span>`).join('');
 $('insight').textContent=s.k===5?'完整 5 特徵模型 · 與精簡模型比較後再判斷':`相較完整模型，測試 RMSE ${delta>=0?'降低':'增加'} ${num(Math.abs(delta),1)}%`;
 $('equation').innerHTML=`Profit = <b>${num(m.intercept,2)}</b> `+m.features.map((f,i)=>`${m.weights[i]>=0?'+':'−'} <b>${num(Math.abs(m.weights[i]),5)}</b> × (${f})`).join(' ');
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
  svg+=`<path d="${points.map((m,i)=>`${i?'L':'M'}${x(m.k)} ${y(m[metric])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${active?3:1.8}" opacity="${active?1:.65}" ${active?'':'stroke-dasharray="5 3"'}/>`;
  points.forEach(m=>{const label=`${method} · k=${m.k} · ${metric==='rmse'?money(m.rmse):num(m.r2)}`;svg+=`<circle cx="${x(m.k)}" cy="${y(m[metric])}" r="${m.k===s.k&&active?6:3.5}" fill="${c}" stroke="var(--surface)" stroke-width="2" tabindex="0" role="button" data-method="${method}" data-k="${m.k}" aria-label="${label}"><title>${label}</title></circle>`;});
 });
 svg+=`<text x="${(W+L-R)/2}" y="${H-1}" text-anchor="middle" style="font-size:9px">Number of features (k)</text></svg>`;
 $(id).innerHTML=`<div class="legend">${visible.map(m=>`<button data-method="${m}" class="${m===s.method?'selected':''}"><i style="background:${colors[methods.indexOf(m)]}"></i>${m}</button>`).join('')}</div>`+svg+'<div class="chart-tooltip">點選圖例或資料點，切換目前模型</div>';
 $(id).querySelectorAll('.legend button').forEach(b=>b.onclick=()=>choose(b.dataset.method,s.k));
 $(id).querySelectorAll('circle').forEach(dot=>{const activate=()=>choose(dot.dataset.method,+dot.dataset.k);dot.onclick=activate;dot.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};dot.onmouseenter=()=>{$(id).querySelector('.chart-tooltip').textContent=dot.getAttribute('aria-label');};});
}
function coefficients(m){
 $('coefficients').innerHTML=m.coefficients.map(c=>`<tr><td>${c.name}</td><td><strong>${num(c.estimate)}</strong></td><td>${num(c.se)}</td><td>${num(c.t,2)}</td><td><span class="pill ${c.p<.05?'green-pill':'purple-pill'}">${pv(c.p)}</span></td><td>[${num(c.low,3)}, ${num(c.high,3)}]</td><td>${c.vif===null?'—':`<span class="${c.vif>=5?'negative':'positive'}">${num(c.vif,2)}</span>`}</td></tr>`).join('');
 const effects=m.features.map((f,i)=>Math.abs(m.weights[i]*(f.startsWith('State_')?1:Math.max(1,D.ranges[f].max-D.ranges[f].min)))),largest=Math.max(...effects,1);
 $('coefficient-bars').innerHTML='<div class="eyebrow purple">FEATURE EFFECT · 訓練範圍內的利潤變化幅度</div>'+m.features.map((f,i)=>`<div class="bar-row"><span>${f}</span><div class="bar-track"><div class="bar-fill" style="width:${effects[i]/largest*100}%"></div></div><span>${num(effects[i]/1000,1)}k</span></div>`).join('');
}
function inputs(m){
 $('inputs').innerHTML=fields.map((f,i)=>`<div class="${m.features.includes(f)?'':'inactive'}"><label class="field-label" for="number-${i}">${names[f]}<span>${m.features.includes(f)?'使用中':'未納入目前模型'}</span></label><div class="input-pair"><input id="number-${i}" type="number" min="0" max="1000000000" step="1000" value="${s.values[f]}" aria-label="${names[f]}金額"><input id="range-${i}" type="range" min="0" max="${Math.ceil(Math.max(D.ranges[f].max,s.values[f],100000)/50000)*50000}" step="1000" value="${s.values[f]}" aria-label="${names[f]}滑桿"></div></div>`).join('');
 fields.forEach((f,i)=>{const n=$('number-'+i),r=$('range-'+i);n.oninput=()=>{if(n.value===''||!Number.isFinite(n.valueAsNumber)||n.valueAsNumber<0||n.valueAsNumber>1e9)return;s.values[f]=n.valueAsNumber;r.max=Math.max(+r.max,s.values[f]);r.value=s.values[f];predict();};n.onchange=()=>{n.value=s.values[f];};r.oninput=()=>{s.values[f]=+r.value;n.value=r.value;predict();};});
 $('state').value=s.location;$('state-status').textContent=m.features.some(f=>f.startsWith('State_'))?'地區特徵使用中':'地區未納入目前模型';predict();
}
function predict(){
 const m=model(),v={...s.values,'State_Florida':+(s.location==='Florida'),'State_New York':+(s.location==='New York')},parts=m.weights.map((w,i)=>w*v[m.features[i]]),profit=m.intercept+parts.reduce((a,b)=>a+b,0);
 $('profit').textContent=money(profit);$('contributions').textContent=m.features.map((f,i)=>`${names[f]||f} ${parts[i]>=0?'+':'−'}${money(Math.abs(parts[i]))}`).join(' · ');
 const outside=m.features.filter(f=>fields.includes(f)&&(v[f]<D.ranges[f].min||v[f]>D.ranges[f].max));
 $('extrapolation').textContent=outside.length?`提示：${outside.map(f=>names[f]).join('、')}超出訓練資料範圍，屬於外插預測。`:'以目前模型的點估計呈現，並非保證收益；未納入的特徵不影響結果。';return profit;
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
$('reset').onclick=()=>{Object.assign(s,{method:'LASSO',k:2,compare:true,location:'California',sort:'abs',values:{'R&D Spend':100000,'Marketing Spend':150000,'Administration':85000},scenarios:[]});$('sort').value='abs';render();scenarios();toast('已重設模型、輸入與情境');};
$('save-scenario').onclick=()=>{if(s.scenarios.length>=5){toast('最多保留 5 個情境，請先移除一個');return;}s.scenarios.push({method:s.method,k:s.k,values:{...s.values},location:s.location,profit:predict()});scenarios();toast('已加入情境比較');};
$('download').onclick=()=>{const m=model(),q=v=>'"'+String(v).replaceAll('"','""')+'"',rows=[['Method','Feature count','Test RMSE','Test R2','Adjusted R2','MAE','Selected features'],[m.method,m.k,m.rmse,m.r2,m.adjustedR2,m.mae,m.features.join(', ')],[],['Predictor','Estimate','SE','t','p','CI low','CI high','VIF'],...m.coefficients.map(c=>[c.name,c.estimate,c.se,c.t,c.p,c.low,c.high,c.vif??'']),[],['Sample ID','Actual profit','Predicted profit','Residual','Error %'],...m.residuals.map(r=>[r.id,r.actual,r.predicted,r.residual,r.percent])],url=URL.createObjectURL(new Blob(['\ufeff'+rows.map(r=>r.map(q).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`startup-${m.method.replaceAll(' ','-')}-k${m.k}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已匯出目前模型與測試結果');};
$('ranking').innerHTML=D.comparison.map(r=>`<tr><td><span class="feature-chip">${r.Rank===1?'★ ':''}${r.Rank}</span></td><td>${r.Model}</td><td>${r.Features}</td><td><strong>${money(r['CV RMSE'])}</strong></td><td>± ${num(r['CV Std'],2)}</td><td>${money(r['Test RMSE'])}</td><td>${num(r['Test R2'])}</td></tr>`).join('');
$('split-tag').textContent=`${D.trainSize} / ${D.testSize} 訓練與測試`;$('generated').textContent='資料產生時間：'+new Date(D.generatedAt).toLocaleString('zh-TW',{timeZone:'Asia/Taipei'})+'（台灣時間）。重新執行 main.py 後，重新整理頁面即可載入更新數據。';render();
})();
