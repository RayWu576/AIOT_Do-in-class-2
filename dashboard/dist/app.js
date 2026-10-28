(() => {
'use strict';
const BUNDLE=window.DASHBOARD_DATA,$=id=>document.getElementById(id);
let datasetId=Object.keys(BUNDLE?.datasets||{})[0],D=BUNDLE?.datasets?.[datasetId]||BUNDLE;
if(!D){document.querySelector('main').innerHTML='<section class="panel"><h2>尚未產生資料</h2><p>請先執行 python main.py，再重新開啟網站。</p></section>';return;}
const methods=['Correlation','Mutual Information','RFE','RFECV','LASSO'],colors=['#5585ef','#09ad87','#e7a43c','#9c68db','#e16a87'];
const desc=['Pearson 相關係數排名','互資訊 · 非線性關聯','遞迴消除特徵','交叉驗證選擇數量','L1 正則化 · 係數排名'],icons=['↗','⋈','✂','⌘','λ'];
const h=v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
let names={},fields=[];
const counts=()=>D.featureCounts||Array.from({length:D.metadata.maxFeatures},(_,i)=>i+1);
const usedInputs=m=>m.originalFeatures||m.features;

const num=(v,d=4)=>v===null||v===undefined||!Number.isFinite(Number(v))?'—':Number(v).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const money=v=>v===null||v===undefined?'—':(D.metadata?.unitPrefix||'')+num(v,2)+(!D.metadata?.unitPrefix&&D.metadata?.unit?' '+D.metadata.unit:'');
const pv=v=>v===null||v===undefined?'—':v<.0001?'< 0.0001':num(v);
const defaults=()=>!D.encoding&&D.metadata.id==='startups'?{'R&D Spend':100000,'Marketing Spend':150000,'Administration':85000}:Object.fromEntries(D.inputs.map(f=>[f.name,f.default]));
const s={method:D.defaultMethod||'LASSO',k:D.defaultK||2,compare:true,values:{},location:'California',sort:'abs',scenarios:[],page:0};
const model=()=>D.models.find(m=>m.method===s.method&&m.k===s.k);
const isBest=m=>Math.abs(m.rmse-Math.min(...D.models.map(x=>x.rmse)))<1e-6;
const storage={get:k=>{try{return localStorage.getItem(k)}catch{return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch{}}};
if(storage.get('startup-theme')==='dark')document.body.classList.add('dark');
$('theme').onclick=()=>{document.body.classList.toggle('dark');storage.set('startup-theme',document.body.classList.contains('dark')?'dark':'light');};
function toast(t){$('toast').textContent=t;$('toast').classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>$('toast').classList.remove('show'),2500);}
function choose(method,k){s.method=method;s.k=k;s.page=0;render();}
function controls(){
 $('methods').innerHTML=methods.map((m,i)=>`<button class="method-button ${m===s.method?'active':''}" data-method="${m}" aria-pressed="${m===s.method}"><span class="method-icon">${icons[i]}</span><strong>${m}</strong><small>${desc[i]}</small></button>`).join('');
 $('methods').querySelectorAll('button').forEach(b=>b.onclick=()=>choose(b.dataset.method,s.k));
 $('counts').innerHTML=counts().map(k=>`<button data-k="${k}" aria-pressed="${k===s.k}" class="${k===s.k?'active':''}">k = ${k}${isBest(D.models.find(m=>m.method===s.method&&m.k===k))?'<small>★</small>':''}</button>`).join('');
 $('counts').querySelectorAll('button').forEach(b=>b.onclick=()=>choose(s.method,+b.dataset.k));
 const notes=['依訓練資料的絕對 Pearson 相關係數選取前 k 個特徵。','依訓練資料的互資訊選取前 k 個特徵；二元特徵按離散特徵處理。','標準化後，依線性迴歸係數遞迴移除特徵，保留 k 個。','固定 k 曲線使用 RFECV 的 RFE 消除路徑；自動選取結果請見下方完整工作流程比較。','固定 k 時以標準化 LassoCV 係數絕對值排名；零係數同分按原欄位順序。自動選取結果另列於下方。'];
 $('method-note').textContent=notes[methods.indexOf(s.method)];$('compare').checked=s.compare;
}
function metrics(m){
 const base=D.models.find(x=>x.method===s.method&&x.k===D.metadata.maxFeatures);const full=base?.rmse??D.baselineRmse??m.rmse,delta=(full-m.rmse)/full*100;
 const items=[['TEST RMSE',money(m.rmse),isBest(m)?'★ 固定 k 比較中的最低誤差':'越低越好 · 保留測試集','↘'],['R² SCORE',num(m.r2),`解釋 ${num(m.r2*100,2)}% 的測試集變異`,'↗'],['ADJUSTED R²',num(m.adjustedR2),'測試集 · 校正特徵數量','≋'],['MEAN ABSOLUTE ERROR',money(m.mae),'測試集平均絕對誤差','±'],['F-STATISTIC',num(m.f,1),`訓練集 OLS · p ${pv(m.fP)}`,'ƒ']];
 $('metrics').innerHTML=items.map(([label,v,note,icon])=>`<article class="metric"><div class="metric-label">${label}<span class="metric-symbol">${icon}</span></div><strong>${h(v)}</strong><p>${h(note)}</p></article>`).join('');
 $('active-features').innerHTML=m.features.map(f=>`<span class="feature-chip">${h(f)}</span>`).join('');
 $('insight').textContent=s.k===D.metadata.maxFeatures?`完整 ${D.metadata.maxFeatures} 特徵模型 · 與精簡模型比較後再判斷`:`相較完整模型，測試 RMSE ${delta>=0?'降低':'增加'} ${num(Math.abs(delta),1)}%`;
 $('equation').innerHTML=`${h(D.metadata.target)} = <b>${num(m.intercept,2)}</b> `+m.features.map((f,i)=>`${m.weights[i]>=0?'+':'−'} <b>${num(Math.abs(m.weights[i]),5)}</b> × (${h(f)})`).join(' ');
}
function chart(id,metric){
 const visible=s.compare?methods:[s.method],data=D.models.filter(m=>visible.includes(m.method)),vals=data.map(m=>m[metric]);
 const min=Math.min(...vals),max=Math.max(...vals),pad=(max-min)*.16||1,lo=min-pad,hi=max+pad,W=560,H=265,L=55,R=15,T=15,B=36,x=k=>L+(k-Math.min(...counts()))*(W-L-R)/Math.max(1,Math.max(...counts())-Math.min(...counts())),y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${metric} 隨特徵數變化">`;
 for(let i=0;i<5;i++){const v=lo+(hi-lo)*i/4;svg+=`<line class="grid-line" x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"/><text x="${L-8}" y="${y(v)+3}" text-anchor="end">${metric==='rmse'?num(v,Math.abs(v)<100?1:0):num(v,3)}</text>`;}
 for(const k of counts())svg+=`<line class="grid-line" x1="${x(k)}" y1="${T}" x2="${x(k)}" y2="${H-B}" opacity=".5"/><text x="${x(k)}" y="${H-15}" text-anchor="middle">${k}</text>`;
 svg+=`<line x1="${x(s.k)}" y1="${T}" x2="${x(s.k)}" y2="${H-B}" stroke="var(--purple)" opacity=".25" stroke-dasharray="4 4"/>`;
 visible.filter(v=>v!==s.method).concat(s.method).forEach(method=>{
  const points=D.models.filter(m=>m.method===method),c=colors[methods.indexOf(method)],active=method===s.method;
  svg+=`<path d="${points.map((m,i)=>`${i?'L':'M'}${x(m.k)} ${y(m[metric])}`).join(' ')}" fill="none" stroke="${c}" stroke-width="${active?3:1.8}" opacity="${active?1:.65}" ${active?'':'stroke-dasharray="5 3"'}/>`;
  points.forEach(m=>{const label=`${method} · k=${m.k} · ${metric==='rmse'?money(m.rmse):num(m.r2)}`;svg+=`<circle cx="${x(m.k)}" cy="${y(m[metric])}" r="${m.k===s.k&&active?6:3.5}" fill="${c}" stroke="var(--surface)" stroke-width="2" tabindex="0" role="button" data-method="${method}" data-k="${m.k}" aria-label="${h(label)}"><title>${h(label)}</title></circle>`;});
 });
 svg+=`<text x="${(W+L-R)/2}" y="${H-1}" text-anchor="middle" style="font-size:9px">Number of features (k)</text></svg>`;
 $(id).innerHTML=`<div class="legend">${visible.map(m=>`<button data-method="${m}" class="${m===s.method?'selected':''}"><i style="background:${colors[methods.indexOf(m)]}"></i>${m}</button>`).join('')}</div>`+svg+'<div class="chart-tooltip">點選圖例或資料點，切換目前模型</div>';
 $(id).querySelectorAll('.legend button').forEach(b=>b.onclick=()=>choose(b.dataset.method,s.k));
 $(id).querySelectorAll('circle').forEach(dot=>{const activate=()=>choose(dot.dataset.method,+dot.dataset.k);dot.onclick=activate;dot.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};dot.onmouseenter=()=>{$(id).querySelector('.chart-tooltip').textContent=dot.getAttribute('aria-label');};});
}
function coefficients(m){
 $('coefficients').innerHTML=m.coefficients.map(c=>`<tr><td>${h(c.name)}</td><td><strong>${num(c.estimate)}</strong></td><td>${num(c.se)}</td><td>${num(c.t,2)}</td><td><span class="pill ${c.p!==null&&c.p<.05?'green-pill':'purple-pill'}">${pv(c.p)}</span></td><td>[${num(c.low,3)}, ${num(c.high,3)}]</td><td>${c.vif===null?'—':`<span class="${c.vif>=5?'negative':'positive'}">${num(c.vif,2)}</span>`}</td></tr>`).join('');
 const effects=m.features.map((f,i)=>Math.abs(m.weights[i]*(D.ranges[f].max-D.ranges[f].min))),largest=Math.max(...effects,1);
 $('coefficient-bars').innerHTML='<div class="eyebrow purple">FEATURE EFFECT · 訓練範圍內的目標變化幅度（'+h(D.metadata.unit)+'）</div>'+m.features.map((f,i)=>`<div class="bar-row"><span>${h(f)}</span><div class="bar-track"><div class="bar-fill" style="width:${effects[i]/largest*100}%"></div></div><span>${Math.abs(effects[i])<1000?num(effects[i],2):num(effects[i]/1000,1)+'k'}</span></div>`).join('');
}
function inputs(m){
 $('inputs').innerHTML=D.inputs.map((f,i)=>{
  const active=usedInputs(m).includes(f.name),value=s.values[f.name];
  const label=`<label class="field-label" for="number-${i}">${h(f.label)}<span>${active?'使用中':'未納入目前模型'}</span></label>`;
  if(f.kind==='categorical')return `<div class="${active?'':'inactive'}">${label}<select id="number-${i}" data-category="${i}" aria-label="${h(f.label)}">${f.options.map(v=>`<option value="${h(v)}" ${v===value?'selected':''}>${h(v)}</option>`).join('')}</select></div>`;
  const min=D.encoding?-1e12:0,max=f.binary?1:1e12;
  return `<div class="${active?'':'inactive'}">${label}<div class="input-pair"><input id="number-${i}" type="number" min="${min}" max="${max}" step="${f.step}" value="${value??''}" aria-label="${h(f.label)}"><input id="range-${i}" type="range" min="${Math.min(f.min,value??f.default)}" max="${Math.max(f.max,value??f.default)}" step="${f.step}" value="${value??f.default}" aria-label="${h(f.label)}滑桿"></div></div>`;
 }).join('');
 D.inputs.forEach((f,i)=>{const n=$('number-'+i);if(f.kind==='categorical'){n.onchange=()=>{s.values[f.name]=n.value;predict();};return;}
  const r=$('range-'+i);n.oninput=()=>{if(n.value===''&&D.encoding){s.values[f.name]=null;predict();return;}
   const v=n.valueAsNumber;if(n.value===''||!Number.isFinite(v)||v<(D.encoding?-1e12:0)||v>(f.binary?1:1e12)||(f.binary&&!Number.isInteger(v)))return;
   s.values[f.name]=v;r.min=Math.min(+r.min,v);r.max=Math.max(+r.max,v);r.value=v;predict();};
  n.onchange=()=>{n.value=s.values[f.name]??'';};r.oninput=()=>{s.values[f.name]=+r.value;n.value=r.value;predict();};});
 // Compatibility adapter for the already-published 50 Startups snapshot.
 const hasLegacyState=!D.encoding&&D.metadata.id==='startups';$('state-label').hidden=!hasLegacyState;$('state').hidden=!hasLegacyState;
 $('state').value=s.location;$('state-status').textContent=m.features.some(f=>f.startsWith('State_'))?'地區特徵使用中':'地區未納入目前模型';predict();
}
function encodedValues(dataset,raw){
 if(!dataset.encoding)return {...raw,'State_Florida':+(s.location==='Florida'),'State_New York':+(s.location==='New York')};
 return Object.fromEntries(dataset.encoding.map(e=>{const v=raw[e.column];const missing=v===null||v===undefined||v==='';return [e.name,e.kind==='numeric'?Number(missing?e.fill:v):+(String(missing?e.fill:v)===e.category)];}));
}
window.regressionPredict=(dataset,m,raw)=>{const v=encodedValues(dataset,raw);return m.intercept+m.weights.reduce((total,w,i)=>total+w*v[m.features[i]],0);};
function predict(){
 const m=model(),v=encodedValues(D,s.values),parts=m.weights.map((w,i)=>w*v[m.features[i]]),value=window.regressionPredict(D,m,s.values);
 $('profit').textContent=money(value);$('contributions').textContent=m.features.map((f,i)=>`${names[f]||f} ${parts[i]>=0?'+':'−'}${money(Math.abs(parts[i]))}`).join(' · ');
 const outside=D.inputs.filter(f=>f.kind!=='categorical'&&usedInputs(m).includes(f.name)&&s.values[f.name]!==null&&(s.values[f.name]<f.min||s.values[f.name]>f.max));
 $('extrapolation').textContent=outside.length?'提示：'+outside.map(f=>f.label).join('、')+'超出訓練資料範圍。':'此為目前模型的點估計，空白數值會使用訓練中位數補值。'+D.metadata.targetNote;return value;
}
function scenarios(){
 $('scenarios').innerHTML=s.scenarios.map((v,i)=>`<div class="scenario"><span>${i+1}. ${v.method} · k=${v.k}<br><span class="muted">${fields.map(f=>`${h(names[f]||f)} ${h(typeof v.values[f]==='string'?v.values[f]:num(v.values[f],2))}`).join(' / ')} · ${D.metadata.id==='startups'?v.location:''}</span></span><strong>${h(money(v.profit))}</strong><button data-index="${i}" aria-label="移除情境 ${i+1}">×</button></div>`).join('');
 $('scenarios').querySelectorAll('button').forEach(b=>b.onclick=()=>{s.scenarios.splice(+b.dataset.index,1);scenarios();});
}
function residuals(m){
 const sorted=[...m.residuals].sort((a,b)=>s.sort==='id'?a.id-b.id:s.sort==='percent'?b.percent-a.percent:Math.abs(b.residual)-Math.abs(a.residual));
 const rows=sorted.slice(s.page*20,(s.page+1)*20);
 $('residuals').innerHTML=rows.map(r=>`<tr><td>${h(D.metadata.sampleLabel)} #${r.id}</td><td>${h(money(r.actual))}</td><td>${h(money(r.predicted))}</td><td class="${r.residual>=0?'positive':'negative'}">${r.residual>=0?'+':'−'}${h(money(Math.abs(r.residual)))}</td><td>${h(money(Math.abs(r.residual)))}</td><td>${num(r.percent,2)}%</td><td><span class="pill ${Math.abs(r.residual)>m.rmse?'purple-pill':'green-pill'}">${Math.abs(r.residual)>m.rmse?'高殘差':'範圍內'}</span></td></tr>`).join('');
 const W=900,H=190,L=68,R=24,T=20,B=28,all=m.residuals.flatMap(r=>[r.actual,r.predicted]),lo=Math.min(...all)-Math.max(1,Math.max(...all)-Math.min(...all))*.1,hi=Math.max(...all)+Math.max(1,Math.max(...all)-Math.min(...all))*.1,x=v=>L+(v-lo)/(hi-lo)*(W-L-R),y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="真實與預測利潤散佈圖，虛線為完美預測"><line x1="${x(lo)}" y1="${y(lo)}" x2="${x(hi)}" y2="${y(hi)}" stroke="var(--muted)" opacity=".35" stroke-dasharray="4 4"/>`;
 for(let i=0;i<4;i++){const v=lo+(hi-lo)*i/3;svg+=`<line class="grid-line" x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}"/><text x="${L-8}" y="${y(v)+3}" text-anchor="end">${Math.abs(v)<1000?num(v,1):num(v/1000,0)+'k'}</text><text x="${x(v)}" y="${H-8}" text-anchor="middle">${Math.abs(v)<1000?num(v,1):num(v/1000,0)+'k'}</text>`;}
 svg+=m.residuals.map(r=>`<circle cx="${x(r.actual)}" cy="${y(r.predicted)}" r="5" fill="${Math.abs(r.residual)>m.rmse?'#9c68db':'#09ad87'}" opacity=".85"><title>${h(D.metadata.sampleLabel)} #${r.id} · 真實 ${h(money(r.actual))} · 預測 ${h(money(r.predicted))}</title></circle>`).join('');
 $('scatter').innerHTML=svg+`<text x="${W-R}" y="${T}" text-anchor="end">X 真實目標值 / Y 預測目標值 · 虛線 = 完美預測</text></svg>`;
 const pages=Math.ceil(m.residuals.length/20);
 $('residual-pager').innerHTML=pages<=1?'':`<button id="prev-page" ${s.page===0?'disabled':''}>← 上一頁</button><span>第 ${s.page+1} / ${pages} 頁，共 ${m.residuals.length} 筆</span><button id="next-page" ${s.page+1===pages?'disabled':''}>下一頁 →</button>`;
 if(pages>1){$('prev-page').onclick=()=>{s.page--;residuals(m);};$('next-page').onclick=()=>{s.page++;residuals(m);};}
}
function renderAssumptions(m){
 const d=m.diagnostics;$('assumptions').hidden=!d;if(!d)return;
 const ptext=v=>v?.p===null||!v?'—':pv(v.p);
 $('assumption-summary').textContent=`Shapiro–Wilk p=${ptext(d.shapiro)} · Breusch–Pagan p=${ptext(d.breuschPagan)} · Durbin–Watson=${num(d.durbinWatson,3)}`;
 $('assumption-note').textContent=d.note+' '+d.independenceNote+' 圖形最多顯示 500 個訓練資料點；完整數據包含在 JSON 匯出中。';
 function plot(id,xs,ys,title,zero){
  if(!xs.length){$(id).textContent='資料不足，無法繪圖。';return;}
  const W=550,H=220,L=55,R=18,T=20,B=30;
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const x=v=>L+(v-xmin)/(xmax-xmin||1)*(W-L-R),y=v=>H-B-(v-ymin)/(ymax-ymin||1)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${h(title)}">`;
  for(let i=0;i<4;i++){const val=ymin+(ymax-ymin)*i/3;svg+=`<line class="grid-line" x1="${L}" y1="${y(val)}" x2="${W-R}" y2="${y(val)}"/><text x="${L-5}" y="${y(val)+3}" text-anchor="end">${num(val,1)}</text>`;}
  if(zero&&ymin<=0&&ymax>=0)svg+=`<line x1="${L}" y1="${y(0)}" x2="${W-R}" y2="${y(0)}" stroke="var(--purple)" stroke-dasharray="4 4"/>`;
  const step=Math.max(1,Math.ceil(xs.length/500));for(let i=0;i<xs.length;i+=step)svg+=`<circle cx="${x(xs[i])}" cy="${y(ys[i])}" r="2.5" fill="var(--purple)" opacity=".65"/>`;
  $(id).innerHTML=svg+'</svg>';
 }
 plot('linearity-chart',d.linearity.fitted,d.linearity.residuals,'訓練殘差與擬合值',true);
 plot('qq-chart',d.qq.map(q=>q.theoretical),d.qq.map(q=>q.observed),'訓練殘差 Q-Q 圖',false);
 $('diagnostic-downloads').innerHTML=D.apiBase&&D.jobId?['linearity','qq'].map(kind=>`<a class="feature-chip" href="${h(D.apiBase)}/api/analyses/${h(D.jobId)}/plots/${encodeURIComponent(m.method)}/${m.k}/${kind}.png" target="_blank" rel="noopener">↓ ${kind==='qq'?'Q-Q':'殘差'} PNG</a>`).join(''):'';
}
function render(){const m=model();controls();metrics(m);chart('rmse-chart','rmse');chart('r2-chart','r2');coefficients(m);inputs(m);residuals(m);renderAssumptions(m);}
$('compare').onchange=e=>{s.compare=e.target.checked;chart('rmse-chart','rmse');chart('r2-chart','r2');};$('state').onchange=e=>{s.location=e.target.value;predict();};$('sort').onchange=e=>{s.sort=e.target.value;s.page=0;residuals(model());};
$('reset').onclick=()=>{resetState();render();scenarios();toast('已重設目前資料集的模型、輸入與情境');};
$('save-scenario').onclick=()=>{if(s.scenarios.length>=5){toast('最多保留 5 個情境，請先移除一個');return;}s.scenarios.push({method:s.method,k:s.k,values:{...s.values},location:s.location,profit:predict()});scenarios();toast('已加入情境比較');};
$('download').onclick=()=>{const m=model(),q=v=>'"'+String(v??'').replaceAll('"','""')+'"',rows=[['Dataset',D.metadata.title],['Target unit',D.metadata.unit],[],['Method','Feature count','Train RMSE','Tuning CV RMSE','Test RMSE','Test MAE','Test MSE','Test R2','Adjusted R2','Selected features'],[m.method,m.k,m.trainRmse??'',m.cvRmse??'',m.rmse,m.mae,m.mse??m.rmse*m.rmse,m.r2,m.adjustedR2,m.features.join(', ')],[],['Predictor','Estimate','SE','t','p','CI low','CI high','VIF'],...m.coefficients.map(c=>[c.name,c.estimate,c.se,c.t,c.p,c.low,c.high,c.vif??'']),[],['Sample ID','Actual target','Predicted target','Residual','Error %'],...m.residuals.map(r=>[r.id,r.actual,r.predicted,r.residual,r.percent])],url=URL.createObjectURL(new Blob(['\ufeff'+rows.map(r=>r.map(q).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`${datasetId}-${m.method.replaceAll(' ','-')}-k${m.k}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('已匯出目前模型與測試結果');};
function resetState(){
 names={...D.metadata.fieldLabels};
 fields=D.inputs.map(f=>f.name);
 Object.assign(s,{method:D.defaultMethod,k:D.defaultK,compare:true,values:defaults(),location:'California',sort:'abs',scenarios:[],page:0});
 $('sort').value='abs';
}
function setupDataset(){
 resetState();
 const meta=D.metadata;
 $('breadcrumb').textContent='分析工作室 / '+meta.title+' / 特徵選擇探索';
 $('dataset-intro').textContent=meta.description+' 比較五種特徵選擇方法，觀察特徵數與預測表現的變化。';
 $('dataset-tag').textContent=`◉ ${meta.title} · ${meta.cleanSize} 筆有效資料`;
 $('dataset-summary').textContent=`${meta.rawSize} 筆原始資料 → ${meta.cleanSize} 筆有效資料 · ${meta.maxFeatures} 個候選特徵 · 目標：${meta.target}（${meta.unit}）`;
 $('predictor-title').textContent='調整輸入，預測'+meta.targetLabel;
 $('prediction-target').textContent=`預測${meta.targetLabel} · ${meta.target}（${meta.unit}）`;
 $('rmse-unit').textContent=`RMSE（${meta.unit}）`;
 $('residual-description').textContent=`保留測試集 ${D.testSize} 筆資料 · 殘差 = 真實值 − 預測值 · 單位：${meta.unit}`;
 $('dataset-methodology').textContent=`${meta.title}：原始 ${meta.rawSize} 筆，有效 ${meta.cleanSize} 筆。${meta.missingPolicy} random_state=${D.seed}。${meta.targetNote}`;
 $('cv-methodology').textContent=`五種方法選取特徵後均使用 LinearRegression。正式排名使用外層 ${D.config?.outer_folds||5} 折、內層 ${D.config?.inner_folds||4} 折巢狀交叉驗證，預處理只擬合各次訓練資料；測試集不參與選擇。`;$('source-url').textContent=meta.source;
 if(meta.source.startsWith('https://'))$('source-url').href=meta.source;else $('source-url').removeAttribute('href');
 const rankingTable=$('ranking').closest('table');
 rankingTable.querySelector('thead').innerHTML='<tr><th>排名</th><th>方法</th><th>特徵數</th><th>Train RMSE</th><th>Nested CV RMSE</th><th>CV 標準差</th><th>Test RMSE</th><th>Test MAE</th><th>Test MSE</th><th>Test R²</th><th>Adjusted R²</th></tr>';
 $('ranking').innerHTML=D.comparison.map(r=>`<tr><td><span class="feature-chip">${r.Rank===1?'★ ':''}${r.Rank}</span></td><td>${h(r.Model)}</td><td>${r.Features}</td><td>${h(money(r['Train RMSE']))}</td><td><strong>${h(money(r['CV RMSE']))}</strong></td><td>± ${num(r['CV Std'],2)}</td><td>${h(money(r['Test RMSE']))}</td><td>${h(money(r['Test MAE']))}</td><td>${num(r['Test MSE'],3)}</td><td>${num(r['Test R2'])}</td><td>${num(r['Test Adjusted R2'])}</td></tr>`).join('');
 $('split-tag').textContent=`${D.trainSize} / ${D.testSize} 訓練與測試`;
 $('generated').textContent='資料產生時間：'+new Date(D.generatedAt).toLocaleString('zh-TW',{timeZone:'Asia/Taipei'})+'（台灣時間）。';
 document.title='DataLab · '+meta.problem;
 render();scenarios();
}
$('dataset-select').innerHTML=Object.entries(BUNDLE.datasets).map(([id,d])=>`<option value="${h(id)}">${h(d.metadata.problem)}</option>`).join('');
$('dataset-select').onchange=e=>{datasetId=e.target.value;D=BUNDLE.datasets[datasetId];setupDataset();};
window.regressionDashboard={addDataset:(id,result)=>{
 BUNDLE.datasets[id]=result;
 const option=[...$('dataset-select').options].find(o=>o.value===id);
 if(!option)$('dataset-select').add(new Option(result.metadata.problem,id));
 datasetId=id;D=result;$('dataset-select').value=id;setupDataset();
},current:()=>({dataset:D,method:s.method,k:s.k,values:{...s.values}})};
setupDataset();
})();
