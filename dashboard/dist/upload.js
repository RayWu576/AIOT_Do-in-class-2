(() => {
'use strict';
const $=id=>document.getElementById(id),h=v=>String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
let session=null,overview=null,lastJob=null;
const base=()=>$('api-base').value.trim().replace(/\/$/,'');
$('api-base').value=location.protocol==='file:'?'http://127.0.0.1:8000':location.origin;
async function request(path,options={}){
 const url=base();if(!/^https?:\/\//i.test(url))throw Error('請輸入 http:// 或 https:// 後端網址。');
 let response;try{response=await fetch(url+path,options);}catch{throw Error('無法連線至 Python 後端。請啟動 uvicorn api:app，並由 http://127.0.0.1:8000 開啟網站。內建資料集仍可使用。');}
 let body;try{body=await response.json();}catch{throw Error('此網址沒有分析 API；GitHub Pages 只能提供靜態結果。請使用 Python 後端網址。');}
 if(!response.ok)throw Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||body));
 return body;
}
async function health(){
 try{const info=await request('/api/health');if(info.status!=='ok')throw Error('後端尚未就緒');$('api-status').textContent='● 分析後端已連線 · CSV / Excel · 最多 '+info.maxRows+' 筆';}
 catch(e){$('api-status').textContent=e.message;}
}
$('check-api').onclick=health;
if(location.protocol!=='file:')health();else $('api-status').textContent='目前為靜態檔案模式；上傳分析請啟動後端並開啟 http://127.0.0.1:8000。';
function featurePicker(){
 if(!overview)return;
 const target=$('upload-target').value;
 $('upload-features').innerHTML=overview.columns.map((name,i)=>{
  const disabled=name===target||overview.constants.includes(name),recommended=overview.suggestedFeatures.includes(name)&&!disabled;
  return `<label><input type="checkbox" data-column="${i}" ${recommended?'checked':''} ${disabled?'disabled':''}><span>${h(name)}<small>${name===target?'目標 Y':overview.constants.includes(name)?'常數／全缺值':overview.identifiers.includes(name)?'ID：分析時排除':overview.numeric.includes(name)?'數值':'類別'}</small></span></label>`;
 }).join('');
}
$('upload-target').onchange=featurePicker;$('select-inputs').onclick=featurePicker;
$('upload-file').onchange=async()=>{
 const file=$('upload-file').files[0];if(!file)return;
 session=null;overview=null;$('upload-config').hidden=true;$('upload-preview').replaceChildren();$('analysis-status').textContent='正在讀取與檢查資料…';
 try{
  if(file.size>10*1024*1024)throw Error('檔案超過 10 MB，請先取樣或縮減資料。');
  if(!/\.(csv|xlsx)$/i.test(file.name))throw Error('請選擇 CSV 或 .xlsx。');
  const form=new FormData();form.append('file',file);
  const result=await request('/api/uploads',{method:'POST',body:form});session=result.session_id;overview=result.overview;
  $('upload-overview').textContent=`${overview.samples} 筆 · ${overview.columns.length} 欄 · ${overview.numeric.length} 數值 / ${overview.categorical.length} 類別 · ${Object.values(overview.missing).reduce((a,b)=>a+b,0)} 個缺值 · ${overview.duplicates} 重複列 · ${overview.constants.length} 常數欄位`;
  $('upload-preview').innerHTML='<thead><tr>'+overview.columns.map(c=>`<th>${h(c)}</th>`).join('')+'</tr></thead><tbody>'+overview.preview.map(row=>'<tr>'+overview.columns.map(c=>`<td>${row[c]===null?'—':h(row[c])}</td>`).join('')+'</tr>').join('')+'</tbody>';
  const targets=overview.numeric.filter(c=>!overview.constants.includes(c));
  if(!targets.length)throw Error('此資料沒有可用的連續數值目標欄位。');
  $('upload-target').innerHTML=targets.map(c=>`<option value="${h(c)}">${h(c)}</option>`).join('');$('upload-target').value=targets.filter(c=>!overview.identifiers.includes(c)).at(-1)||targets.at(-1);
  featurePicker();$('upload-config').hidden=false;$('analysis-status').textContent='已顯示前 10 筆。請選擇 Y 與 X，再開始分析。';
 }catch(e){$('analysis-status').textContent=e.message;}
};
$('clear-upload').onclick=async()=>{
 try{if(session)await request('/api/uploads/'+session,{method:'DELETE'});session=null;overview=null;$('upload-file').value='';$('upload-config').hidden=true;$('upload-preview').replaceChildren();$('upload-overview').textContent='';$('analysis-status').textContent='已清除暫存上傳；既有分析畫面仍可瀏覽。';}
 catch(e){$('analysis-status').textContent=e.message;}
};
$('upload-form').onsubmit=async event=>{
 event.preventDefault();if(!session||!overview)return;
 try{
  const features=[...$('upload-features').querySelectorAll('input:checked')].map(el=>overview.columns[+el.dataset.column]);
  if(!features.length)throw Error('請至少選取一個輸入欄位。');
  const counts=$('upload-counts').value.trim();const maxK=+$('upload-max-k').value;
  const config={target:$('upload-target').value,features,test_size:+$('upload-split').value,seed:+$('upload-seed').value,
    inner_folds:+$('upload-inner').value,outer_folds:+$('upload-outer').value,max_comparison_features:maxK,
    title:($('upload-file').files[0]?.name||'Uploaded dataset').slice(0,120),unit:$('upload-unit').value.trim()||'原始目標單位',
    ordered_observations:$('upload-ordered').checked};
  if(counts){config.feature_counts=counts.split(',').map(x=>Number(x.trim()));if(config.feature_counts.some(k=>!Number.isInteger(k)||k<1))throw Error('自訂 k 請輸入以逗號分隔的正整數。');}
  $('upload-fields').disabled=true;$('api-base').disabled=true;$('check-api').disabled=true;$('export-analysis').hidden=true;$('analysis-status').textContent='分析排程中…每次驗證都會重新擬合前處理與選特徵。';
  const created=await request('/api/analyses',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:session,config})});
  lastJob={id:created.job_id,base:base()};
  const started=Date.now();let result;
  while(Date.now()-started<420000){
    const state=await request('/api/analyses/'+created.job_id);
    if(state.status==='failed')throw Error(state.detail);
    if(state.status==='completed'){result=state.result;break;}
    $('analysis-status').textContent=state.status==='queued'?'分析等待中…':'正在進行訓練、內層選特徵與外層驗證… '+Math.round((Date.now()-started)/1000)+' 秒';
    await new Promise(resolve=>setTimeout(resolve,700));
  }
  if(!result)throw Error('等待分析結果逾時。請檢查後端狀態，或降低特徵數與 CV 折數。');
  result.apiBase=lastJob.base;result.jobId=lastJob.id;
  window.regressionDashboard.addDataset(result.metadata.id,result);
  $('analysis-status').textContent=`分析完成：${result.models.length} 組固定 k 結果。依訓練資料 Nested CV 排名；最佳流程：${result.comparison[0].Model}。`+(result.warnings.length?' '+result.warnings.join('；'):'');
  $('export-analysis').hidden=false;
 }catch(e){$('analysis-status').textContent=e.message;}
 finally{$('upload-fields').disabled=false;$('api-base').disabled=false;$('check-api').disabled=false;}
};
$('export-analysis').onclick=async()=>{
 if(!lastJob)return;
 try{
  const response=await fetch(lastJob.base+'/api/analyses/'+lastJob.id+'/export');if(!response.ok)throw Error('分析已過期，請重新執行。');
  const url=URL.createObjectURL(await response.blob()),a=document.createElement('a');a.href=url;a.download='regression-analysis.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 }catch(e){$('analysis-status').textContent=e.message;}
};
})();
