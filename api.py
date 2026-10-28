"""Local FastAPI adapter. Run: python -m uvicorn api:app --host 127.0.0.1 --port 8000.

The static site remains usable without this server. Uploaded data is temporary,
not written to Git or existing experiment results. Do not expose without auth.
"""
import io
import json
import sys
import time
import uuid
import tempfile
import subprocess
import threading
import shutil
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI,UploadFile,File,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response,FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field,ConfigDict
from regression.loading import load_table,inspect_table,MAX_BYTES
from regression.config import AnalysisConfig,AnalysisError

ROOT=Path(__file__).resolve().parent
TEMP=tempfile.TemporaryDirectory(prefix='regression-upload-')
SESSIONS={};JOBS={}
LOCK=threading.Lock();POOL=ThreadPoolExecutor(max_workers=1)
MAX_SESSIONS=8;TTL=3600


@asynccontextmanager
async def lifespan(app):
    yield
    POOL.shutdown(wait=True,cancel_futures=True)
    TEMP.cleanup()


app=FastAPI(title='CRISP-DM Regression API',lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=['http://127.0.0.1:8000','http://localhost:8000','https://raywu576.github.io'],
    allow_methods=['GET','POST','DELETE'],allow_headers=['Content-Type'])


class AnalyzeRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session_id:str
    config:dict


class PredictRequest(BaseModel):
    method:str
    k:int
    values:dict


def clean():
    now=time.time()
    for key,session in list(SESSIONS.items()):
        if now-session['created']>TTL and not any(j['session']==key and j['status'] in ('queued','running') for j in JOBS.values()):
            shutil.rmtree(session['directory'],ignore_errors=True)
            SESSIONS.pop(key,None)
            for job_id in [j for j in JOBS if JOBS[j]['session']==key]:JOBS.pop(job_id,None)


def session_or_404(key):
    with LOCK:
        clean();session=SESSIONS.get(key)
        if session is None:raise HTTPException(404,'Upload session expired or not found; upload again.')
        return session


def job_or_404(key):
    with LOCK:
        clean();job=JOBS.get(key)
        if job is None:raise HTTPException(404,'Analysis job not found or expired.')
        return job


@app.get('/api/health')
def health():return {'status':'ok','maxFileMB':10,'maxRows':5000,'maxEncodedFeatures':60,'maxSeconds':180}


@app.post('/api/uploads')
async def upload(file:UploadFile=File(...)):
    raw=await file.read(MAX_BYTES+1)
    try:
        frame=load_table(raw,filename=file.filename)
        if len(frame)>5000:raise AnalysisError('Upload analysis supports at most 5000 rows. Sample data first.')
        overview=inspect_table(frame)
    except (AnalysisError,ValueError) as exc:raise HTTPException(422,str(exc))
    with LOCK:
        clean()
        if len(SESSIONS)>=MAX_SESSIONS:raise HTTPException(429,'At most 8 temporary uploads; delete an old session first.')
        key=uuid.uuid4().hex;directory=Path(TEMP.name)/key;directory.mkdir()
        frame.to_csv(directory/'dataset.csv',index=False)
        SESSIONS[key]=dict(directory=directory,filename=Path(file.filename or 'upload.csv').name,created=time.time())
    return {'session_id':key,'filename':SESSIONS[key]['filename'],'overview':overview}


def run_job(job_id,cfg):
    job=JOBS[job_id];job['status']='running'
    directory=job['directory']
    try:
        # One bounded process at a time; communicate kills and drains it on timeout.
        process=subprocess.Popen([sys.executable,'-X','utf8','-m','regression.worker',str(directory)],
            cwd=str(ROOT),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
            text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            out,err=process.communicate(json.dumps(cfg),timeout=cfg['max_seconds']+15)
        except subprocess.TimeoutExpired:
            process.kill();process.communicate();raise AnalysisError('Analysis exceeded the hard time limit; reduce folds or feature comparisons.')
        if process.returncode:
            detail=json.loads((directory/'error.json').read_text(encoding='utf-8'))['detail'] if (directory/'error.json').exists() else 'Analysis failed; check data or reduce workload.'
            raise AnalysisError(detail)
        result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        result['metadata']['id']='upload-'+job_id
        result['metadata']['source']=job['filename']
        job.update(status='completed',result=result)
    except Exception as exc:job.update(status='failed',detail=str(exc))


@app.post('/api/analyses',status_code=202)
def start_analysis(request:AnalyzeRequest):
    session=session_or_404(request.session_id)
    try:
        cfg=AnalysisConfig(**request.config).validate()
        # Client limits may be lower, never raise backend resource ceilings.
        if cfg.max_seconds>180 or cfg.max_encoded_features>60 or cfg.max_rows>5000 or cfg.max_workload>15000:
            raise AnalysisError('Backend limits: 180 seconds, 60 encoded features, 5000 rows and 15000 workload units.')
    except (TypeError,ValueError) as exc:raise HTTPException(422,str(exc))
    with LOCK:
        if sum(j['status'] in ('queued','running') for j in JOBS.values())>=2:
            raise HTTPException(429,'Analysis capacity reached; wait for an active job.')
        if len(JOBS)>=24:raise HTTPException(429,'Temporary analysis limit reached; delete old uploads or restart server.')
        key=uuid.uuid4().hex;directory=session['directory']/key;directory.mkdir()
        shutil.copyfile(session['directory']/'dataset.csv',directory/'dataset.csv')
        JOBS[key]=dict(session=request.session_id,directory=directory,filename=session['filename'],status='queued')
        from dataclasses import asdict
        POOL.submit(run_job,key,asdict(cfg))
    return {'job_id':key,'status':'queued'}


@app.get('/api/analyses/{job_id}')
def analysis_status(job_id:str):
    job=job_or_404(job_id)
    result={'job_id':job_id,'status':job['status']}
    if job['status']=='completed':result['result']=job['result']
    if job['status']=='failed':result['detail']=job['detail']
    return result


@app.post('/api/analyses/{job_id}/predict')
def predict(job_id:str,request:PredictRequest):
    job=job_or_404(job_id)
    if job['status']!='completed':raise HTTPException(409,'Analysis is not complete.')
    result=job['result'];inputs=result['inputs']
    if set(request.values)-{v['name'] for v in inputs}:raise HTTPException(422,'Unknown input columns.')
    row={}
    try:
        for col in inputs:
            value=request.values.get(col['name'])
            if value is None or value=='':row[col['name']]=np.nan
            elif col['kind']=='numeric':
                value=float(value)
                if not np.isfinite(value):raise ValueError('Numeric inputs must be finite.')
                row[col['name']]=value
            else:row[col['name']]=str(value)
        models=joblib.load(job['directory']/'models.joblib')
        model=models.get((request.method,request.k))
        if model is None:raise ValueError('Unknown method or feature count.')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');value=float(model.predict(pd.DataFrame([row]))[0])
        if not np.isfinite(value):raise ValueError('Prediction is undefined.')
        return {'prediction':value,'target':result['metadata']['target'],'unit':result['metadata']['unit']}
    except (ValueError,TypeError) as exc:raise HTTPException(422,str(exc))


@app.get('/api/analyses/{job_id}/export')
def export_result(job_id:str):
    job=job_or_404(job_id)
    if job['status']!='completed':raise HTTPException(409,'Analysis is not complete.')
    return Response(json.dumps(job['result'],ensure_ascii=False,allow_nan=False),media_type='application/json',
                    headers={'Content-Disposition':'attachment; filename="analysis.json"'})


@app.get('/api/analyses/{job_id}/plots/{method}/{k}/{kind}.png')
def plot(job_id:str,method:str,k:int,kind:str):
    job=job_or_404(job_id)
    if job['status']!='completed':raise HTTPException(409,'Analysis is not complete.')
    m=next((m for m in job['result']['models'] if m['method']==method and m['k']==k),None)
    if m is None or kind not in ('linearity','qq'):raise HTTPException(404,'Plot not found.')
    from regression.artifacts import diagnostic_png
    return Response(diagnostic_png(m,kind),media_type='image/png')


@app.delete('/api/uploads/{session_id}')
def delete_upload(session_id:str):
    session=session_or_404(session_id)
    with LOCK:
        if any(j['session']==session_id and j['status'] in ('queued','running') for j in JOBS.values()):
            raise HTTPException(409,'Wait for the analysis to finish before deleting this session.')
        shutil.rmtree(session['directory'],ignore_errors=True);SESSIONS.pop(session_id,None)
        for key in [key for key in JOBS if JOBS[key]['session']==session_id]:JOBS.pop(key,None)
    return {'deleted':True}


# API routes are registered before the static fallback.
app.mount('/',StaticFiles(directory=ROOT/'dashboard/dist',html=True),name='dashboard')
