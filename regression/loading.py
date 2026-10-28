"""CSV/XLSX loading and non-mutating schema inspection."""
import io
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from .config import AnalysisError

MAX_BYTES=10*1024*1024


def load_table(source, filename=None):
    if isinstance(source, pd.DataFrame):
        frame=source.copy()
    else:
        if isinstance(source, (str, Path)):
            path=Path(source)
            if path.stat().st_size > MAX_BYTES:
                raise AnalysisError("File exceeds 10 MB limit.")
            data=path.read_bytes(); filename=filename or path.name
        else:
            data=source.read() if hasattr(source,'read') else bytes(source)
        if len(data)>MAX_BYTES:
            raise AnalysisError("File exceeds 10 MB limit.")
        suffix=Path(filename or '').suffix.lower()
        try:
            if suffix=='.csv':
                try: frame=pd.read_csv(io.BytesIO(data),encoding='utf-8-sig')
                except UnicodeDecodeError: frame=pd.read_csv(io.BytesIO(data),encoding='cp950')
            elif suffix=='.xlsx':
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    if sum(i.file_size for i in z.infolist())>50*1024*1024:
                        raise AnalysisError("Expanded Excel file exceeds 50 MB limit.")
                frame=pd.read_excel(io.BytesIO(data),engine='openpyxl')
            else: raise AnalysisError("Only CSV and .xlsx files are supported.")
        except AnalysisError: raise
        except Exception as exc: raise AnalysisError("Cannot parse this CSV/Excel file.") from exc
    names=[str(c).strip() for c in frame.columns]
    if any(not n for n in names) or len(names)!=len(set(names)):
        raise AnalysisError("Column names must be nonempty and unique after trimming.")
    if any(len(n)>120 for n in names):
        raise AnalysisError("Column names must be at most 120 characters.")
    frame.columns=names
    if frame.empty or len(names)<2:
        raise AnalysisError("Dataset needs observations and at least two columns.")
    frame=frame.replace([np.inf,-np.inf],np.nan)
    for col in frame.select_dtypes(exclude='number').columns:
        frame[col]=frame[col].astype(object).map(lambda v: str(v).strip() if pd.notna(v) and str(v).strip() else np.nan)
    return frame.reset_index(drop=True)


def identifier_columns(frame):
    names=[]
    for name in frame:
        col=frame[name]
        named=bool(re.search(r'(^|[_\s])(id|uuid|index)($|[_\s])',name,re.I) or re.search(r'(Id|ID)$',name))
        serial=(pd.api.types.is_numeric_dtype(col) and col.notna().all() and col.nunique()==len(col)
                and np.allclose(col,np.round(col)) and (col.is_monotonic_increasing or col.is_monotonic_decreasing))
        # Numeric order alone does not make a legitimate year/measurement an ID.
        index_named=name.lower() in {'unnamed: 0','row_number','rownum','serial','record_no'}
        if named or (serial and index_named): names.append(name)
    return names


def inspect_table(frame):
    numeric=frame.select_dtypes(include='number').columns.tolist()
    constants=[c for c in frame if frame[c].nunique(dropna=True)<=1]
    ids=identifier_columns(frame)
    return dict(samples=len(frame),features=len(frame.columns),columns=frame.columns.tolist(),
                numeric=numeric,categorical=[c for c in frame if c not in numeric],
                missing={c:int(frame[c].isna().sum()) for c in frame},
                duplicates=int(frame.duplicated().sum()),constants=constants,identifiers=ids,
                suggestedFeatures=[c for c in frame if c not in constants+ids],
                preview=frame.head(10).astype(object).where(pd.notna(frame.head(10)),None).to_dict(orient='records'))
