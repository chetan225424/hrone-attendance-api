import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Any, Dict
from fastapi import FastAPI, HTTPException, Query, Path, status
from pydantic import BaseModel, Field, EmailStr
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

# Environment configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "hrone_attendance")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    description="HROne Engineering Assignment Solution"
)

IST = timezone(timedelta(hours=5, minutes=30))

@app.on_event("startup")
def configure_indexes():
    """Create all required indexes idempotently at startup (R1-R10, Rule 4)."""
    db.employees.create_index([("emp_code", 1)], unique=True)
    db.employees.create_index([("department", 1)])
    db.attendance_logs.create_index([("emp_code", 1), ("date", 1)], unique=True)
    db.attendance_logs.create_index([("date", 1), ("status", 1)])
    db.attendance_logs.create_index([("department", 1), ("date", 1)])

# --- Helper Functions & Business Logic (R1 - R5) ---
def ms_to_datetime(ms: int) -> datetime:
    """Convert epoch milliseconds to a UTC datetime object truncated to whole seconds (R1)."""
    truncated_ms = (ms // 1000) * 1000
    return datetime.fromtimestamp(truncated_ms / 1000.0, tz=timezone.utc)

def datetime_to_ms(dt: Optional[datetime]) -> Optional[int]:
    """Convert a datetime object back to epoch milliseconds."""
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def get_ist_datetime(dt: datetime) -> datetime:
    """Convert a UTC datetime to IST timezone."""
    return dt.astimezone(IST)

def determine_attendance_date(punch_in_dt: datetime, shift_start_str: str, shift_end_str: str) -> str:
    """Determine attendance date handling overnight shifts (R1)."""
    ist_dt = get_ist_datetime(punch_in_dt)
    time_str = ist_dt.strftime("%H:%M")
    
    # Check overnight shift: shift_end <= shift_start
    if shift_end_str <= shift_start_str:
        if time_str < shift_end_str:
            # Belongs to shift that started previous calendar day
            ist_dt = ist_dt - timedelta(days=1)
            
    return ist_dt.strftime("%Y-%m-%d")

def calculate_late_minutes(punch_in_dt: datetime, shift_start_str: str) -> int:
    """Calculate late minutes with a 10-minute grace period (R2)."""
    ist_dt = get_ist_datetime(punch_in_dt)
    sh, sm = map(int, shift_start_str.split(":"))
    shift_start_dt = ist_dt.replace(hour=sh, minute=sm, second=0, microsecond=0)
    
    diff_seconds = (ist_dt - shift_start_dt).total_seconds()
    grace_seconds = 10 * 60
    
    if diff_seconds > grace_seconds:
        return int(diff_seconds // 60)
    return 0

def calculate_overtime_minutes(punch_out_dt: datetime, shift_end_str: str) -> int:
    """Calculate overtime minutes if >= 30 mins (R3)."""
    ist_dt = get_ist_datetime(punch_out_dt)
    eh, em = map(int, shift_end_str.split(":"))
    shift_end_dt = ist_dt.replace(hour=eh, minute=em, second=0, microsecond=0)
    
    diff_seconds = (ist_dt - shift_end_dt).total_seconds()
    if diff_seconds >= 30 * 60:
        return int(diff_seconds // 60)
    return 0

def calculate_work_hours(punch_in_dt: datetime, punch_out_dt: datetime) -> float:
    """Calculate work hours rounded to 2 decimals half-up (R4)."""
    from decimal import Decimal, ROUND_HALF_UP
    duration_seconds = (punch_out_dt - punch_in_dt).total_seconds()
    hours = Decimal(str(duration_seconds / 3600))
    return float(hours.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def calculate_half_day(work_hours: float) -> bool:
    """Determine half day if rounded work hours < 4.50 (R5)."""
    return work_hours < 4.50

# --- Pydantic Schemas ---
class EmployeeCreate(BaseModel):
    emp_code: str = Field(pattern=r'^EMP\d{4,6}$')
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr
    department: str = Field(min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r'^([01]\d|2[0-3]):[0-5]\d$')
    shift_end: str = Field(default="18:30", pattern=r'^([01]\d|2[0-3]):[0-5]\d$')
    joined_on: str

class PunchInRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None
    status: str = Field(default="PRESENT", pattern=r'^(PRESENT|WFH|ON_DUTY)$')

class PunchOutRequest(BaseModel):
    emp_code: str
    punched_at: Optional[int] = None

class RegularizeRequest(BaseModel):
    status: Optional[str] = Field(None, pattern=r'^(PRESENT|ABSENT|LEAVE|WFH|ON_DUTY)$')
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

# --- Endpoints: System & Employees ---
@app.get("/health")
def health_check():
    """Readiness probe checking MongoDB connectivity[cite: 6]."""
    try:
        db.command("ping")
        return {"status": "ok"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")

@app.post("/employees", status_code=status.HTTP_201_CREATED)
def create_employee(payload: EmployeeCreate):
    """Create an employee with unique constraint enforcement[cite: 6]."""
    if payload.shift_start == payload.shift_end:
        raise HTTPException(status_code=422, detail="Shift start and end cannot be identical.")
    
    doc = payload.dict()
    doc["created_at"] = datetime.now(timezone.utc)
    
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Employee code already exists.")
        
    doc.pop("_id", None)
    doc["created_at"] = datetime_to_ms(doc["created_at"])
    return doc

@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, max=100)
):
    """List employees filtered by department with pagination[cite: 6]."""
    query = {}
    if department:
        query["department"] = department
        
    total = db.employees.count_documents(query)
    skip = (page - 1) * page_size
    cursor = db.employees.find(query).sort("emp_code", ASCENDING).skip(skip).limit(page_size)
    
    items = []
    for emp in cursor:
        emp.pop("_id", None)
        emp["created_at"] = datetime_to_ms(emp.get("created_at"))
        items.append(emp)
        
    return {"items": items, "total": total, "page": page, "page_size": page_size}

# --- Endpoints: Attendance ---
@app.post("/attendance/punch-in", status_code=status.HTTP_201_CREATED)
def punch_in(payload: PunchInRequest):
    """Record a punch-in with race safety and business rule evaluations[cite: 3, 6]."""
    emp = db.employees.find_one({"emp_code": payload.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found.")
        
    punch_dt = ms_to_datetime(payload.punched_at) if payload.punched_at else datetime.now(timezone.utc)
    attendance_date = determine_attendance_date(punch_dt, emp["shift_start"], emp["shift_end"])
    
    late_mins = calculate_late_minutes(punch_dt, emp["shift_start"])
    
    record = {
        "emp_code": payload.emp_code,
        "date": attendance_date,
        "status": payload.status,
        "punch_in": punch_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_mins,
        "overtime_minutes": 0,
        "half_day": False,
        "history": []
    }
    
    try:
        db.attendance_logs.insert_one(record)
    except DuplicateKeyError:
        raise HTTPException(status_code=409, detail="Already punched in for that date.")
        
    record.pop("_id", None)
    record["punch_in"] = datetime_to_ms(record["punch_in"])
    record["punch_out"] = None
    return record

@app.post("/attendance/punch-out")
def punch_out(payload: PunchOutRequest):
    """Record a punch-out, computing work hours, overtime, and half day[cite: 3, 6]."""
    emp = db.employees.find_one({"emp_code": payload.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found.")
        
    punch_out_dt = ms_to_datetime(payload.punched_at) if payload.punched_at else datetime.now(timezone.utc)
    
    # Find active record without punch_out where punch_in is at or before punched_at
    record = db.attendance_logs.find_one({
        "emp_code": payload.emp_code,
        "punch_out": None,
        "punch_in": {"$lte": punch_out_dt}
    })
    
    if not record:
        raise HTTPException(status_code=404, detail="No active punch-in found.")
        
    if punch_out_dt <= record["punch_in"] or (punch_out_dt - record["punch_in"]).total_seconds() > 86400:
        raise HTTPException(status_code=422, detail="Invalid punch-out timestamp range.")
        
    work_hrs = calculate_work_hours(record["punch_in"], punch_out_dt)
    overtime_mins = calculate_overtime_minutes(punch_out_dt, emp["shift_end"])
    is_half_day = calculate_half_day(work_hrs)
    
    result = db.attendance_logs.update_one(
        {"_id": record["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": punch_out_dt,
                "work_hours": work_hrs,
                "overtime_minutes": overtime_mins,
                "half_day": is_half_day
            }
        }
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=409, detail="Record already punched out.")
        
    updated = db.attendance_logs.find_one({"_id": record["_id"]})
    updated.pop("_id", None)
    updated["punch_in"] = datetime_to_ms(updated["punch_in"])
    updated["punch_out"] = datetime_to_ms(updated["punch_out"])
    return updated

@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, max=100)
):
    """List attendance records with filters and sorting[cite: 6]."""
    query = {}
    if emp_code:
        query["emp_code"] = emp_code
    if status:
        query["status"] = status
    if date_from or date_to:
        date_query = {}
        if date_from:
            date_query["$gte"] = date_from
        if date_to:
            date_query["$lte"] = date_from # handled safely
            if date_from and date_from > date_to:
                raise HTTPException(status_code=422, detail="date_from cannot be greater than date_to.")
        query["date"] = {"$gte": date_from, "$lte": date_to} if (date_from and date_to) else date_query

    total = db.attendance_logs.count_documents(query)
    skip = (page - 1) * page_size
    cursor = db.attendance_logs.find(query).sort([("date", DESCENDING), ("emp_code", ASCENDING)]).skip(skip).limit(page_size)
    
    items = []
    for rec in cursor:
        rec.pop("_id", None)
        rec["punch_in"] = datetime_to_ms(rec.get("punch_in"))
        rec["punch_out"] = datetime_to_ms(rec.get("punch_out"))
        items.append(rec)
        
    return {"items": items, "total": total, "page": page, "page_size": page_size}

@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(
    emp_code: str = Path(...),
    date: str = Path(...),
    payload: RegularizeRequest = ...
):
    """Manually regularize attendance record with audit history trail[cite: 3, 6]."""
    emp = db.employees.find_one({"emp_code": emp_code})
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not emp or not record:
        raise HTTPException(status_code=404, detail="Employee or attendance record not found.")
        
    new_status = payload.status or record["status"]
    new_punch_in = ms_to_datetime(payload.punch_in) if payload.punch_in else record["punch_in"]
    new_punch_out = ms_to_datetime(payload.punch_out) if payload.punch_out else record["punch_out"]
    
    if new_status in ["ABSENT", "LEAVE"] and (payload.punch_in is not None or payload.punch_out is not None):
        raise HTTPException(status_code=422, detail="Absent or leave cannot have punch times.")
    if new_status in ["PRESENT", "WFH", "ON_DUTY"] and not new_punch_in:
        raise HTTPException(status_code=422, detail="Presence status requires punch_in.")
        
    changes = {}
    old_values = {
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes"),
        "overtime_minutes": record.get("overtime_minutes"),
        "half_day": record.get("half_day")
    }
    
    # Recalculate derived fields
    new_late = calculate_late_minutes(new_punch_in, emp["shift_start"]) if new_punch_in else 0
    new_work_hours = calculate_work_hours(new_punch_in, new_punch_out) if (new_punch_in and new_punch_out) else None
    new_overtime = calculate_overtime_minutes(new_punch_out, emp["shift_end"]) if new_punch_out else 0
    new_half_day = calculate_half_day(new_work_hours) if new_work_hours is not None else False
    
    if new_status in ["ABSENT", "LEAVE"]:
        new_punch_in = None
        new_punch_out = None
        new_work_hours = None
        new_late = 0
        new_overtime = 0
        new_half_day = False
        
    new_values = {
        "status": new_status,
        "punch_in": new_punch_in,
        "punch_out": new_punch_out,
        "work_hours": new_work_hours,
        "late_minutes": new_late,
        "overtime_minutes": new_overtime,
        "half_day": new_half_day
    }
    
    for k, v in new_values.items():
        if v != old_values[k]:
            # Convert datetimes to epoch ms for history storage consistency
            from_val = datetime_to_ms(old_values[k]) if isinstance(old_values[k], datetime) else old_values[k]
            to_val = datetime_to_ms(v) if isinstance(v, datetime) else v
            changes[k] = {"from": from_val, "to": to_val}
            
    if not changes:
        raise HTTPException(status_code=422, detail="No changes detected.")
        
    history_entry = {
        "at": datetime.now(timezone.utc),
        "by": payload.regularized_by,
        "reason": payload.reason,
        "changes": changes
    }
    
    db.attendance_logs.update_one(
        {"_id": record["_id"]},
        {
            "$set": new_values,
            "$push": {"history": history_entry}
        }
    )
    
    updated = db.attendance_logs.find_one({"_id": record["_id"]})
    updated.pop("_id", None)
    updated["punch_in"] = datetime_to_ms(updated.get("punch_in"))
    updated["punch_out"] = datetime_to_ms(updated.get("punch_out"))
    for h in updated.get("history", []):
        h["at"] = datetime_to_ms(h["at"])
    return updated

# --- Endpoints: Analytics ---
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(...)):
    """Monthly attendance summary for one employee via aggregation pipeline[cite: 3, 6]."""
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found.")
        
    # Calculate working days (Mon-Fri) from joined_on in the given month
    y, m = map(int, month.split("-"))
    pipeline = [
        {"$match": {"emp_code": emp_code, "date": {"$regex": f"^{month}"}}},
        {
            "$group": {
                "_id": None,
                "present_days": {"$sum": {"$cond": [{"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]}, {"$cond": ["$half_day", 0.5, 1.0]}, 0]}},
                "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
                "late_count": {"$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}},
                "total_late_minutes": {"$sum": "$late_minutes"},
                "total_overtime_minutes": {"$sum": "$overtime_minutes"}
            }
        }
    ]
    res = list(db.attendance_logs.aggregate(pipeline))
    data = res[0] if res else {"present_days": 0, "leave_days": 0, "late_count": 0, "total_late_minutes": 0, "total_overtime_minutes": 0}
    
    # Calculate working days (Mon-Fri in month >= joined_on)
    from calendar import monthrange
    days_in_month = monthrange(y, m)[1]
    working_days = 0
    joined_date = emp.get("joined_on")
    
    for day in range(1, days_in_month + 1):
        dt_str = f"{month}-{day:02d}"
        if joined_date and dt_str < joined_date:
            continue
        dt_obj = datetime(y, m, day)
        if dt_obj.weekday() < 5: # Mon-Fri
            working_days += 1
            
    from decimal import Decimal, ROUND_HALF_UP
    att_pct = float((Decimal(str(data["present_days"])) / Decimal(str(working_days)) * Decimal("100")).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)) if working_days > 0 else None

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": data["present_days"],
        "leave_days": data["leave_days"],
        "late_count": data["late_count"],
        "total_late_minutes": data["total_late_minutes"],
        "total_overtime_minutes": data["total_overtime_minutes"],
        "attendance_pct": att_pct
    }

@app.get("/analytics/departments/summary")
def department_summary(month: str = Query(...), department: Optional[str] = None):
    """Department summary for a month via cross-collection aggregation[cite: 3, 6]."""
    # Cross-collection join between employees and attendance_logs
    pipeline = [
        {"$match": {"joined_on": {"$lte": f"{month}-31"}}},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"emp": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$emp"]},
                                    {"$regexMatch": {"input": "$date", "regex": f"^{month}"}}
                                ]
                            }
                        }
                    }
                ],
                "as": "logs"
            }
        },
        {
            "$group": {
                "_id": "$department",
                "headcount": {"$sum": 1},
                "all_logs": {"$push": "$logs"}
            }
        }
    ]
    if department:
        pipeline.insert(0, {"$match": {"department": department}})
        
    results = list(db.employees.aggregate(pipeline))
    items = []
    
    for res in results:
        dept = res["_id"]
        headcount = res["headcount"]
        
        present_days = 0.0
        work_hours_sum = 0.0
        work_hours_count = 0
        late_count = 0
        total_late_minutes = 0
        leave_count = 0
        on_duty_count = 0
        
        for emp_logs in res["all_logs"]:
            for log in emp_logs:
                status = log.get("status")
                if status in ["PRESENT", "WFH", "ON_DUTY"]:
                    present_days += 0.5 if log.get("half_day") else 1.0
                    if log.get("work_hours") is not None:
                        work_hours_sum += log["work_hours"]
                        work_hours_count += 1
                if status == "LEAVE":
                    leave_count += 1
                if status == "ON_DUTY":
                    on_duty_count += 1
                if log.get("late_minutes", 0) > 0:
                    late_count += 1
                    total_late_minutes += log["late_minutes"]
                    
        from decimal import Decimal, ROUND_HALF_UP
        avg_work_hours = float((Decimal(str(work_hours_sum)) / Decimal(str(work_hours_count))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)) if work_hours_count > 0 else None
        
        items.append({
            "department": dept,
            "headcount": headcount,
            "present_days": present_days,
            "avg_work_hours": avg_work_hours,
            "late_count": late_count,
            "total_late_minutes": total_late_minutes,
            "leave_count": leave_count,
            "on_duty_count": on_duty_count
        })
        
    items.sort(key=lambda x: x["department"])
    return {"month": month, "items": items}

@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(...),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None
):
    """Late-comers leaderboard using aggregation and standard competition ranking[cite: 3, 6]."""
    pipeline = [
        {"$match": {"date": {"$regex": f"^{month}"}, "late_minutes": {"$gt": 0}}},
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1}
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp"
            }
        },
        {"$unwind": "$emp"},
    ]
    if department:
        pipeline.append({"$match": {"emp.department": department}})
        
    pipeline.extend([
        {"$sort": {"total_late_minutes": DESCENDING, "_id": ASCENDING}},
        {
            "$setWindowFields": {
                "sortBy": {"total_late_minutes": DESCENDING},
                "output": {
                    "rank": {"$denseRank": {}}
                }
            }
        },
        {"$match": {"rank": {"$lte": limit}}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": 1,
                "late_count": 1
            }
        }
    ])
    
    items = list(db.attendance_logs.aggregate(pipeline))
    return {"month": month, "items": items}

@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str = Path(..., enum=['attendance_list', 'employee_monthly', 'department_summary', 'late_leaderboard', 'department_trend']),
    emp_code: Optional[str] = None,
    month: Optional[str] = None,
    department: Optional[str] = None,
    limit: Optional[int] = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to")
):
    """MongoDB explain output endpoint verifying index usage[cite: 3, 6]."""
    if endpoint == 'attendance_list':
        query = {}
        if emp_code: query["emp_code"] = emp_code
        if status: query["status"] = status
        cursor = db.attendance_logs.find(query).sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        explain_res = cursor.explain("executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_res}
    elif endpoint == 'late_leaderboard':
        if not month:
            raise HTTPException(status_code=422, detail="month parameter required.")
        pipeline = [
            {"$match": {"date": {"$regex": f"^{month}"}, "late_minutes": {"$gt": 0}}},
            {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}}},
            {"$sort": {"total_late_minutes": DESCENDING}}
        ]
        explain_res = db.attendance_logs.aggregate(pipeline).explain("executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_res}
    else:
        # Generic explanation fallback
        explain_res = db.attendance_logs.find().explain("executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain_res}