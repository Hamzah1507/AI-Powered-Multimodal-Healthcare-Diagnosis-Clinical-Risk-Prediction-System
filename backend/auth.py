from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from database import db
from passlib.context import CryptContext
from datetime import datetime

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
users_collection = db["users"]

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

@router.post("/register")
async def register(user: UserRegister):
    existing = await users_collection.find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered!")
    hashed = pwd_context.hash(user.password)
    new_user = {
        "username": user.username,
        "email": user.email,
        "password": hashed,
        "created_at": datetime.utcnow()
    }
    await users_collection.insert_one(new_user)
    return {"status": "success", "message": "Account created successfully!"}

@router.post("/login")
async def login(user: UserLogin):
    db_user = await users_collection.find_one({"email": user.email})
    if not db_user:
        raise HTTPException(status_code=400, detail="Email not found!")
    if not pwd_context.verify(user.password, db_user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect password!")
    return {
        "status": "success",
        "username": db_user["username"],
        "email": db_user["email"],
        "message": "Login successful!"
    }