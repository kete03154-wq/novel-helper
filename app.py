from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil, os
from novel_core import NovelAgent

app = FastAPI(title="AI 长篇小说创作助手")

# 初始化全局 Agent（项目数据保存在 ./novel_data）
agent = NovelAgent(project_path="./novel_data", genre="玄幻")

# 静态文件（前端页面）
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def index():
    return FileResponse("static/index.html")

# ---------- 生成章节 ----------
@app.post("/generate_chapter")
async def generate_chapter(
    ch: int = Form(...),
    title: str = Form(...),
    outline: str = Form(...),
    author_style: str = Form("")
):
    result = agent.generate_chapter(ch, title, outline, author_style=author_style if author_style else None)
    return JSONResponse(content=result)

# ---------- 润色 ----------
@app.post("/polish")
async def polish(
    content: str = Form(...),
    intensity: float = Form(0.5)
):
    result = agent.polish(content, intensity)
    return result

# ---------- 拆书分析 ----------
@app.post("/dissect")
async def dissect_book(file: UploadFile = File(...)):
    # 保存上传文件
    filepath = f"upload_{file.filename}"
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    result = agent.dissect_book(filepath, title=file.filename)
    os.remove(filepath)
    return result

# ---------- 导出 ----------
@app.get("/export/{fmt}")
async def export(fmt: str):
    result = agent.export(fmt)
    if result.get("success") and "filepath" in result:
        return FileResponse(result["filepath"], filename=os.path.basename(result["filepath"]))
    return result

# ---------- 项目状态 ----------
@app.get("/status")
async def status():
    return agent.get_status()

# ---------- 作家风格学习 ----------
@app.post("/learn_author")
async def learn_author(
    name: str = Form(...),
    chapters: str = Form(...)  # 多章文本用特殊分隔符分开，客户端处理
):
    # 简单起见，这里按固定分隔符 "===CHAPTER===" 拆分
    chapter_list = chapters.split("===CHAPTER===")
    chapter_list = [c.strip() for c in chapter_list if c.strip()]
    result = agent.learn_author(name, chapter_list)
    return result

# ---------- 分支管理 ----------
@app.post("/create_branch")
async def create_branch(name: str = Form(...), from_chapter: int = Form(...)):
    result = agent.create_branch(name, from_chapter)
    return result

@app.post("/switch_branch")
async def switch_branch(name: str = Form(...)):
    result = agent.switch_branch(name)
    return result