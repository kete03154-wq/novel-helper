"""
================================================================================
长篇小说AI智能体 —— 完全体 · 终极版 (DeepSeek适配版)
================================================================================
功能：
- 多模型动态调度（当前仅DeepSeek）
- 三层记忆 + 智能上下文组装
- 伏笔全生命周期管理（自动检测）
- 质量控制 / 去AI化 / 节奏分析
- 导出（MD、PDF、DOCX、打包下载）
- 拆书学习（世界观、角色、大纲提取）
- 文风学习 & 作家风格研习（至少三章样本）
- 动态状态自动更新（角色、场景、经济、实力）
- 多线叙事支持
- 情节一致性图谱
- 类型适配（玄幻/都市/科幻/历史/悬疑等）
- 回溯分支管理（从任意章节分叉，切换版本）
================================================================================
"""

import os, re, json, time, zipfile, shutil, requests
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import Counter, defaultdict

# ---------- DeepSeek 配置 ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-your-api-key")  # 请设置环境变量
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

def call_llm(messages: List[Dict[str, str]], model: str = "deepseek-chat",
             temperature: float = 0.8, max_tokens: int = 8192) -> str:
    """调用 DeepSeek API（兼容 OpenAI 格式）"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    response = requests.post(f"{DEEPSEEK_BASE_URL}/chat/completions",
                             headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"DeepSeek API 调用失败: {response.text}")
    return response.json()["choices"][0]["message"]["content"]

def generate_with_retry(messages: List[Dict], model: str = "deepseek-chat", temperature: float = 0.8,
                        max_tokens: int = 8192, max_retries: int = 3) -> Tuple[str, bool]:
    for attempt in range(max_retries):
        try:
            content = call_llm(messages, model, temperature, max_tokens)
            return content, True
        except Exception as e:
            print(f"LLM 失败 ({attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return f"生成失败: {e}", False
            time.sleep(2 ** attempt)
    return "失败", False

# ============================================================
# 提示词模板（未修改）
# ============================================================
PROMPTS = {
    "system_role": "你是一位资深小说作家，擅长各种类型的长篇创作。语言鲜活，结构严谨。",

    "world_gen": """根据关键词生成世界观设定：
关键词：{keywords}
请包含：时代背景、地理环境、社会势力、特殊规则、核心冲突。""",

    "character_gen": """基于以下世界观，生成主要角色档案：
世界观：{world_setting}
为每个角色提供：姓名、身份、外貌、性格、动机、成长弧线、初始实力、经济状况（如适用）。""",

    "volume_outline": """为第{vol}卷（共{total}卷）撰写大纲：
标题：{title}
世界观：{world}
角色：{chars}
请给出核心冲突、阶段目标、关键情节点。""",

    "chapter_outline": """为第{ch}章生成梗概（卷大纲：{vol_outline}）：
包含标题、3-5个情节点、悬念、出场角色、可埋伏笔。""",

    "chapter_content": """创作第{ch}章正文：
标题：{title}
梗概：{outline}
字数：{word_range}字左右
风格要求：{style}
近期上下文：{context}
角色状态：{chars}
伏笔提醒：{foreshadow}
作家风格注入：{author_style}
当前活跃故事线：{storyline}
请直接输出正文，结尾带有悬念钩子。""",

    "chapter_summary": """为第{ch}章生成小结（关键事件、角色变化、新伏笔）：
正文：{content}""",

    "auto_state_extraction": """从以下小说正文中提取状态变化（JSON格式）：
小说类型：{genre}
正文：{content}
提取：
1. 角色状态：位置、情绪、目标、实力变化、经济变化、关系变化
2. 场景状态：名称、势力变更
3. 潜在伏笔

输出示例：
{{
  "character_updates": [{{"name":"...","location":"...","emotion":"...","goal":"...","power_change":"...","economic_change":"...","relationship_change":"..."}}],
  "scene_updates": [{{"name":"...","description":"...","control_change":"..."}}],
  "potential_foreshadowing": ["..."]
}}
如果没有变化则空数组。""",

    "author_style_analysis": """分析以下{count}章样本的作家写作手法：
样本：
{combined}

从以下维度深入分析：
1. 句式特征 2. 修辞手法 3. 叙事视角 4. 对话风格
5. 描写密度 6. 节奏控制 7. 词汇偏好 8. 情节推进手法
最后给出800字以内的写作指导摘要。""",

    "author_style_inject": """你已学习作家「{name}」的风格。写作指导如下：
{guide}

创作要求：{req}""",

    "polish": """润色以下文本，去除重复用词、打断单调句式、统一风格：
原文：{original_content}""",

    "book_dissect_world": """分析以下小说内容，提取世界观设定（时代背景、地理、势力、特殊规则）：
{content}""",

    "book_dissect_chars": """分析以下小说内容，提取所有重要角色（姓名、身份、性格、关系）：
{content}""",

    "book_dissect_outline": """为以下小说章节撰写梗概（3-5句话），并标出冲突与悬念：
{content}""",

    "book_dissect_foreshadow": """从以下内容中找出可能的伏笔：
{content}"""
}

def get_prompt(key, **kwargs):
    return PROMPTS[key].format(**kwargs)

# ============================================================
# 类型配置（未修改）
# ============================================================
@dataclass
class GenreConfig:
    name: str
    track_power: bool = False
    track_economy: bool = False
    track_scene_control: bool = False
    power_levels: List[str] = field(default_factory=list)
    default_power_unit: str = ""

GENRE_PRESETS = {
    "玄幻": GenreConfig(name="玄幻", track_power=True, track_economy=True, track_scene_control=True,
                      power_levels=["凡人","炼气","筑基","金丹","元婴","化神","渡劫","大乘","真仙"],
                      default_power_unit="境界"),
    "都市": GenreConfig(name="都市", track_power=False, track_economy=True, track_scene_control=False),
    "科幻": GenreConfig(name="科幻", track_power=True, track_economy=True, track_scene_control=True,
                      power_levels=["普通人","基因强化","机械改造","精神觉醒","维度生物"],
                      default_power_unit="科技等级"),
    "历史": GenreConfig(name="历史", track_power=False, track_economy=True, track_scene_control=True,
                      default_power_unit="官职"),
    "悬疑": GenreConfig(name="悬疑", track_power=False, track_economy=False, track_scene_control=False),
    "言情": GenreConfig(name="言情", track_power=False, track_economy=True, track_scene_control=False),
    "武侠": GenreConfig(name="武侠", track_power=True, track_economy=True, track_scene_control=True,
                      power_levels=["不入流","三流","二流","一流","绝顶","宗师"],
                      default_power_unit="武学境界"),
    "末世": GenreConfig(name="末世", track_power=True, track_economy=True, track_scene_control=True),
    "无限流": GenreConfig(name="无限流", track_power=True, track_economy=True, track_scene_control=True),
}

# ============================================================
# 基础数据模型（未修改）
# ============================================================
@dataclass
class ChapterContent:
    chapter_number: int
    chapter_title: str
    content: str
    summary: str = ""
    storyline_id: str = "main"
    foreshadowing: List[str] = field(default_factory=list)
    key_events: List[str] = field(default_factory=list)

@dataclass
class CharacterState:
    name: str
    identity: str = ""
    location: str = ""
    emotion: str = ""
    goal: str = ""
    power_level: str = ""
    economic_status: str = ""
    items: List[str] = field(default_factory=list)
    relationships: Dict[str, str] = field(default_factory=dict)
    last_chapter: int = 0

@dataclass
class SceneState:
    name: str
    description: str = ""
    controlling_faction: str = ""
    last_chapter: int = 0

@dataclass
class Storyline:
    id: str
    name: str
    pov_character: str = ""
    current_chapter: int = 1
    current_volume: int = 1
    active: bool = True
    outline: str = ""
    associated_characters: List[str] = field(default_factory=list)

@dataclass
class GraphEntity:
    id: str
    type: str
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[str, str] = field(default_factory=dict)

@dataclass
class Foreshadowing:
    id: str
    content: str
    fs_type: str
    volume: int
    chapter: int
    resolve_chapter: Optional[int] = None
    status: str = "pending"
    importance: int = 5

@dataclass
class ChapterStats:
    chapter_number: int
    word_count: int
    event_count: int = 0
    scene_count: int = 0
    dialogue_ratio: float = 0
    climax_intensity: str = "medium"

# ============================================================
# 动态状态更新器（未修改）
# ============================================================
class DynamicStateUpdater:
    def __init__(self, genre: GenreConfig):
        self.genre = genre
        self.characters: Dict[str, CharacterState] = {}
        self.scenes: Dict[str, SceneState] = {}
        self.economic_ledger: Dict[str, float] = defaultdict(float)

    def extract_and_update(self, chapter_content: str, chapter_number: int) -> Dict:
        msgs = [
            {"role": "system", "content": "你是一位小说状态分析专家，请严格按 JSON 格式输出。"},
            {"role": "user", "content": get_prompt("auto_state_extraction",
                                                   genre=self.genre.name,
                                                   content=chapter_content[:6000])}
        ]
        raw, ok = generate_with_retry(msgs, temperature=0.2, max_tokens=2048)
        if not ok:
            return {"character_changes": 0, "scene_changes": 0, "potential_foreshadowing": []}
        try:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(json_match.group(0)) if json_match else {}
        except:
            data = {}
        for upd in data.get("character_updates", []):
            name = upd.get("name")
            if not name: continue
            if name not in self.characters:
                self.characters[name] = CharacterState(name=name)
            ch = self.characters[name]
            if upd.get("location"): ch.location = upd["location"]
            if upd.get("emotion"): ch.emotion = upd["emotion"]
            if upd.get("goal"): ch.goal = upd["goal"]
            if upd.get("power_change") and self.genre.track_power:
                ch.power_level = upd["power_change"]
            if upd.get("economic_change") and self.genre.track_economy:
                ch.economic_status = upd["economic_change"]
            if upd.get("relationship_change"):
                try:
                    rels = json.loads(upd["relationship_change"]) if isinstance(upd["relationship_change"], str) else upd["relationship_change"]
                    ch.relationships.update(rels)
                except:
                    pass
            ch.last_chapter = chapter_number
        for sc in data.get("scene_updates", []):
            name = sc.get("name")
            if not name: continue
            if name not in self.scenes:
                self.scenes[name] = SceneState(name=name)
            s = self.scenes[name]
            if sc.get("description"): s.description = sc["description"]
            if sc.get("control_change") and self.genre.track_scene_control:
                s.controlling_faction = sc["control_change"]
            s.last_chapter = chapter_number
        return {
            "character_changes": len(data.get("character_updates", [])),
            "scene_changes": len(data.get("scene_updates", [])),
            "potential_foreshadowing": data.get("potential_foreshadowing", [])
        }

    def get_character_summary(self) -> str:
        lines = []
        for name, ch in self.characters.items():
            parts = [f"{name}: 位置{ch.location}, 情绪{ch.emotion}, 目标{ch.goal}"]
            if self.genre.track_power:
                parts.append(f"实力{ch.power_level}")
            if self.genre.track_economy:
                parts.append(f"经济{ch.economic_status}")
            lines.append(", ".join(parts))
        return "\n".join(lines) if lines else "无角色信息"

    def get_scene_summary(self) -> str:
        if not self.scenes:
            return "无场景信息"
        return "\n".join(f"{name}: 势力{sc.controlling_faction}" for name, sc in self.scenes.items())

# ============================================================
# 多线叙事管理器（未修改）
# ============================================================
class StorylineManager:
    def __init__(self):
        self.storylines: Dict[str, Storyline] = {"main": Storyline(id="main", name="主线")}
        self.current_storyline_id: str = "main"

    def add_storyline(self, sid: str, name: str, pov: str = "", outline: str = ""):
        self.storylines[sid] = Storyline(id=sid, name=name, pov_character=pov, outline=outline)

    def switch_to(self, sid: str):
        if sid not in self.storylines:
            raise ValueError(f"故事线 {sid} 不存在")
        self.current_storyline_id = sid

    def get_current(self) -> Storyline:
        return self.storylines[self.current_storyline_id]

    def advance_chapter(self, sid: Optional[str] = None):
        target = sid or self.current_storyline_id
        if target in self.storylines:
            self.storylines[target].current_chapter += 1

    def get_active_context(self, memory, foreshadowing, state_updater, sid: Optional[str] = None):
        target = sid or self.current_storyline_id
        storyline = self.storylines[target]
        context = memory.get_recent_context(4)
        associated = set(storyline.associated_characters)
        if storyline.pov_character:
            associated.add(storyline.pov_character)
        char_summary = state_updater.get_character_summary()
        return {
            "context": context,
            "char_summary": char_summary,
            "storyline": storyline.name,
            "pov": storyline.pov_character
        }

# ============================================================
# 情节一致性图谱（未修改）
# ============================================================
class StoryGraph:
    def __init__(self):
        self.entities: Dict[str, GraphEntity] = {}

    def add_or_update(self, eid: str, etype: str, name: str, attrs: Dict = None):
        if eid in self.entities:
            self.entities[eid].attributes.update(attrs or {})
        else:
            self.entities[eid] = GraphEntity(id=eid, type=etype, name=name, attributes=attrs or {})

    def add_relation(self, src: str, dst: str, relation: str):
        if src in self.entities and dst in self.entities:
            self.entities[src].relations[dst] = relation

    def check_consistency(self, new_content: str) -> List[str]:
        warnings = []
        known_names = {ent.name for ent in self.entities.values()}
        mentioned = set(re.findall(r'[“"”]([^“"”]{1,5})[”"“]', new_content))
        for m in mentioned:
            if m not in known_names and len(m) >= 2:
                warnings.append(f"未知实体：{m}")
        return warnings

# ============================================================
# 记忆管理器（未修改）
# ============================================================
class ThreeLayerMemoryManager:
    def __init__(self, project_path: str, genre_cfg: GenreConfig):
        self.project_path = project_path
        self.recent_chapters: List[ChapterContent] = []
        self.current_volume_outline = ""
        self.genre = genre_cfg
        self.state_updater = DynamicStateUpdater(genre_cfg)
        self.storylines = StorylineManager()
        self.graph = StoryGraph()
        self.world_text = ""
        self.character_profiles = ""
        self._ensure_dirs()
        self._load_persisted()

    def _ensure_dirs(self):
        for d in ["设定", "章节", "总结", "状态"]:
            os.makedirs(os.path.join(self.project_path, d), exist_ok=True)

    def _load_persisted(self):
        world_path = os.path.join(self.project_path, "设定", "世界观.md")
        if os.path.exists(world_path):
            with open(world_path, 'r', encoding='utf-8') as f:
                self.world_text = f.read()
        char_path = os.path.join(self.project_path, "设定", "角色档案.md")
        if os.path.exists(char_path):
            with open(char_path, 'r', encoding='utf-8') as f:
                self.character_profiles = f.read()
        self._load_recent_chapters()

    def _load_recent_chapters(self):
        ch_dir = os.path.join(self.project_path, "章节")
        if not os.path.exists(ch_dir): return
        files = sorted([f for f in os.listdir(ch_dir) if f.endswith('.md') and f.startswith('第')])[-4:]
        for fname in files:
            path = os.path.join(ch_dir, fname)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            title = fname.replace('.md', '')
            num = 0
            m = re.search(r'第(\d+)章', title)
            if m: num = int(m.group(1))
            self.recent_chapters.append(ChapterContent(chapter_number=num, chapter_title=title, content=content))

    def add_chapter(self, chapter: ChapterContent):
        self.recent_chapters.append(chapter)
        if len(self.recent_chapters) > 4:
            self.recent_chapters.pop(0)
        result = self.state_updater.extract_and_update(chapter.content, chapter.chapter_number)
        warnings = self.graph.check_consistency(chapter.content)
        return result, warnings

    def get_recent_context(self, count: int = 4) -> str:
        return "\n\n".join(f"【{ch.chapter_title}】\n{ch.content}" for ch in self.recent_chapters[-count:])

# ============================================================
# 伏笔管理器（未修改）
# ============================================================
class ForeshadowingManager:
    def __init__(self, path: str):
        self.path = path
        self.items: List[Foreshadowing] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.items = [Foreshadowing(**d) for d in data.get("items", [])]

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump({"items": [asdict(i) for i in self.items]}, f, ensure_ascii=False, indent=2)

    def add(self, content: str, fs_type: str, vol: int, ch: int, **kwargs):
        fs_id = f"fs_{vol}_{ch}_{len(self.items)+1:04d}"
        self.items.append(Foreshadowing(id=fs_id, content=content, fs_type=fs_type, volume=vol, chapter=ch, **kwargs))
        self._save()

    def add_from_list(self, potentials: List[str], vol: int, ch: int):
        for p in potentials:
            self.add(p, "potential", vol, ch)

    def resolve(self, fs_id: str, resolution: str, ch: int, notes: str = ""):
        for f in self.items:
            if f.id == fs_id:
                f.status = "resolved"
                f.resolve_chapter = ch
                self._save()
                return True
        return False

    def get_pending(self) -> List[Foreshadowing]:
        return [f for f in self.items if f.status == "pending"]

    def get_for_chapter(self, ch: int, vol: int) -> Dict:
        pending = self.get_pending()
        recs = [f"⚠️ 伏笔「{f.content[:30]}...」应在第{ch}章回收" for f in pending if f.resolve_chapter == ch]
        return {"pending_count": len(pending), "recommendations": recs}

# ============================================================
# 作家风格研习（未修改）
# ============================================================
class AuthorStyleLearner:
    def __init__(self, project_path: str):
        self.path = os.path.join(project_path, "风格研习")
        os.makedirs(self.path, exist_ok=True)

    def learn(self, author_name: str, chapters: List[str]) -> str:
        if len(chapters) < 3:
            return "至少需要三章样本"
        combined = "\n\n---\n\n".join(f"第{i+1}章：\n{c[:4000]}" for i, c in enumerate(chapters))
        msgs = [
            {"role": "system", "content": PROMPTS["author_style_analysis"]},
            {"role": "user", "content": get_prompt("author_style_analysis", count=len(chapters), combined=combined)}
        ]
        analysis, _ = generate_with_retry(msgs, temperature=0.4, max_tokens=4000)
        with open(os.path.join(self.path, f"{author_name}.json"), 'w', encoding='utf-8') as f:
            json.dump({"name": author_name, "guide": analysis}, f, ensure_ascii=False)
        return analysis

    def get_guide(self, author_name: str) -> str:
        filepath = os.path.join(self.path, f"{author_name}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f).get("guide", "")
        return ""

# ============================================================
# 质量控制（未修改）
# ============================================================
class QualityController:
    def __init__(self):
        self.stats: List[ChapterStats] = []

    def analyze_rhythm(self, content: str, ch_num: int) -> ChapterStats:
        events = len(re.findall(r'(突然|顿时|忽然|没想到|猛然)', content))
        dialogues = re.findall(r'[""\'\'](.*?)[""\'\']', content)
        dialogue_len = sum(len(d) for d in dialogues)
        ratio = dialogue_len / len(content) if content else 0
        cs = ChapterStats(chapter_number=ch_num, word_count=len(content),
                          event_count=events, scene_count=len(re.findall(r'(来到|走进|离开)', content)),
                          dialogue_ratio=ratio)
        self.stats.append(cs)
        return cs

    def check_pacing_issues(self) -> List[Dict[str, Any]]:
        if len(self.stats) < 3:
            return []
        recent = self.stats[-5:]
        low = sum(1 for s in recent if s.event_count < 5 and s.dialogue_ratio < 0.1)
        if low >= 3:
            return [{"type": "pacing_slow", "msg": "连续多章情节密度过低，建议增加冲突或对话"}]
        return []

    def polish(self, content: str, intensity: float = 0.5) -> str:
        msgs = [
            {"role": "system", "content": PROMPTS["polish"]},
            {"role": "user", "content": f"原文：{content[:5000]}"}
        ]
        polished, _ = generate_with_retry(msgs, temperature=0.3+intensity*0.3, max_tokens=4096)
        return polished

# ============================================================
# 拆书学习（未修改）
# ============================================================
class BookDissector:
    def __init__(self, project_path: str):
        self.project_path = project_path

    def dissect_file(self, filepath: str, book_title: str = "") -> Dict:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return self.dissect_text(text, book_title or os.path.basename(filepath))

    def dissect_text(self, text: str, book_title: str) -> Dict:
        chapters = self._split_chapters(text)
        world_content = "\n".join(chapters[:2]) if len(chapters) >= 2 else chapters[0]
        world = self._extract_world(world_content)
        characters = self._extract_characters(text)
        outlines = [self._extract_outline(ch) for ch in chapters]
        foreshadowings = self._extract_foreshadowing(text)
        return {
            "title": book_title,
            "chapter_count": len(chapters),
            "world": world,
            "characters": characters,
            "outlines": outlines,
            "foreshadowings": foreshadowings
        }

    def _split_chapters(self, text: str) -> List[str]:
        parts = re.split(r'(第[零一二三四五六七八九十百千]+章\s*.*)', text)
        chapters = []
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                title = parts[i].strip()
                body = parts[i+1] if i+1 < len(parts) else ""
                chapters.append(f"{title}\n{body}")
        else:
            chunk = 5000
            for i in range(0, len(text), chunk):
                chapters.append(text[i:i+chunk])
        return chapters

    def _extract_world(self, content: str) -> str:
        msgs = [{"role": "system", "content": "你是文学分析专家。"},
                {"role": "user", "content": get_prompt("book_dissect_world", content=content[:5000])}]
        result, _ = generate_with_retry(msgs, temperature=0.5, max_tokens=2048)
        return result

    def _extract_characters(self, content: str) -> str:
        msgs = [{"role": "system", "content": "你是文学分析专家。"},
                {"role": "user", "content": get_prompt("book_dissect_chars", content=content[:5000])}]
        result, _ = generate_with_retry(msgs, temperature=0.5, max_tokens=4096)
        return result

    def _extract_outline(self, chapter_text: str) -> str:
        msgs = [{"role": "system", "content": "你是文学分析专家。"},
                {"role": "user", "content": get_prompt("book_dissect_outline", content=chapter_text[:3000])}]
        result, _ = generate_with_retry(msgs, temperature=0.5, max_tokens=1024)
        return result

    def _extract_foreshadowing(self, content: str) -> str:
        msgs = [{"role": "system", "content": "你是文学分析专家。"},
                {"role": "user", "content": get_prompt("book_dissect_foreshadow", content=content[:5000])}]
        result, _ = generate_with_retry(msgs, temperature=0.5, max_tokens=2048)
        return result

# ============================================================
# 导出模块（替换 Coze 文档生成，改用本地库）
# ============================================================
class NovelExporter:
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.export_dir = os.path.join(project_path, "导出")
        os.makedirs(self.export_dir, exist_ok=True)
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        path = os.path.join(self.project_path, "config.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"title": "未命名小说", "author": "AI创作"}

    def export_to_markdown(self, output_path: Optional[str] = None) -> str:
        if not output_path:
            output_path = os.path.join(self.export_dir, "全文.md")
        chapters_dir = os.path.join(self.project_path, "章节")
        parts = [f"# {self.config.get('title','')}\n\n*作者：{self.config.get('author','')}*\n\n---\n\n"]
        if os.path.exists(chapters_dir):
            for fname in sorted(os.listdir(chapters_dir)):
                if fname.endswith('.md'):
                    with open(os.path.join(chapters_dir, fname), 'r', encoding='utf-8') as f:
                        parts.append(f.read())
                        parts.append("\n\n---\n\n")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(parts))
        return output_path

    def export_to_pdf(self) -> str:
        md_path = self.export_to_markdown()
        try:
            import markdown
            from weasyprint import HTML
            with open(md_path, 'r', encoding='utf-8') as f:
                md_text = f.read()
            html = markdown.markdown(md_text)
            pdf_path = md_path.replace('.md', '.pdf')
            HTML(string=html).write_pdf(pdf_path)
            return pdf_path
        except ImportError:
            return f"PDF 生成需要安装 weasyprint 和 markdown 库，MD 已保存至 {md_path}"

    def export_to_docx(self) -> str:
        md_path = self.export_to_markdown()
        try:
            from docx import Document
            doc = Document()
            with open(md_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                doc.add_paragraph(line.strip())
            docx_path = md_path.replace('.md', '.docx')
            doc.save(docx_path)
            return docx_path
        except ImportError:
            return f"DOCX 生成需要安装 python-docx 库，MD 已保存至 {md_path}"

    def export_to_epub(self) -> str:
        return "EPUB 导出功能暂未实现，请使用 MD 导出后转换"

    def create_package(self, formats: List[str] = None) -> str:
        if formats is None:
            formats = ["md"]
        zip_name = f"{self.config.get('title','novel')}_package.zip"
        zip_path = os.path.join(self.export_dir, zip_name)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            setting_dir = os.path.join(self.project_path, "设定")
            if os.path.exists(setting_dir):
                for root, _, files in os.walk(setting_dir):
                    for file in files:
                        zf.write(os.path.join(root, file), arcname=os.path.join("设定", file))
            chapters_dir = os.path.join(self.project_path, "章节")
            if os.path.exists(chapters_dir):
                for root, _, files in os.walk(chapters_dir):
                    for file in files:
                        zf.write(os.path.join(root, file), arcname=os.path.join("章节", file))
            if "md" in formats:
                md_path = self.export_to_markdown()
                zf.write(md_path, arcname="全文.md")
        return zip_path

# ============================================================
# 回溯分支管理（未修改）
# ============================================================
class BranchManager:
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.branch_dir = os.path.join(project_path, "分支")
        os.makedirs(self.branch_dir, exist_ok=True)
        self.branches: Dict[str, Dict] = {}
        self.current_branch = "main"
        self._load()

    def _load(self):
        pass

    def create_branch(self, branch_name: str, from_chapter: int, storyline_id: str = "main"):
        branch_path = os.path.join(self.branch_dir, branch_name)
        os.makedirs(branch_path, exist_ok=True)
        self.branches[branch_name] = {
            "name": branch_name,
            "base_chapter": from_chapter,
            "storyline_id": storyline_id,
            "current_chapter": from_chapter,
            "created_at": datetime.now().isoformat()
        }
        self._save_branch_info(branch_name)
        return branch_name

    def switch_branch(self, branch_name: str):
        if branch_name not in self.branches:
            info_path = os.path.join(self.branch_dir, branch_name, "info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.branches[branch_name] = json.load(f)
            else:
                raise ValueError(f"分支 {branch_name} 不存在")
        self.current_branch = branch_name

    def get_current_chapter(self) -> int:
        if self.current_branch in self.branches:
            return self.branches[self.current_branch].get("current_chapter", 1)
        return 1

    def advance_chapter(self):
        if self.current_branch in self.branches:
            self.branches[self.current_branch]["current_chapter"] += 1
            self._save_branch_info(self.current_branch)

    def _save_branch_info(self, branch_name: str):
        branch_path = os.path.join(self.branch_dir, branch_name)
        os.makedirs(branch_path, exist_ok=True)
        with open(os.path.join(branch_path, "info.json"), 'w', encoding='utf-8') as f:
            json.dump(self.branches[branch_name], f, ensure_ascii=False, indent=2)

# ============================================================
# 配置管理（简化，仅用 DeepSeek）
# ============================================================
@dataclass
class ModelConfig:
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096

class NovelConfigManager:
    def __init__(self, path=None):
        # 全部使用 DeepSeek
        self.models = {
            "creativity": ModelConfig(model_name="deepseek-chat", temperature=0.9, max_tokens=8192),
            "generation": ModelConfig(model_name="deepseek-chat", temperature=0.8, max_tokens=8192),
            "summary": ModelConfig(model_name="deepseek-chat", temperature=0.5, max_tokens=2048)
        }
        self.assignments = {"creativity": "creativity", "generation": "generation", "summarization": "summary"}

    def get_model_for_role(self, role: str) -> Optional[ModelConfig]:
        name = self.assignments.get(role)
        return self.models.get(name) if name else None

# ============================================================
# 核心生成代理
# ============================================================
class NovelGenerationAgent:
    def __init__(self, project_path: str, genre_name: str = "玄幻"):
        self.project_path = project_path
        self.genre = GENRE_PRESETS.get(genre_name, GenreConfig(name=genre_name))
        self.config = NovelConfigManager()
        self.memory = ThreeLayerMemoryManager(project_path, self.genre)
        self.foreshadowing = ForeshadowingManager(os.path.join(project_path, "设定", "伏笔管理器.json"))
        self.style_learner = AuthorStyleLearner(project_path)
        self.brancher = BranchManager(project_path)

    def generate_chapter(self, chapter_number: int, title: str, outline: str,
                         style_cfg: Dict = None, author_style: str = None,
                         word_range: Tuple[int,int] = (3000,8000)) -> Dict:
        model, temp = self._model("generation")
        storyline = self.memory.storylines.get_current()
        ctx = self.memory.storylines.get_active_context(self.memory, self.foreshadowing, self.memory.state_updater)
        fs = self.foreshadowing.get_for_chapter(chapter_number, storyline.current_volume)
        author_inject = ""
        if author_style:
            guide = self.style_learner.get_guide(author_style)
            if guide:
                author_inject = get_prompt("author_style_inject", name=author_style, guide=guide, req="创作小说章节")
        prompt = get_prompt("chapter_content",
                            ch=chapter_number, title=title, outline=outline,
                            word_range=f"{word_range[0]}-{word_range[1]}",
                            style=get_style_requirements(style_cfg or {}),
                            context=ctx["context"][:2000],
                            chars=ctx["char_summary"][:1000],
                            foreshadow="\n".join(fs["recommendations"])[:500],
                            author_style=author_inject[:1000],
                            storyline=storyline.name)
        msgs = [{"role": "system", "content": PROMPTS["system_role"]},
                {"role": "user", "content": prompt}]
        content, ok = generate_with_retry(msgs, model, temp, 8192)
        if not ok:
            return {"success": False, "error": content}
        ch_obj = ChapterContent(chapter_number=chapter_number, chapter_title=title, content=content,
                                storyline_id=storyline.id)
        update_result, warnings = self.memory.add_chapter(ch_obj)
        if update_result.get("potential_foreshadowing"):
            self.foreshadowing.add_from_list(update_result["potential_foreshadowing"],
                                             storyline.current_volume, chapter_number)
        self.memory.storylines.advance_chapter(storyline.id)
        self.brancher.advance_chapter()
        return {"success": True, "content": content, "warnings": warnings, "chapter_number": chapter_number}

    def _model(self, role: str) -> Tuple[str, float]:
        cfg = self.config.get_model_for_role(role)
        if cfg:
            return cfg.model_name, cfg.temperature
        return "deepseek-chat", 0.8

# ============================================================
# 主控 Agent
# ============================================================
class NovelAgent:
    def __init__(self, project_path: str = None, genre: str = "玄幻"):
        self.project_path = project_path or "./novel_project"
        self.genre_name = genre
        self.genre = GENRE_PRESETS.get(genre, GenreConfig(name=genre))
        os.makedirs(self.project_path, exist_ok=True)
        self.generator = NovelGenerationAgent(self.project_path, genre)
        self.memory = self.generator.memory
        self.foreshadowing = self.generator.foreshadowing
        self.style_learner = self.generator.style_learner
        self.quality = QualityController()
        self.exporter = NovelExporter(self.project_path)
        self.dissector = BookDissector(self.project_path)
        self.brancher = self.generator.brancher

    def learn_author(self, name: str, chapters: List[str]) -> Dict:
        guide = self.style_learner.learn(name, chapters)
        return {"success": True, "guide": guide[:800]}

    def generate_chapter(self, ch: int, title: str, outline: str, **kwargs) -> Dict:
        return self.generator.generate_chapter(ch, title, outline, **kwargs)

    def polish(self, content: str, intensity: float = 0.5) -> Dict:
        polished = self.quality.polish(content, intensity)
        return {"success": True, "polished": polished}

    def analyze_rhythm(self, content: str, ch: int) -> Dict:
        stats = self.quality.analyze_rhythm(content, ch)
        issues = self.quality.check_pacing_issues()
        return {"success": True, "stats": asdict(stats), "issues": issues}

    def export(self, fmt: str = "md") -> Dict:
        if fmt == "md":
            path = self.exporter.export_to_markdown()
            return {"success": True, "filepath": path}
        elif fmt == "pdf":
            url = self.exporter.export_to_pdf()
            return {"success": True, "url": url}
        elif fmt == "docx":
            url = self.exporter.export_to_docx()
            return {"success": True, "url": url}
        elif fmt == "epub":
            path = self.exporter.export_to_epub()
            return {"success": True, "info": path}
        elif fmt == "package":
            zip_path = self.exporter.create_package()
            return {"success": True, "filepath": zip_path}
        else:
            return {"success": False, "error": f"不支持的格式: {fmt}"}

    def dissect_book(self, filepath: str, title: str = "") -> Dict:
        result = self.dissector.dissect_file(filepath, title)
        return {"success": True, "data": result}

    def create_branch(self, name: str, from_chapter: int) -> Dict:
        bid = self.brancher.create_branch(name, from_chapter)
        return {"success": True, "branch": name}

    def switch_branch(self, name: str) -> Dict:
        self.brancher.switch_branch(name)
        return {"success": True, "current_branch": name}

    def get_status(self) -> Dict:
        storyline = self.memory.storylines.get_current()
        chars = self.memory.state_updater.characters
        scenes = self.memory.state_updater.scenes
        return {
            "genre": self.genre.name,
            "current_storyline": storyline.name,
            "current_chapter": storyline.current_chapter,
            "active_characters": list(chars.keys()),
            "scenes": list(scenes.keys()),
            "pending_foreshadowing": len(self.foreshadowing.get_pending())
        }

def get_style_requirements(cfg: Dict) -> str:
    maps = {
        "style_level": {0: "极简", 5: "适中", 10: "华丽"},
        "dialogue_ratio": {0: "少对话", 5: "对话叙述平衡", 10: "多对话"},
        "description_density": {0: "轻描写", 5: "中等描写", 10: "重描写"},
        "pacing_tendency": {0: "快节奏", 5: "中等节奏", 10: "慢节奏"}
    }
    parts = []
    for k, mp in maps.items():
        val = cfg.get(k, 5)
        parts.append(f"{k}: {mp.get(val, '适中')}")
    return "; ".join(parts)