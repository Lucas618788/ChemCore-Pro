import sys
sys.path.append("/usr/lib/python3/dist-packages")

import streamlit as st
from rdkit import Chem  
import streamlit as st
import json
import re
from openai import OpenAI
import chromadb
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import py3Dmol
from stmol import showmol

# ==========================================
# 1. 页面基本配置与 Apple 风格 CSS
# ==========================================
st.set_page_config(page_title="ChemCore Pro", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #f5f5f7; color: #1d1d1f;
    }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e8e8ed !important; }
    .stButton>button { background-color: #0071e3 !important; color: white !important; border-radius: 8px !important; border: none !important; transition: all 0.3s ease !important; }
    .stButton>button:hover { background-color: #0077ed !important; box-shadow: 0 4px 12px rgba(0,113,227,0.3) !important; }
    [data-testid="stChatMessage"] { background-color: #ffffff !important; border-radius: 18px !important; padding: 24px !important; margin-bottom: 16px !important; border: 1px solid #e8e8ed !important; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02) !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    header {background: rgba(255,255,255,0.8) !important; backdrop-filter: blur(20px) !important;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 初始化本地数据库与 Session
# ==========================================
db_client = chromadb.PersistentClient(path="./chem_db")
db_collection = db_client.get_or_create_collection(name="reactions")

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "saved_sessions" not in st.session_state: st.session_state.saved_sessions = {}
if "current_mode" not in st.session_state: st.session_state.current_mode = "正向机理推演"
if "matched_smiles" not in st.session_state: st.session_state.matched_smiles = None


# ==========================================
# 3. 核心大模型请求 (带安全标签与 DOI 标签提取)
# ==========================================
def query_chemical_agent(query, mode, api_key, history=[]):
    results = db_collection.query(query_texts=[query], n_results=1)
    distance = results['distances'][0][0] if results['distances'][0] else 999

    rag_context = ""
    if distance < 1.5:
        matched_name = results['metadatas'][0][0]['name']
        matched_smiles = results['metadatas'][0][0]['smiles_list']
        st.session_state.matched_smiles = json.loads(matched_smiles) if isinstance(matched_smiles,
                                                                                   str) else matched_smiles
        rag_context = f"\n[标准知识库匹配] 涉及反应: {matched_name}，SMILES: {st.session_state.matched_smiles}。"
    else:
        st.session_state.matched_smiles = None

    # 🌟 新增：强制要求大模型输出 <SAFETY> 和 <DOI> 标签的 Prompt
    base_prompt = f"""
    {rag_context}
    你是一个顶级的有机化学专家。当前模式：{mode}。
    【重要输出规则】：
    1. 安全预警：如果涉及到剧毒、爆炸、强腐蚀性试剂，请务必用 <SAFETY>具体的警告内容和防护建议</SAFETY> 标签包裹。如果没有高危风险可省略。
    2. 文献溯源：请务必在回答末尾，用 <DOI>经典的JACS/Angew/Nature/Science文献DOI链接或引用格式</DOI> 标签包裹输出。
    """

    messages = [{"role": "system", "content": base_prompt}]
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["agent_raw"]})  # 传递原始文本给大模型
    messages.append({"role": "user", "content": query})

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(model="deepseek-chat", messages=messages, temperature=0.2)
        return response.choices[0].message.content
    except Exception as e:
        return f"API 请求失败: {str(e)}"


# ==========================================
# 4. 辅助函数：将 SMILES 转换为 3D 可视化
# ==========================================
def render_3d_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # 加上氢原子
        AllChem.EmbedMolecule(mol, randomSeed=42)  # 生成 3D 坐标
        mblock = Chem.MolToMolBlock(mol)

        view = py3Dmol.view(width=400, height=350)
        view.addModel(mblock, 'mol')
        view.setStyle({'stick': {}})  # 使用棍状模型显示
        view.zoomTo()
        showmol(view, height=350, width=400)
    except Exception as e:
        st.warning(f"3D 模型渲染失败: {e}")


# ==========================================
# 5. UI 侧边栏与头部
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='font-weight:600; color:#1d1d1f; margin-bottom:20px;'>控制中心</h2>", unsafe_allow_html=True)
    api_key = st.text_input("DeepSeek API Key", type="password", value="")
    st.markdown("<hr style='border-top: 1px solid #e8e8ed;'>", unsafe_allow_html=True)
    st.session_state.current_mode = st.radio("模式选择", ["正向机理推演", "逆合成分析助推", "智能试剂配比计算"],
                                             label_visibility="collapsed")
    st.markdown("<hr style='border-top: 1px solid #e8e8ed;'>", unsafe_allow_html=True)
    if st.button("➕ 开启新实验"):
        st.session_state.chat_history = []
        st.session_state.matched_smiles = None
        st.rerun()
    for session_title in st.session_state.saved_sessions.keys():
        if st.button(f"📄 {session_title[:12]}...", key=f"btn_{session_title}"):
            st.session_state.chat_history = st.session_state.saved_sessions[session_title]
            st.rerun()

st.markdown(
    f"<div style='margin-top:20px; margin-bottom:30px;'><h1 style='font-size:42px; font-weight:700; color:#1d1d1f;'>{st.session_state.current_mode}.</h1><p style='color:#86868b;'>让百大反应真理库，成为你的本能。</p></div>",
    unsafe_allow_html=True)


# ==========================================
# 6. 对话与结果渲染逻辑
# ==========================================
def render_message(role, raw_content):
    """解析标签并美化输出"""
    if role == "user":
        st.markdown(f"<div style='color:#1d1d1f; font-size:16px;'>{raw_content}</div>", unsafe_allow_html=True)
        return

    # 正则提取标签内容
    safety_match = re.search(r'<SAFETY>(.*?)</SAFETY>', raw_content, re.DOTALL)
    doi_match = re.search(r'<DOI>(.*?)</DOI>', raw_content, re.DOTALL)

    # 清理掉标签用于展示正文
    clean_reply = re.sub(r'<SAFETY>.*?</SAFETY>', '', raw_content, flags=re.DOTALL)
    clean_reply = re.sub(r'<DOI>.*?</DOI>', '', clean_reply, flags=re.DOTALL)

    # 🌟 特效 1：安全预警红框
    if safety_match:
        st.error(f"**⚠️ 实验室安全预警**\n\n{safety_match.group(1).strip()}")

    st.markdown(f"<div style='color:#1d1d1f; font-size:16px; line-height:1.6;'>{clean_reply.strip()}</div>",
                unsafe_allow_html=True)

    # 🌟 特效 2：文献溯源蓝框
    if doi_match:
        st.info(f"**📚 真实文献溯源**\n\n{doi_match.group(1).strip()}")


# 渲染历史对话
for msg in st.session_state.chat_history:
    with st.chat_message("user"): render_message("user", msg['user'])
    with st.chat_message("assistant"): render_message("assistant", msg['agent_raw'])

# 🌟 特效 3：左 2D + 右 3D 双列绘图视窗
if st.session_state.matched_smiles and len(st.session_state.matched_smiles) > 0:
    st.markdown(
        "<div style='background-color:#ffffff; padding:15px; border-radius:18px; border:1px solid #e8e8ed; margin-bottom:20px;'><p style='font-size:14px; font-weight:600; color:#0071e3; margin-bottom:5px;'>🔮 多维分子结构视窗</p></div>",
        unsafe_allow_html=True)

    steps = len(st.session_state.matched_smiles)
    current_step = st.slider("拖动滑块演变分子结构：", 1, steps, 1)
    current_smiles = st.session_state.matched_smiles[current_step - 1]

    col1, col2 = st.columns(2)
    with col1:
        st.caption("平面的 2D 骨架图 (RDKit)")
        mol_2d = Chem.MolFromSmiles(current_smiles)
        if mol_2d:
            img = Draw.MolToImage(mol_2d, size=(400, 350))
            st.image(img, use_container_width=True)
    with col2:
        st.caption("可交互的 3D 立体构象 (拖拽旋转)")
        render_3d_molecule(current_smiles)

# ==========================================
# 7. 底部输入交互
# ==========================================
if user_input := st.chat_input("输入你要推演的反应或分子..."):
    if not api_key:
        st.warning("请先在左侧输入 API Key！")
    else:
        with st.chat_message("user"):
            render_message("user", user_input)

        with st.chat_message("assistant"):
            with st.spinner("AI 正在解析多维数据并调取文献..."):
                raw_reply = query_chemical_agent(user_input, st.session_state.current_mode, api_key,
                                                 st.session_state.chat_history)
                render_message("assistant", raw_reply)

        st.session_state.chat_history.append({"user": user_input, "agent_raw": raw_reply})
        if user_input not in st.session_state.saved_sessions:
            st.session_state.saved_sessions[user_input] = st.session_state.chat_history
            st.rerun()
