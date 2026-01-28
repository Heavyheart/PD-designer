import streamlit as st
import math
from dataclasses import dataclass
import pandas as pd

# ==========================================
# 1. 核心逻辑 (原始算法)  streamlit run single_joint_PD_design.py 
# ==========================================

@dataclass
class PDDesignResult:
    # 输入
    J: float
    f_n_des: float          # 期望自然频率 (Hz)
    zeta_des: float         # 期望阻尼比
    kp_max: float
    kd_max: float

    # 理论（无约束）设计值
    omega_n_des: float      # 期望自然角频率 (rad/s)
    kp_des: float           # 理论 Kp
    kd_des: float           # 理论 Kd

    # 约束后的实际 Kp, Kd
    kp_actual: float
    kd_actual: float

    # 约束后的实际闭环指标
    omega_n_actual: float   # 实际自然角频率 (rad/s)
    f_n_actual: float       # 实际自然频率 (Hz)
    zeta_actual: float      # 实际阻尼比
    t_r_actual: float       # 实际上升时间近似 (s)
    t_s_actual: float       # 实际调节时间近似 (s)

def design_pd_with_limits(
    J: float,
    f_n_des: float,
    zeta_des: float,
    kp_max: float = 500.0,
    kd_max: float = 5.0,
) -> PDDesignResult:
    """
    计算 PD 参数的核心函数
    """
    if J <= 0:
        return None # 简单的错误处理

    # 1) 期望的自然角频率
    omega_n_des = 2.0 * math.pi * f_n_des

    # 2) 理论（无约束）PD 设计值
    kp_des = J * (omega_n_des ** 2)
    kd_des = 2.0 * zeta_des * J * omega_n_des

    # 3) 计算满足约束条件且保持阻尼比的最大 omega_n
    omega_n_limits = [omega_n_des]

    # Kp 限制
    if kp_max is not None and kp_max > 0:
        omega_n_limit_kp = math.sqrt(kp_max / J)
        if omega_n_limit_kp > 0:
            omega_n_limits.append(omega_n_limit_kp)

    # Kd 限制
    if kd_max is not None and kd_max > 0 and zeta_des > 0:
        omega_n_limit_kd = kd_max / (2.0 * zeta_des * J)
        if omega_n_limit_kd > 0:
            omega_n_limits.append(omega_n_limit_kd)

    omega_n_actual = min(omega_n_limits)

    # 4) 计算实际 Kp, Kd
    kp_actual = J * (omega_n_actual ** 2)
    kd_actual = 2.0 * zeta_des * J * omega_n_actual

    # 5) 计算实际闭环指标
    f_n_actual = omega_n_actual / (2.0 * math.pi)
    
    # 反算 Zeta (应该等于 zeta_des，除非被极值卡住导致无法维持，但此算法优先维持 zeta)
    zeta_actual = 0.0
    if kp_actual > 0:
        zeta_actual = kd_actual / (2.0 * math.sqrt(J * kp_actual))

    t_r_actual = float("inf")
    t_s_actual = float("inf")
    if omega_n_actual > 0:
        t_r_actual = 1.8 / omega_n_actual
        if zeta_actual > 0:
            t_s_actual = 4.0 / (zeta_actual * omega_n_actual)

    return PDDesignResult(
        J=J,
        f_n_des=f_n_des,
        zeta_des=zeta_des,
        kp_max=kp_max,
        kd_max=kd_max,
        omega_n_des=omega_n_des,
        kp_des=kp_des,
        kd_des=kd_des,
        kp_actual=kp_actual,
        kd_actual=kd_actual,
        omega_n_actual=omega_n_actual,
        f_n_actual=f_n_actual,
        zeta_actual=zeta_actual,
        t_r_actual=t_r_actual,
        t_s_actual=t_s_actual,
    )

# ==========================================
# 2. 预定义数据 (电机型号)
# ==========================================

ACTUATORS = {
    "4310": 0.0231825,
    "4315": 0.0415820,
    "6408": 0.0390686,
    "8112": 0.0596423,
    "8116": 0.0753178
}

# ==========================================
# 3. Streamlit 网页界面构建
# ==========================================

def main():
    st.set_page_config(page_title="关节PD参数计算器", layout="wide")
    
    st.title("🤖 EnCos单关节执行器 PD 参数设计工具")
    st.markdown("选择电机型号，输入期望的控制性能，计算受约束后的最佳 Kp/Kd 参数。")
    st.markdown("---")

    # --- 左侧边栏：参数配置 ---
    with st.sidebar:
        st.header("⚙️ 参数配置")
        
        # 1. 选择电机
        st.subheader("1. 执行器型号")
        selected_model = st.selectbox(
            "选择电机型号",
            options=list(ACTUATORS.keys()),
            index=2 # 默认选中间那个
        )
        
        # 自动获取惯量，但也允许微调
        default_J = ACTUATORS[selected_model]
        J_input = st.number_input(
            "转动惯量 J (kg·m²)", 
            value=default_J, 
            format="%.7f",
            help="根据所选型号自动填充，可手动修改"
        )

        # 2. 期望目标
        st.subheader("2. 期望性能指标")
        f_n_input = st.number_input(
            "期望自然频率 (Hz)", 
            value=12.0, 
            step=0.5,
            help="系统响应的快慢，频率越高响应越快"
        )
        zeta_input = st.number_input(
            "期望阻尼比 Zeta", 
            value=1.5, 
            step=0.1,
            help="<1 为欠阻尼(有超调)，=1 临界阻尼，>1 过阻尼(无超调)"
        )

        # 3. 硬件约束
        st.subheader("3. Encos软件约束")
        kp_max_input = st.number_input("最大 Kp 限制", value=500.0)
        kd_max_input = st.number_input("最大 Kd 限制", value=5.0)

    # --- 主界面：计算与显示 ---
    
    # 调用核心计算函数
    res = design_pd_with_limits(
        J=J_input,
        f_n_des=f_n_input,
        zeta_des=zeta_input,
        kp_max=kp_max_input,
        kd_max=kd_max_input
    )

    if res:
        # 检查是否触发了限制
        is_limited = (res.kp_actual < res.kp_des) or (res.kd_actual < res.kd_des)

        # --- 第一行：核心结果大字显示 ---
        st.subheader("🚀 计算结果")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="推荐 Kp", 
                value=f"{res.kp_actual:.3f}",
                delta=f"理论: {res.kp_des:.3f}" if is_limited else None,
                delta_color="inverse"
            )
        with col2:
            st.metric(
                label="推荐 Kd", 
                value=f"{res.kd_actual:.3f}",
                delta=f"理论: {res.kd_des:.3f}" if is_limited else None,
                delta_color="inverse"
            )
        with col3:
            st.metric(
                label="实际自然频率 (Hz)", 
                value=f"{res.f_n_actual:.2f}",
                delta=f"期望: {res.f_n_des:.2f}" if is_limited else None,
                delta_color="inverse"
            )
        with col4:
            st.metric(
                label="上升时间 (ms)", 
                value=f"{res.t_r_actual * 1000:.1f}",
                help="系统从 0 上升到目标值附近所需的时间 (1.8/ωn)"
            )

        if is_limited:
            st.warning(f"⚠️ 注意：由于 Kp 或 Kd 的最大值限制，系统无法达到期望的 {res.f_n_des} Hz。已自动降频至 {res.f_n_actual:.2f} Hz 以保持阻尼特性。")
        else:
            st.success("✅ 当前硬件约束下，可以完美满足期望性能。")

        st.markdown("---")

        # --- 第二行：详细数据表格对比 ---
        st.subheader("📊 详细参数对比表")
        
        # 构建对比数据
        data = {
            "参数指标": [
                "自然频率 (Hz)", 
                "自然角频率 (rad/s)", 
                "阻尼比 (Zeta)", 
                "比例增益 Kp", 
                "微分增益 Kd",
                "上升时间 (ms)",
                "调节时间 (ms)"
            ],
            "理论无约束值 (期望值)": [
                f"{res.f_n_des:.3f}",
                f"{res.omega_n_des:.3f}",
                f"{res.zeta_des:.3f}",
                f"{res.kp_des:.3f}",
                f"{res.kd_des:.3f}",
                f"{(1.8/res.omega_n_des)*1000:.1f}",
                f"{(4.0/(res.zeta_des*res.omega_n_des))*1000:.1f}"
            ],
            "实际约束值 (推荐值)": [
                f"{res.f_n_actual:.3f}",
                f"{res.omega_n_actual:.3f}",
                f"{res.zeta_actual:.3f}",
                f"{res.kp_actual:.3f}",
                f"{res.kd_actual:.3f}",
                f"{res.t_r_actual*1000:.1f}",
                f"{res.t_s_actual*1000:.1f}"
            ]
        }
        
        df = pd.DataFrame(data)
        st.table(df)

        # --- 补充说明 ---
        with st.expander("ℹ️ 查看公式与说明"):
            st.markdown(r"""
            **计算公式：**
            * $J$ = 转动惯量 (来自电机型号)
            * $\omega_n = 2 \pi f_n$
            * $K_p = J \cdot \omega_n^2$
            * $K_d = 2 \zeta J \cdot \omega_n$
            * $t_r \approx 1.8 / \omega_n$ (上升时间)
            * $t_s \approx 4.0 / (\zeta \omega_n)$ (调节时间, 2% 误差带)
            
            **逻辑说明：**
            如果计算出的 $K_p$ 或 $K_d$ 超过了设定的最大值，程序会**优先保持阻尼比 $\zeta$ 不变**，
            通过降低自然频率 $\omega_n$ 来满足约束。这保证了系统不会因为饱和而产生不可预期的震荡，
            代价是响应速度变慢。
            """)

if __name__ == "__main__":
    main()