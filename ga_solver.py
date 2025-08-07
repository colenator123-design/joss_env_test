import random
import numpy as np
from deap import base, creator, tools, algorithms
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduling_environment.job import Job
from scheduling_environment.operation import Operation
from scheduling_environment.machine import Machine
from scheduling_environment.jobShop import JobShop
from visualization import gantt_with_pm

# ====================================================================
# 1. 參數設定
# ====================================================================
# 生產排程：(job_id, op_id, machine_id, start_time, end_time)
# GA 不會改變這個排程，只會根據它來排 PM
production_schedule = [
    (0, 0, 0, 0, 10), (0, 1, 1, 10, 25), (0, 2, 0, 30, 45),
    (1, 0, 1, 0, 10), (1, 1, 2, 10, 20), (1, 2, 1, 30, 40),
    (2, 0, 2, 0, 10), (2, 1, 0, 10, 25), (2, 2, 2, 25, 35),
]
num_machines = 3

# 來自第一階段的動態參數
dynamic_params = {
    'utilization': [0.4, 0.35, 0.3], # 機器利用率 (U_k)
    'switch_cost_factor': [2, 2, 2], # 作業切換次數 (S_k)
    'alpha_uk': [0.1, 0.15, 0.1],     # 利用率影響因子 (α_uk)
    'alpha_sk': [0.05, 0.05, 0.04]     # 切換影響因子 (α_sk)
}

# 維護與成本參數
cost_params = {
    'Cp_k': [100, 120, 110],     # PM 固定成本 (C_pk)
    'Cf_k': [1000, 1100, 1050],  # 故障成本 (C_fk)
    'Tp_k': [5, 6, 5],           # PM 維護時間 (T_pk)
    'w_makespan_cost': 1.0,      # makespan 權重
    'beta': [2.0, 2.5, 2.0],     # Weibull 分布的形狀參數 (β)
    'eta_0': [50.0, 60.0, 55.0], # Weibull 分布的初始尺度參數 (η₀)
}

# ====================================================================
# 2. 核心函數
# ====================================================================
def get_failure_rate_integral(start_t, end_t, machine_id, dynamic_params, cost_params):
    """
    計算動態 Weibull 分布在特定時間區間的累積故障風險。
    """
    # 根據機器ID取出對應的參數
    uk = dynamic_params['utilization'][machine_id]
    sk = dynamic_params['switch_cost_factor'][machine_id]
    alpha_uk = dynamic_params['alpha_uk'][machine_id]
    alpha_sk = dynamic_params['alpha_sk'][machine_id]
    
    beta = cost_params['beta'][machine_id]
    eta_0 = cost_params['eta_0'][machine_id]

    # 根據論文公式計算動態尺度參數 η_rk
    eta_dynamic = eta_0 / (1 + alpha_uk * uk + alpha_sk * sk)
    
    if eta_dynamic <= 0:
        return float('inf')

    # 計算 Weibull 分布的累積故障率積分
    integral_end = (end_t / eta_dynamic)**beta if end_t >= 0 else 0
    integral_start = (start_t / eta_dynamic)**beta if start_t >= 0 else 0
    
    return integral_end - integral_start

# 適應度函數
def evaluate_schedule_revised(individual):
    pm_starts = individual
    total_cost = 0
    
    # 1. 計算PM固定成本
    total_cost += sum(cost_params['Cp_k'])

    machine_ops = {k: [] for k in range(num_machines)}
    for job_info in production_schedule:
        machine_id = job_info[2]
        machine_ops[machine_id].append(job_info)

    machine_end_times = [0] * num_machines
    
    for k in range(num_machines):
        pm_start = pm_starts[k]
        pm_duration = cost_params['Tp_k'][k]
        pm_end = pm_start + pm_duration

        # 檢查PM與生產作業是否有重疊 (硬性約束)
        is_overlap = False
        ops_on_machine = sorted(machine_ops[k], key=lambda x: x[3])
        for op_info in ops_on_machine:
            op_start, op_end = op_info[3], op_info[4]
            if max(pm_start, op_start) < min(pm_end, op_end): # 有重疊
                is_overlap = True
                break
        
        # 若有重疊，則給予巨大懲罰
        if is_overlap:
            return 1000000, 1000000 # 返回一個巨大的 makespan 懲罰

        # 2. 計算PM對排程造成的延遲
        machine_production_end = 0
        current_time_on_machine = 0 # 追蹤機器上當前時間
        
        for op_info in ops_on_machine:
            op_start, op_end = op_info[3], op_info[4]
            op_duration = op_end - op_start

            # 如果作業在 PM 之前完成，則不受影響
            if op_end <= pm_start:
                current_time_on_machine = max(current_time_on_machine, op_end)
            # 如果作業與 PM 重疊或在 PM 之後開始，則其開始時間會被 PM 結束時間推遲
            else:
                new_op_start = max(op_start, pm_end) 
                new_op_end = new_op_start + op_duration
                current_time_on_machine = new_op_end
            
            machine_production_end = max(machine_production_end, current_time_on_machine)

        machine_end_times[k] = max(machine_production_end, pm_end) # 機器最終結束時間

        # 3. 根據動態Weibull模型計算預期故障成本
        # 這裡的邏輯是：PM前的運轉時間 + PM後的運轉時間
        # 假設 PM 發生後，故障率會重置
        
        # 計算 PM 前的累積故障風險
        risk_before_pm = get_failure_rate_integral(0, pm_start, k, dynamic_params, cost_params)
        
        # 計算 PM 後的累積故障風險
        # 這裡的 '0' 表示 PM 後時鐘重置，從 0 開始計算運轉時間
        # 'machine_end_times[k] - pm_end' 是 PM 結束後，機器實際運轉的時長
        risk_after_pm = get_failure_rate_integral(0, max(0, machine_end_times[k] - pm_end), k, dynamic_params, cost_params)
        
        failure_cost_for_machine_k = (
            risk_before_pm * cost_params['Cf_k'][k] +
            risk_after_pm * cost_params['Cf_k'][k]
        )
        total_cost += failure_cost_for_machine_k
        
    # 計算最終makespan，並將其成本納入總成本
    current_makespan = max(machine_end_times)
    total_cost += current_makespan * cost_params['w_makespan_cost']

    return total_cost, current_makespan # 返回總成本和最終makespan

# ====================================================================
# 3. DEAP 相關設定
# ====================================================================
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) # 最小化成本和makespan
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# GA 變數的生成範圍
max_possible_pm_start = max(op[4] for op in production_schedule) + max(cost_params['Tp_k']) + 20 # 留一些餘裕
toolbox.register("attr_pm_start", random.randint, 0, max_possible_pm_start)

# 個體生成器：一個包含 num_machines 個 PM 開始時間的列表
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_pm_start, n=num_machines)

# 族群生成器
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 註冊遺傳操作
toolbox.register("evaluate", evaluate_schedule_revised)
# 交叉操作：兩點交叉
toolbox.register("mate", tools.cxTwoPoint)
# 變異操作：隨機改變個體中的一個基因
toolbox.register("mutate", tools.mutUniformInt, low=0, up=max_possible_pm_start, indpb=0.1)
# 選擇操作：錦標賽選擇
toolbox.register("select", tools.selTournament, tournsize=3)

# ====================================================================
# 4. 執行遺傳演算法
# ====================================================================
def main():
    random.seed(42) # 為了結果的可重現性

    population = toolbox.population(n=50) # 族群大小
    ngen = 100 # 迭代代數
    cxpb = 0.7 # 交叉概率
    mutpb = 0.2 # 變異概率

    # 執行簡單遺傳演算法
    # with_logbook=True 可以獲取演化過程的詳細日誌
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                        stats=None, halloffame=None, verbose=True)

    best_individual = tools.selBest(population, 1)[0]
    best_cost, final_makespan_from_eval = toolbox.evaluate(best_individual) # 獲取最佳個體的成本和makespan

    print(f"\n最佳 PM 排程: {best_individual}")
    print(f"最低總成本: {best_cost}")
    print(f"最終 Makespan: {final_makespan_from_eval}")

    # ====================================================================
    # 5. 視覺化最佳解
    # ====================================================================
    job_shop = JobShop()
    for i in range(num_machines):
        job_shop.add_machine(Machine(machine_id=i))
    
    # 創建 Job 物件並添加到 job_shop
    num_jobs = max(op[0] for op in production_schedule) + 1
    jobs = [Job(job_id=i) for i in range(num_jobs)] # 這裡創建 Job 物件
    for job_obj in jobs:
        job_shop.add_job(job_obj)

    for job_info in production_schedule:
        job_id, op_id, machine_id, start_time, end_time = job_info
        op = Operation(jobs[job_id], job_id, op_id) # 使用正確的 Job 物件
        op.add_operation_scheduling_information(machine_id, start_time, 0, end_time - start_time)
        job_shop.machines[machine_id]._processed_operations.append(op)

    pm_schedule = []
    for k in range(num_machines):
        pm_start_time = best_individual[k]
        pm_duration = cost_params['Tp_k'][k]
        pm_schedule.append((k, pm_start_time, pm_duration))
    
    # 使用從 evaluate_schedule_revised 獲取的 makespan
    plt = gantt_with_pm.plot(job_shop, pm_schedule)
    plt.xlim(0, final_makespan_from_eval + 5) 
    plt.savefig('gantt_with_pm_ga.png')
    print("生成甘特圖: gantt_with_pm_ga.png")

if __name__ == '__main__':
    main()
