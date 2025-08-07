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
# 1. 參數動態生成函數
# ====================================================================

def generate_random_job_shop_schedule(num_jobs, num_machines, max_duration=15):
    """
    生成一個隨機但可行的 Job Shop 排程。
    """
    production_schedule = []
    machine_end_times = [0] * num_machines
    job_end_times = [0] * num_jobs
    
    for j in range(num_jobs):
        op_sequence = list(range(num_machines))
        random.shuffle(op_sequence)
        
        for o, m in enumerate(op_sequence):
            op_duration = random.randint(5, max_duration)
            
            start_time = max(machine_end_times[m], job_end_times[j])
            end_time = start_time + op_duration
            
            production_schedule.append((j, o, m, start_time, end_time))
            machine_end_times[m] = end_time
            job_end_times[j] = end_time
            
    production_schedule.sort(key=lambda x: x[3])
    return production_schedule

def generate_random_params(num_machines):
    """
    根據機器數量生成隨機的成本與動態參數。
    """
    dynamic_params = {
        'utilization': [random.uniform(0.3, 0.6) for _ in range(num_machines)],
        'switch_cost_factor': [random.randint(1, 3) for _ in range(num_machines)],
        'alpha_uk': [random.uniform(0.1, 0.2) for _ in range(num_machines)],
        'alpha_sk': [random.uniform(0.05, 0.1) for _ in range(num_machines)]
    }

    cost_params = {
        'Cp_k': [random.randint(80, 150) for _ in range(num_machines)],
        'Cf_k': [random.randint(800, 1500) for _ in range(num_machines)],
        'Tp_k': [random.randint(4, 8) for _ in range(num_machines)],
        'w_makespan_cost': 1.0,
        'beta': [random.uniform(1.5, 3.0) for _ in range(num_machines)],
        'eta_0': [random.uniform(40.0, 70.0) for _ in range(num_machines)],
    }
    return dynamic_params, cost_params

# ====================================================================
# 2. 核心函數
# ====================================================================
def get_failure_rate_integral(start_t, end_t, machine_id, dynamic_params, cost_params):
    """
    計算動態 Weibull 分布在特定時間區間的累積故障風險。
    """
    uk = dynamic_params['utilization'][machine_id]
    sk = dynamic_params['switch_cost_factor'][machine_id]
    alpha_uk = dynamic_params['alpha_uk'][machine_id]
    alpha_sk = dynamic_params['alpha_sk'][machine_id]
    
    beta = cost_params['beta'][machine_id]
    eta_0 = cost_params['eta_0'][machine_id]

    eta_dynamic = eta_0 / (1 + alpha_uk * uk + alpha_sk * sk)
    
    if eta_dynamic <= 0:
        return float('inf')

    integral_end = (end_t / eta_dynamic)**beta if end_t >= 0 else 0
    integral_start = (start_t / eta_dynamic)**beta if start_t >= 0 else 0
    
    return integral_end - integral_start

def evaluate_schedule_revised(individual, production_schedule, num_machines, dynamic_params, cost_params):
    pm_starts = individual
    total_cost = 0
    
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

        is_overlap = False
        ops_on_machine = sorted(machine_ops[k], key=lambda x: x[3])
        for op_info in ops_on_machine:
            op_start, op_end = op_info[3], op_info[4]
            if max(pm_start, op_start) < min(pm_end, op_end):
                is_overlap = True
                break
        
        if is_overlap:
            return 1000000, 1000000 # 返回一個巨大的 makespan 懲罰

        machine_production_end = 0
        current_time = 0
        
        for op_info in ops_on_machine:
            op_start, op_end = op_info[3], op_info[4]
            op_duration = op_end - op_start

            if op_end <= pm_start:
                current_time = max(current_time, op_end)
            else:
                new_op_start = max(op_start, pm_end) 
                new_op_end = new_op_start + op_duration
                current_time = new_op_end
            
            machine_production_end = max(machine_production_end, current_time)

        machine_end_times[k] = max(machine_production_end, pm_end)

        failure_risk_before_pm = get_failure_rate_integral(0, pm_start, k, dynamic_params, cost_params)
        failure_risk_after_pm = get_failure_rate_integral(0, max(0, machine_end_times[k] - pm_end), k, dynamic_params, cost_params)
        
        failure_cost_for_machine_k = (
            failure_risk_before_pm * cost_params['Cf_k'][k] +
            failure_risk_after_pm * cost_params['Cf_k'][k]
        )
        total_cost += failure_cost_for_machine_k
        
    current_makespan = max(machine_end_times)
    total_cost += current_makespan * cost_params['w_makespan_cost']

    return total_cost, current_makespan

# ====================================================================
# 3. DEAP 相關設定
# ====================================================================
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0)) # 最小化成本和makespan
creator.create("Individual", list, fitness=creator.FitnessMin)

def create_toolbox(num_machines, max_pm_start):
    toolbox = base.Toolbox()
    toolbox.register("attr_pm_start", random.randint, 0, int(max_pm_start))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_pm_start, n=num_machines)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=int(max_pm_start), indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

# ====================================================================
# 4. 主函數
# ====================================================================
def main():
    random.seed(42)

    # === 在這裡設定你的實驗規模 ===
    NUM_JOBS = 10
    NUM_MACHINES = 5
    POPULATION_SIZE = 100
    NGEN = 200
    # ==============================

    print(f"Generating a problem with {NUM_JOBS} jobs and {NUM_MACHINES} machines...")

    production_schedule = generate_random_job_shop_schedule(NUM_JOBS, NUM_MACHINES)
    dynamic_params, cost_params = generate_random_params(NUM_MACHINES)

    max_production_time = max(op[4] for op in production_schedule)
    max_pm_start = max_production_time + max(cost_params['Tp_k']) + 20
    
    toolbox = create_toolbox(NUM_MACHINES, max_pm_start)
    toolbox.register("evaluate", evaluate_schedule_revised, production_schedule=production_schedule, num_machines=NUM_MACHINES, dynamic_params=dynamic_params, cost_params=cost_params)

    population = toolbox.population(n=POPULATION_SIZE)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=NGEN, 
                        stats=None, halloffame=None, verbose=True)

    best_individual = tools.selBest(population, 1)[0]
    best_cost, final_makespan_from_eval = toolbox.evaluate(best_individual) # 獲取最佳個體的成本和makespan

    print(f"\n最佳 PM 排程: {best_individual}")
    print(f"最低總成本: {best_cost}")
    print(f"最終 Makespan: {final_makespan_from_eval}")

    # 可視化 (只在小規模時使用，以免圖表過於擁擠)
    if NUM_JOBS <= 10 and NUM_MACHINES <= 10:
        job_shop = JobShop()
        for i in range(NUM_MACHINES):
            job_shop.add_machine(Machine(machine_id=i))
        
        # 創建 Job 物件並添加到 job_shop
        num_jobs_vis = max(op[0] for op in production_schedule) + 1
        jobs_vis = [Job(job_id=i) for i in range(num_jobs_vis)] 
        for job_obj in jobs_vis:
            job_shop.add_job(job_obj)

        for job_info in production_schedule:
            job_id, op_id, machine_id, start_time, end_time = job_info
            op = Operation(jobs_vis[job_id], job_id, op_id) 
            op.add_operation_scheduling_information(machine_id, start_time, 0, end_time - start_time)
            job_shop.machines[machine_id]._processed_operations.append(op)
        
        pm_schedule = []
        for k in range(NUM_MACHINES):
            pm_start_time = best_individual[k]
            pm_duration = cost_params['Tp_k'][k]
            pm_schedule.append((k, pm_start_time, pm_duration))
        
        final_makespan_vis = max(max(op[4] for op in production_schedule), max(pm[1] + pm[2] for pm in pm_schedule))
        
        plt = gantt_with_pm.plot(job_shop, pm_schedule)
        plt.xlim(0, final_makespan_vis + 5) 
        plt.savefig('gantt_with_pm_ga.png')
        print("生成甘特圖: gantt_with_pm_ga.png")
    else:
        print("問題規模過大，跳過甘特圖可視化。")

if __name__ == '__main__':
    main()
