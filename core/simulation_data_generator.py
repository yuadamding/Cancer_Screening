# simulation_data_generator.py
"""
High-cohesion data generation module for hereditary cancer families.
Generates pedigrees, covariates, HMM trajectories, and false-negative outcomes.
Low coupling: no dependencies on RL code; pure data pipeline.
"""
import numpy as np
import pandas as pd

def build_3gen_family(rng, min_size=30, max_size=40, carrier_prob_first_gen=0.1):
    members = []
    next_id = 0
    # Generation 1
    for _ in range(2):
        members.append({
            'id': next_id,
            'gen': 1,
            'parents': None,
            'genotype': rng.binomial(1, carrier_prob_first_gen)
        })
        next_id += 1
    # Generation 2
    grand = members[:2]
    n2 = rng.integers(8, 13)
    gen2_ids = []
    for _ in range(n2):
        child_g = inherit_genotype(grand[0]['genotype'], grand[1]['genotype'], rng)
        members.append({'id': next_id, 'gen': 2,
                        'parents': (grand[0]['id'], grand[1]['id']),
                        'genotype': child_g})
        gen2_ids.append(next_id)
        next_id += 1
    # Generation 3
    for pid in gen2_ids:
        partner_g = rng.binomial(1, carrier_prob_first_gen)
        for _ in range(rng.integers(0,5)):
            p = next(m for m in members if m['id']==pid)
            child_g = inherit_genotype(p['genotype'], partner_g, rng)
            members.append({'id': next_id, 'gen': 3,
                            'parents': (pid, None), 'genotype': child_g})
            next_id += 1
    # Adjust size
    cur = len(members)
    if cur<min_size:
        for _ in range(min_size-cur):
            pid = rng.choice(gen2_ids)
            partner_g = rng.binomial(1, carrier_prob_first_gen)
            p = next(m for m in members if m['id']==pid)
            members.append({'id': next_id, 'gen':3,
                            'parents':(pid,None),
                            'genotype': inherit_genotype(p['genotype'], partner_g, rng)})
            next_id+=1
    elif cur>max_size:
        excess = cur-max_size
        gen3 = [m for m in members if m['gen']==3]
        remove = rng.choice(gen3, size=excess, replace=False)
        rem_ids = {m['id'] for m in remove}
        members = [m for m in members if m['id'] not in rem_ids]
    return sorted(members, key=lambda m: m['id'])


def inherit_genotype(g1, g2, rng):
    return int(rng.binomial(1, 0.5)) if (g1 or g2) else 0


def assign_covariates(members, rng):
    for m in members:
        m['covariates'] = rng.binomial(1, [0.3,0.5]).tolist() + \
                            rng.integers(0,3, size=3).tolist()


def compute_rate(k, genotype, cov, lambda_0, beta_geno, beta_cov):
    linear = beta_geno[k]*genotype + np.dot(beta_cov[k], cov)
    return lambda_0[k] * np.exp(linear)

def discrete_p(rate):
    return 1 - np.exp(-rate)

def simulate_stage_trajectory(genotype, cov, T, M, lambda_0, beta_geno, beta_cov, rng):
    X = np.zeros(T+1, int)
    s=0
    for t in range(T):
        if s<M and rng.random() < discrete_p(compute_rate(s, genotype, cov, lambda_0, beta_geno, beta_cov)):
            s+=1
        X[t+1]=min(s, M)
    return X


def stage_outcome(stage, f_neg, rng):
    return 0 if stage==0 else int(rng.random() > f_neg.get(stage,0.2))


def generate_data(
    family_id=0, T=20, M=3,
    lambda_0=None, beta_geno=None, beta_cov=None,
    f_neg=None, min_size=30, max_size=40, seed=None
):
    rng = np.random.default_rng(seed)
    # Defaults
    lambda_0 = lambda_0 or [0.02,0.03,0.04]
    beta_geno = beta_geno or [0.5,0.8,1.0]
    beta_cov = beta_cov or [[0.1,-0.2,0.05,-0.05,0.1],
                             [0.2,0.1,0.0,0.1,-0.1],
                             [0.3,0.2,-0.1,0.0,0.2]]
    f_neg = f_neg or {1:0.2,2:0.1,3:0.0}
    members = build_3gen_family(rng, min_size, max_size)
    assign_covariates(members, rng)
    rows=[]
    for m in members:
        X = simulate_stage_trajectory(
            m['genotype'], m['covariates'], T, M,
            lambda_0, beta_geno, beta_cov, rng
        )
        for t in range(T+1):
            rows.append({
                'family_id':family_id,
                'individual_id':m['id'],
                'genotype':m['genotype'],
                **{f'cov_{i}':c for i,c in enumerate(m['covariates'])},
                'time':t,
                'stage':int(X[t]),
                'test':stage_outcome(int(X[t]), f_neg, rng)
            })
    df = pd.DataFrame(rows)
    return df.sort_values(['individual_id','time']).reset_index(drop=True)

if __name__=='__main__':
    df = generate_data(seed=42)
    print(df.head())
