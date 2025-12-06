# src/metrics.py

def confidence_drop(original_conf, adv_conf):
    return float(original_conf - adv_conf)

def attack_success(orig_label, adv_label):
    return int(orig_label != adv_label)

def robustness_score(original_conf, adv_conf):
    diff = original_conf - adv_conf
    return max(0, 1 - diff)
