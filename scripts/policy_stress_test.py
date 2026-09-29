"""
ClarifySign - Backward Compatibility Module for scripts/policy_stress_test.py
Delegates to evaluation.run_policy_evaluation.
"""

from evaluation.run_policy_evaluation import main

if __name__ == "__main__":
    main()
