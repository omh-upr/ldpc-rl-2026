#!/usr/bin/env python3
"""
Deprecated legacy entry point.

This script has been retired from the active pipeline.
Use scripts/run_expert_experience.py instead.
"""

import sys


def main() -> int:
    print("scripts/generate_expert_experiences.py is deprecated.")
    print("   Use: PYTHONPATH=. python scripts/run_expert_experience.py --config <expert_experience_config.yml>")
    print("   Legacy implementation archived at: deprecated/scripts/generate_expert_experiences.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
