#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

print("Testing ALL module imports...")

modules_to_test = [
    'scoring_engine',
    'leak_detector', 
    'js_analyzer',
    'js_extractor',
    'subdomain_finder',
    'reporter',
    'domain_manager',
    'crawler',
    'ai_analyzer',
    'directory_fuzzer',
    'utils',
    'constants',
    'logger',
    'tech_fingerprinter'
]

for module_name in modules_to_test:
    try:
        module_path = f"modules/{module_name}.py"
        if not os.path.exists(module_path):
            print(f"❌ {module_name}: File not found")
            continue
            
        # Try to import the module
        module = __import__(f'modules.{module_name}', fromlist=[''])
        print(f"✅ {module_name}: SUCCESS")
        
    except SyntaxError as e:
        print(f"❌ {module_name}: SYNTAX ERROR - {e}")
    except Exception as e:
        print(f"❌ {module_name}: {e}")
