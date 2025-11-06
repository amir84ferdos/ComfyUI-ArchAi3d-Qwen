"""
Test script to verify auto_facing feature works correctly in Cinematography Prompt Builder
"""

import sys
sys.path.insert(0, r"E:\Comfy\Qwen\ComfyUI-Easy-Install\ComfyUI\custom_nodes\ComfyUI-ArchAi3d-Qwen")

from nodes.camera.cinematography_prompt_builder import ArchAi3D_Cinematography_Prompt_Builder

# Initialize node
node = ArchAi3D_Cinematography_Prompt_Builder()

print("=" * 80)
print("AUTO_FACING FEATURE TEST - Cinematography Prompt Builder")
print("=" * 80)

# Test 1: Front View (0°) - auto_facing should NOT appear (redundant)
print("\n" + "=" * 80)
print("TEST 1: Front View (0°) with auto_facing=True")
print("EXPECTED: NO 'Facing' clause (front view already implies facing)")
print("=" * 80)

simple1, prof1, sys1, desc1 = node.generate_cinematography_prompt(
    target_subject="the refrigerator",
    shot_type="Full Shot (FS)",
    camera_angle="Eye Level",
    depth_of_field="Auto (based on shot size)",
    style_mood="Natural/Neutral",
    prompt_language="Chinese (Best for dx8152 LoRAs)",
    horizontal_angle="Front View (0°)",
    auto_facing=True
)

print(f"\nProfessional Prompt:\n{prof1}")
print(f"\n✅ PASS" if "面对" not in prof1 and "Facing" not in prof1 else "❌ FAIL: Should NOT have facing clause")

# Test 2: Angled Left 30° - auto_facing SHOULD appear
print("\n" + "=" * 80)
print("TEST 2: Angled Left 30° with auto_facing=True")
print("EXPECTED: '面对the refrigerator' at the BEGINNING")
print("=" * 80)

simple2, prof2, sys2, desc2 = node.generate_cinematography_prompt(
    target_subject="the refrigerator",
    shot_type="Full Shot (FS)",
    camera_angle="Eye Level",
    depth_of_field="Auto (based on shot size)",
    style_mood="Natural/Neutral",
    prompt_language="Chinese (Best for dx8152 LoRAs)",
    horizontal_angle="Angled Left 30°",
    auto_facing=True
)

print(f"\nProfessional Prompt:\n{prof2}")
print(f"\n✅ PASS" if prof2.startswith("面对the refrigerator") else "❌ FAIL: Should start with '面对the refrigerator'")

# Test 3: Side Right (90°) with auto_facing=True - SHOULD appear
print("\n" + "=" * 80)
print("TEST 3: Side Right (90°) with auto_facing=True")
print("EXPECTED: '面对the refrigerator' at the BEGINNING")
print("=" * 80)

simple3, prof3, sys3, desc3 = node.generate_cinematography_prompt(
    target_subject="the refrigerator",
    shot_type="Medium Shot (MS)",
    camera_angle="Eye Level",
    depth_of_field="Auto (based on shot size)",
    style_mood="Natural/Neutral",
    prompt_language="Chinese (Best for dx8152 LoRAs)",
    horizontal_angle="Side Right (90°)",
    auto_facing=True
)

print(f"\nProfessional Prompt:\n{prof3}")
print(f"\n✅ PASS" if prof3.startswith("面对the refrigerator") else "❌ FAIL: Should start with '面对the refrigerator'")

# Test 4: Angled Right 45° with auto_facing=False - should NOT appear
print("\n" + "=" * 80)
print("TEST 4: Angled Right 45° with auto_facing=False")
print("EXPECTED: NO 'Facing' clause (disabled by user)")
print("=" * 80)

simple4, prof4, sys4, desc4 = node.generate_cinematography_prompt(
    target_subject="the refrigerator",
    shot_type="Medium Shot (MS)",
    camera_angle="Eye Level",
    depth_of_field="Auto (based on shot size)",
    style_mood="Natural/Neutral",
    prompt_language="Chinese (Best for dx8152 LoRAs)",
    horizontal_angle="Angled Right 45°",
    auto_facing=False
)

print(f"\nProfessional Prompt:\n{prof4}")
print(f"\n✅ PASS" if "面对" not in prof4 and "Facing" not in prof4 else "❌ FAIL: Should NOT have facing clause (disabled)")

# Test 5: English mode with Angled Left 45°
print("\n" + "=" * 80)
print("TEST 5: Angled Left 45° with auto_facing=True (English mode)")
print("EXPECTED: 'Facing the refrigerator directly' at the BEGINNING")
print("=" * 80)

simple5, prof5, sys5, desc5 = node.generate_cinematography_prompt(
    target_subject="the refrigerator",
    shot_type="Medium Shot (MS)",
    camera_angle="Eye Level",
    depth_of_field="Auto (based on shot size)",
    style_mood="Natural/Neutral",
    prompt_language="English (Simple & Clear)",
    horizontal_angle="Angled Left 45°",
    auto_facing=True
)

print(f"\nProfessional Prompt:\n{prof5}")
print(f"\nSimple Prompt:\n{simple5}")
print(f"\n✅ PASS" if prof5.startswith("Facing the refrigerator directly") and simple5.startswith("Facing the refrigerator directly") else "❌ FAIL: Should start with 'Facing the refrigerator directly'")

# Test 6: Hybrid mode with Side Left (90°)
print("\n" + "=" * 80)
print("TEST 6: Side Left (90°) with auto_facing=True (Hybrid mode)")
print("EXPECTED: '面对the refrigerator' at the BEGINNING")
print("=" * 80)

simple6, prof6, sys6, desc6 = node.generate_cinematography_prompt(
    target_subject="the refrigerator",
    shot_type="Close-Up (CU)",
    camera_angle="Eye Level",
    depth_of_field="Auto (based on shot size)",
    style_mood="Natural/Neutral",
    prompt_language="Hybrid (Chinese + English)",
    horizontal_angle="Side Left (90°)",
    auto_facing=True
)

print(f"\nProfessional Prompt:\n{prof6}")
print(f"\n✅ PASS" if prof6.startswith("面对the refrigerator") else "❌ FAIL: Should start with '面对the refrigerator'")

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("All tests should show ✅ PASS")
print("If any show ❌ FAIL, the auto_facing feature needs debugging")
print("=" * 80)
