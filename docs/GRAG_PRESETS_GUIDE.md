# 🎚️ GRAG Presets Guide - 20 Intensity Levels

> **Node:** `ArchAi3D GRAG Modifier`
> **Version:** 2.2.1
> **Total Presets:** 20 intensity levels (+ Custom mode)
> **Purpose:** Simple intensity-based GRAG control from minimal to maximum

---

## 📋 Quick Reference Table

All presets use **equal values** for Strength, Lambda (λ), and Delta (δ) to provide straightforward intensity control.

| Preset Name | Strength | Lambda (λ) | Delta (δ) | Description |
|-------------|----------|------------|-----------|-------------|
| **Custom** | User | User | User | Manual control - adjust parameters independently |
| **Level 01 - Minimal** | 0.40 | 0.40 | 0.40 | Minimal effect - 40% intensity |
| **Level 02** | 0.48 | 0.48 | 0.48 | Very low effect - 48% intensity |
| **Level 03** | 0.56 | 0.56 | 0.56 | Low effect - 56% intensity |
| **Level 04** | 0.64 | 0.64 | 0.64 | Below moderate - 64% intensity |
| **Level 05** | 0.72 | 0.72 | 0.72 | Moderate low - 72% intensity |
| **Level 06** | 0.80 | 0.80 | 0.80 | Moderate - 80% intensity |
| **Level 07** | 0.88 | 0.88 | 0.88 | Moderate high - 88% intensity |
| **Level 08** | 0.96 | 0.96 | 0.96 | Nearly neutral - 96% intensity |
| **Level 09** | 1.04 | 1.04 | 1.04 | Just above neutral - 104% intensity |
| **Level 10 - Balanced** ⭐ | 1.12 | 1.12 | 1.12 | **Recommended start** - 112% intensity |
| **Level 11** | 1.20 | 1.20 | 1.20 | Above balanced - 120% intensity |
| **Level 12** | 1.28 | 1.28 | 1.28 | Strong low - 128% intensity |
| **Level 13** | 1.36 | 1.36 | 1.36 | Strong - 136% intensity |
| **Level 14** | 1.44 | 1.44 | 1.44 | Strong high - 144% intensity |
| **Level 15** | 1.52 | 1.52 | 1.52 | Very strong - 152% intensity |
| **Level 16** | 1.60 | 1.60 | 1.60 | Very strong high - 160% intensity |
| **Level 17** | 1.68 | 1.68 | 1.68 | Intense - 168% intensity |
| **Level 18** | 1.76 | 1.76 | 1.76 | Very intense - 176% intensity |
| **Level 19** | 1.84 | 1.84 | 1.84 | Near maximum - 184% intensity |
| **Level 20 - Maximum** | 2.00 | 2.00 | 2.00 | Maximum effect - 200% intensity |

---

## 🎯 Understanding the Intensity Levels

### How It Works:

The preset system provides **20 intensity levels** that control how strongly GRAG modifies the attention mechanism during image generation.

- **All three parameters move together** (Strength, Lambda, Delta)
- **Linear progression** from 0.40 to 2.00 in 0.08 increments
- **Simple mental model:** Higher number = stronger effect

### Parameter Ranges Explained:

```
0.40 (Level 01) ────────── 1.00 (neutral) ────────── 2.00 (Level 20)
    ↑                           ↑                          ↑
  Minimal                    No change               Maximum
  suppression                 (baseline)            amplification
```

### What Happens at Each Range:

**Levels 01-08 (0.40-0.96):** Below neutral
- Reduces GRAG effect compared to baseline
- More conservative transformations
- Better structure preservation
- Use for: Subtle refinements, keeping original features

**Level 09 (1.04):** Just above neutral
- Minimal visible change from baseline
- Testing zone to verify GRAG is working

**Level 10 (1.12) - Recommended Start ⭐**
- Visible but balanced effects
- Good starting point for most use cases
- Clear demonstration of GRAG capabilities

**Levels 11-15 (1.20-1.52):** Strong effects
- Clear visible transformations
- Good control and predictability
- Use for: Material changes, style modifications

**Levels 16-20 (1.60-2.00):** Maximum intensity
- Dramatic transformations
- May produce unexpected results
- Use for: Experimentation, creative exploration

---

## 💡 How to Choose a Preset

### Decision Flow:

```
START HERE
    |
    ├─ First time using GRAG?
    |   └─ Start with Level 10 (Balanced) ⭐
    |
    ├─ Need subtle changes?
    |   ├─ Very subtle → Level 05-07 (0.72-0.88)
    |   └─ Moderate → Level 08-09 (0.96-1.04)
    |
    ├─ Need visible transformation?
    |   ├─ Clear but controlled → Level 10-12 (1.12-1.28)
    |   └─ Strong changes → Level 13-15 (1.36-1.52)
    |
    ├─ Want maximum effect?
    |   ├─ Very strong → Level 16-18 (1.60-1.76)
    |   └─ Extreme → Level 19-20 (1.84-2.00)
    |
    └─ Want custom control?
        └─ Select "Custom" and adjust manually
```

### Quick Selection Guide:

| Your Goal | Recommended Level | Why |
|-----------|------------------|-----|
| First test of GRAG | **Level 10** | Balanced, visible effects |
| Preserve structure | Level 05-07 | Conservative changes |
| Remove scaffolding | Level 10-13 | Clear transformation |
| Change materials | Level 11-14 | Strong but controlled |
| Complete redesign | Level 15-18 | Dramatic changes |
| Maximum creativity | Level 19-20 | Extreme experimentation |
| Test if GRAG works | Level 10 | Clear visibility check |

---

## 🎓 Usage Tips

### Testing Strategy:

**Method 1: Find Your Sweet Spot**
```
1. Start with Level 10 (Balanced)
2. If too subtle → Try Level 13
3. If too strong → Try Level 07
4. Narrow down by ±2 levels
5. Fine-tune with Custom mode if needed
```

**Method 2: Range Testing**
```
Same seed, same prompt, test 5 levels:
- Level 05 (subtle)
- Level 10 (balanced)
- Level 15 (strong)
- Level 18 (very strong)
- Level 20 (maximum)

Compare results, pick your favorite range
```

**Method 3: A/B Comparison**
```
Run two generations side-by-side:
- Generation A: Level 08 (below neutral)
- Generation B: Level 12 (above neutral)

See the difference, adjust accordingly
```

### For Clean Room Workflow:

**Recommended Testing Sequence:**

1. **Level 10 (Balanced)** - Start here to see if GRAG is working
2. **Level 08 (Nearly neutral)** - If Level 10 changes too much
3. **Level 13 (Strong)** - If Level 10 is too subtle
4. **Fine-tune** - Once you find the right range, try ±1 level

**Expected Behavior:**
- **Levels 05-08:** Should preserve windows better
- **Levels 10-13:** Clear scaffolding removal, some window changes possible
- **Levels 15+:** Strong transformation, test carefully

---

## 🔧 Custom Mode

### When to Use Custom:

✅ You found your ideal level (e.g., Level 12) and want slight adjustments
✅ You want different values for Strength vs Lambda vs Delta
✅ Testing specific parameter combinations for research
✅ Fine-tuning between two preset levels

### Custom Workflow:

1. **Select "Custom" preset**
2. **Adjust three sliders independently:**
   - `grag_strength`: Master intensity (0.1-2.0) - Currently stored, not applied
   - `grag_cond_b` (λ): Bias control (0.1-2.0) - Main parameter
   - `grag_cond_delta` (δ): Deviation control (0.1-2.0) - Main parameter
3. **Test and iterate**

### Custom Examples:

**Example 1: Between Level 10 and Level 11**
```
grag_strength: 1.16
grag_cond_b: 1.16
grag_cond_delta: 1.16
(Halfway between 1.12 and 1.20)
```

**Example 2: Asymmetric Parameters**
```
grag_strength: 1.0
grag_cond_b: 0.80  (reduce bias)
grag_cond_delta: 1.50  (amplify deviations)
(For window preservation with material changes)
```

**Example 3: Extreme Testing**
```
grag_strength: 1.0
grag_cond_b: 0.10  (minimum bias)
grag_cond_delta: 2.00  (maximum deviation)
(Testing parameter extremes)
```

---

## 📈 Parameter Effect Guide

### Understanding the GRAG Formula:

```
k̂ = λ * k_mean + δ * (k - k_mean)

Where:
- k = original attention keys
- k_mean = group average (bias)
- λ = lambda (grag_cond_b)
- δ = delta (grag_cond_delta)
- k̂ = reweighted keys
```

### What Each Parameter Does:

**Lambda (λ) - Bias Strength:**
- **0.1-0.8:** Reduces shared patterns (more variety, less consistency)
- **1.0:** Neutral (no change to bias component)
- **1.2-2.0:** Enhances shared patterns (more consistency, less variety)

**Delta (δ) - Deviation Intensity:**
- **0.1-0.8:** Suppresses token differences (smoother, more uniform)
- **1.0:** Neutral (no change to deviation component)
- **1.2-2.0:** Amplifies token differences (more variation, more details)

**Strength - Overall Multiplier:**
- **Note:** As of v2.2.1, this parameter is stored but NOT applied to formula
- **Future use:** May control overall GRAG intensity multiplier
- **Current behavior:** Has no mathematical effect

### Critical Understanding:

**At λ=1.0, δ=1.0:**
```
k̂ = 1.0 * k_mean + 1.0 * (k - k_mean)
  = k_mean + k - k_mean
  = k  (NO CHANGE!)
```

**This is why neutral (1.0, 1.0) produces no visible effect!**

---

## ⚠️ Important Notes

### Mathematical Ranges:

- **Testing range:** λ=0.1-2.0, δ=0.1-2.0 (expanded for experimentation)
- **Paper's stable range:** λ=0.95-1.15, δ=0.95-1.15 (conservative, subtle effects)
- **Visible effect range:** λ=0.4-2.0, δ=0.4-2.0 (our preset system)

### Common Issues:

**Problem:** Preset has no effect
**Solution:**
1. Make sure you're using **GRAG Sampler** (not standard KSampler)
2. Check that "enable_grag" is set to True in GRAG Modifier
3. Try Level 13 or higher for more obvious effects

**Problem:** All levels look the same
**Solution:**
1. Verify GRAG Sampler console shows "Patched 60 Attention layers"
2. Try extreme comparison: Level 05 vs Level 18
3. Use same seed for both tests

**Problem:** Even Level 20 is too subtle
**Solution:**
1. Switch to Custom mode
2. Try extreme asymmetric: λ=0.1, δ=2.0
3. Verify your workflow is correct: [Encoder] → [GRAG Modifier] → [GRAG Sampler]

**Problem:** Low levels (01-05) produce artifacts
**Solution:**
1. This is expected at extreme suppression (<0.6)
2. Try Level 06 or higher
3. Use Clean Artifacts workflow if needed

### Performance Notes:

- All presets have the same computational cost
- GRAG adds ~5-10% overhead to sampling time
- No difference in speed between Level 01 and Level 20

---

## 🎯 Use Case Recommendations

| Your Task | Start Here | If Too Subtle | If Too Strong |
|-----------|------------|---------------|---------------|
| First GRAG test | Level 10 | Level 13 | Level 07 |
| Remove scaffolding | Level 10 | Level 12 | Level 08 |
| Change materials | Level 11 | Level 14 | Level 09 |
| Preserve windows | Level 06 | Level 08 | Level 05 |
| Complete redesign | Level 15 | Level 18 | Level 12 |
| Subtle refinement | Level 07 | Level 09 | Level 05 |
| Maximum creativity | Level 18 | Level 20 | Level 15 |

---

## 📊 Preset Progression Examples

### Visual Progression (Conceptual):

```
Level 01 (0.40): [||||                    ] Minimal
Level 05 (0.72): [|||||||||||             ] Moderate low
Level 10 (1.12): [||||||||||||||||||      ] Balanced ⭐
Level 15 (1.52): [||||||||||||||||||||||||] Very strong
Level 20 (2.00): [||||||||||||||||||||||||||] Maximum
```

### Expected Effect Progression:

**Scaffolding Removal Scenario:**

| Level | Scaffolding | Windows | Materials | Overall |
|-------|-------------|---------|-----------|---------|
| 05 | Slightly faded | Fully intact | Unchanged | Very conservative |
| 10 | Mostly removed | Mostly intact | Some change | **Recommended** |
| 15 | Completely gone | May change | Strong change | Dramatic |
| 20 | Gone | Likely changed | Very different | Extreme |

**Material Change Scenario:**

| Level | Structure | Old Material | New Material | Quality |
|-------|-----------|--------------|--------------|---------|
| 05 | Perfect | Mostly visible | Subtle hints | Conservative |
| 10 | Excellent | Fading | Emerging | **Recommended** |
| 15 | Good | Gone | Strong | Dramatic |
| 20 | May shift | Gone | Very strong | Experimental |

---

## 📚 Related Documentation

- [GRAG Modifier Guide](./GRAG_MODIFIER_GUIDE.md) - Main node documentation
- [GRAG Integration Summary](../../E:\Comfy\help\my-work\development\comfyui\GRAG_INTEGRATION_SUMMARY.md) - Technical details
- [Clean Room Prompt Guide](./CLEAN_ROOM_PROMPT_GUIDE.md) - Your primary workflow

---

## 🚀 Quick Start

**Never used GRAG before? Follow these steps:**

1. **Enable GRAG in your workflow:**
   ```
   [Images] → [Encoder] → [GRAG Modifier] → [GRAG Sampler] → [Output]
   ```

2. **In GRAG Modifier node:**
   - Set `enable_grag` to **True**
   - Select `preset`: **Level 10 - Balanced**
   - Leave other parameters at default

3. **Generate and observe:**
   - Note the visual changes compared to baseline
   - Console should show: "Patched 60 Attention layers"

4. **Adjust intensity:**
   - Too subtle? → Try Level 13
   - Too strong? → Try Level 07
   - Just right? → Stay at Level 10

5. **Fine-tune if needed:**
   - Switch to "Custom" preset
   - Copy your favorite level's values
   - Adjust in 0.05 increments

---

**Version:** 2.2.1
**Last Updated:** 2025-11-03
**Total Presets:** 20 intensity levels + Custom mode
**Author:** Amir Ferdos (ArchAi3d)

**Preset Design:** Simple intensity-based system where all parameters move together proportionally from 0.40 (minimal) to 2.00 (maximum) for straightforward control.

Enjoy experimenting with all 20 intensity levels! 🎉
