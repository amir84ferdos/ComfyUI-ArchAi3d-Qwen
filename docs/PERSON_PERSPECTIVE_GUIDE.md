# Person Perspective Control Guide

**Complete guide to using ArchAi3D Qwen Person Perspective node for professional character photography**

Based on community research from Reddit r/StableDiffusion tutorial by Vortexneonlight

---

## Table of Contents

1. [Overview](#overview)
2. [Key Differences: Person vs Object Rotation](#key-differences-person-vs-object-rotation)
3. [The 6 Perspective Presets](#the-6-perspective-presets)
4. [Identity Preservation](#identity-preservation)
5. [Psychological Effects](#psychological-effects)
6. [Advanced Controls](#advanced-controls)
7. [Workflow Examples](#workflow-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [System Prompt Recommendations](#system-prompt-recommendations)

---

## Overview

### What This Node Does

The **ArchAi3D Qwen Person Perspective** node is specialized for changing camera perspectives when photographing **people and characters**. Unlike object rotation (which orbits the camera around a subject), this node changes the **camera viewing angle** while keeping the person's **identity intact**.

### Primary Focus: Identity Preservation

The #1 goal is to preserve the subject's:
- Facial features
- Clothing and styling
- Hairstyle
- Pose and body position
- All identifying characteristics

Only the **camera perspective** changes - the person stays the same!

### Use Cases

Perfect for:
- Portrait photography with different angles
- Fashion photography with dramatic perspectives
- Character concept art and turnarounds
- Editorial photography
- Heroic/power poses with low angles
- Vulnerable/intimate poses with high angles
- Character design consistency studies

---

## Key Differences: Person vs Object Rotation

Understanding the difference between these two approaches is critical:

### Person Perspective (This Node)

**What it does:**
- Changes the **camera viewing angle** relative to a person
- Person stays in place, camera moves up/down/around
- Focus on **identity preservation**
- Psychological effects through angle choice

**Example prompt:**
```
"Rotate the angle of the photo to a high angle shot (bird's eye view) of the subject,
with the camera positioned above and looking directly down. Important: keep the subject's
face, clothes, hairstyle, and pose identical."
```

**Best for:**
- Portraits
- Fashion photography
- Character art
- People-focused shots

### Object Rotation

**What it does:**
- **Orbits the camera** around an object/building
- Shows different sides of the object
- Object can be anything (product, building, furniture)

**Example prompt:**
```
"camera orbit right around the building by 90 degrees"
```

**Best for:**
- Product visualization
- Architectural walkarounds
- 360° turntables
- Object showcases

### Visual Comparison

```
PERSON PERSPECTIVE:
    High Angle (Bird's Eye)
           ↓ camera looks down
        [PERSON]  ← stays in same pose
           ↑
    Low Angle (Worm's Eye)

OBJECT ROTATION:
    Front View              Side View
    [OBJECT]    → camera   [OBJECT]
                  orbits
                  around
```

---

## The 6 Perspective Presets

### 1. High Angle (Bird's Eye View)

**Description:** Looking down at the subject from above

**Camera Position:** Positioned above, looking directly down

**Psychological Effect:** Creates sense of vulnerability, isolation, or intimacy

**Body Focus:** Head, chest, and surrounding environment

**Body Proportions:** Height diminished, upper body foreshortened

**When to Use:**
- Intimate or vulnerable character moments
- Fashion shots emphasizing outfit from above
- Establishing vulnerability in narrative
- Showing environmental context around character

**Example Prompt Output:**
```
Rotate the angle of the photo to a high angle shot (bird's eye view) of the subject
with the camera positioned above and looking directly down. The perspective should
diminish the subject's height and create a sense of vulnerability or isolation,
prominently showcasing the details of the head, chest, and surrounding environment
while the rest of the body is foreshortened but visible. Important: Keep the
subject's id, clothes, facial features, pose, and hairstyle identical.
```

---

### 2. Low Angle (Worm's Eye View)

**Description:** Looking up at the subject from below

**Camera Position:** Positioned very close to the legs, looking up

**Psychological Effect:** Creates sense of power, monumentality, heroism

**Body Focus:** Legs and thighs (foreground), body rises dramatically

**Body Proportions:** Height exaggerated, lower body prominent

**When to Use:**
- Heroic character poses
- Power and authority portraits
- Fashion shots with dramatic impact
- Action character concepts
- Emphasizing strength or dominance

**Example Prompt Output:**
```
Rotate the angle of the photo to an ultra-low angle shot with the camera positioned
very close to the legs looking up. The perspective should exaggerate the subject's
height and create a sense of monumentality and power, prominently showcasing the
details of the legs and thighs while the upper body dramatically rises towards up.
Important: Keep the subject's id, clothes, facial features, pose, and hairstyle identical.
```

---

### 3. Eye Level Front View

**Description:** Standard straight-on portrait

**Camera Position:** Directly in front at eye level

**Psychological Effect:** Balanced, neutral, approachable

**Body Focus:** Face and upper body

**Body Proportions:** Natural, undistorted

**When to Use:**
- Standard portraits
- Professional headshots
- Character reference sheets
- Approachable, friendly character shots
- Baseline view for comparison

**Example Prompt Output:**
```
Rotate the angle of the photo to an eye level shot with the camera positioned
directly in front at eye level. The perspective should create a balanced and
neutral perspective, showcasing the face and upper body maintaining natural
body proportions. Important: Keep the subject's id, clothes, facial features,
pose, and hairstyle identical.
```

---

### 4. Side Profile

**Description:** Full side view showing profile

**Camera Position:** At eye level, perpendicular to subject

**Psychological Effect:** Creates distance, showcases silhouette

**Body Focus:** Side profile from head to toe

**Body Proportions:** Natural, emphasizes body line

**When to Use:**
- Fashion silhouette shots
- Character design profiles
- Architectural body lines
- Formal portraits
- Showing outfit details from side

**Example Prompt Output:**
```
Rotate the angle of the photo to a direct side angle shot with the camera at
eye level perpendicular to the subject. The perspective should clearly showcase
the entire side profile, showcasing the side profile from head to toe maintaining
natural proportions. Important: Keep the subject's id, clothes, facial features,
pose, and hairstyle identical.
```

---

### 5. Three-Quarter View

**Description:** Slight angle showing face and partial side

**Camera Position:** At eye level, approximately 45 degrees from front

**Psychological Effect:** Creates depth while remaining approachable

**Body Focus:** Face, shoulder, and partial side

**Body Proportions:** Natural with dimensional depth

**When to Use:**
- Classic portrait photography
- Fashion photography
- Character concept art
- Editorial photography
- Most versatile perspective

**Example Prompt Output:**
```
Rotate the angle of the photo to a three-quarter angle shot with the camera at
eye level at approximately 45 degrees from front. The perspective should create
depth while showing facial features, showcasing the face, shoulder, and partial
side natural proportions with dimensional depth. Important: Keep the subject's
id, clothes, facial features, pose, and hairstyle identical.
```

---

### 6. Dutch Angle (Tilted)

**Description:** Camera tilted 30-45 degrees

**Camera Position:** At eye level but camera tilted

**Psychological Effect:** Creates tension, drama, unease, visual interest

**Body Focus:** Face and upper body in tilted frame

**Body Proportions:** Diagonal composition

**When to Use:**
- Dynamic action shots
- Dramatic editorial photography
- Unsettling or tense character moments
- Artistic/creative portraits
- High-energy fashion shots

**Example Prompt Output:**
```
Rotate the angle of the photo to a tilted angle shot (dutch angle) with the
camera at eye level but camera tilted approximately 30-45 degrees. The perspective
should create tension, drama, or visual interest, showcasing the face and upper
body in tilted frame diagonal composition. Important: Keep the subject's id,
clothes, facial features, pose, and hairstyle identical.
```

---

## Identity Preservation

### The 4 Preservation Levels

Identity preservation is the **most critical feature** of this node. It determines how strictly the person's appearance is maintained.

#### None
- No identity preservation instructions
- Model may change appearance
- **Not recommended** for person photography

#### Loose
**Instruction:** "Keep the subject recognizable"

**What it preserves:**
- General appearance
- Basic facial features
- Similar outfit type

**Use when:**
- You want creative freedom
- Style variation is acceptable
- General likeness is enough

---

#### Moderate (Default for most cases)
**Instruction:** "Maintain the subject's appearance and clothing"

**What it preserves:**
- Facial features
- Exact clothing
- General pose
- Hairstyle

**Use when:**
- Standard portrait work
- Fashion photography
- Most character art

---

#### Strict (Recommended)
**Instruction:** "Keep the subject's id, clothes, facial features, pose, and hairstyle identical"

**What it preserves:**
- Every facial detail
- Exact clothing with all details
- Precise pose and body position
- Exact hairstyle and styling
- All accessories and details

**Use when:**
- Character turnarounds
- Consistent character views
- Professional portrait series
- Fashion lookbook photography
- Any work requiring exact consistency

**This is the DEFAULT and RECOMMENDED setting.**

---

### Why Identity Preservation Matters

Without strict identity preservation, the model might:
- Change facial features
- Alter clothing style or colors
- Modify hairstyle
- Change accessories
- Shift body position
- Transform the overall character

With strict preservation, only the **camera angle** changes while everything else stays identical.

---

## Psychological Effects

Camera angles convey powerful psychological messages. Use these to enhance your character's story.

### Vulnerability
**Best Angles:** High angle, bird's eye view
**Effect:** Subject appears smaller, isolated, vulnerable
**Use for:** Intimate moments, emotional vulnerability, isolation

---

### Power
**Best Angles:** Low angle, worm's eye view
**Effect:** Subject appears larger, dominant, imposing
**Use for:** Heroic poses, authority figures, powerful characters

---

### Intimacy
**Best Angles:** High angle, close perspective
**Effect:** Creates closeness, personal connection
**Use for:** Portrait intimacy, emotional connection, personal moments

---

### Grandeur
**Best Angles:** Low angle, wide shot
**Effect:** Subject appears monumental, impressive
**Use for:** Epic character moments, architectural fashion shots

---

### Confidence
**Best Angles:** Eye level, slight low angle
**Effect:** Balanced with hint of strength
**Use for:** Professional portraits, confident characters

---

### Mystery
**Best Angles:** Side profile, dutch angle
**Effect:** Concealment, intrigue, artistic distance
**Use for:** Mysterious characters, editorial fashion, artistic portraits

---

### Choosing Effects

You can override the preset's default psychological effect with these controls. For example:
- Use high angle preset with "power" effect for creative tension
- Use low angle with "intimacy" for unusual emotional impact
- Combine effects for complex character portrayal

---

## Advanced Controls

### Focal Point Selection

Control which body part the camera emphasizes:

#### Auto (Recommended)
- Lets the preset decide the optimal focal point
- Each preset has a carefully chosen default

#### Head & Chest
- Upper body focus
- Best for: Portrait emphasis, facial expressions

#### Legs & Thighs
- Lower body focus
- Best for: Fashion, low-angle power shots

#### Full Body
- Equal emphasis on entire figure
- Best for: Character sheets, fashion full-body shots

#### Face & Upper Body
- Face and shoulders
- Best for: Classic portraits, headshots

#### Torso
- Midsection emphasis
- Best for: Fashion, body-focused shots

#### Lower Body & Legs
- Legs and feet
- Best for: Shoe/fashion details, grounded shots

---

### Body Proportion Guidance

Control how body proportions are adjusted for the perspective:

#### Natural (Default)
- Realistic proportions for the angle
- Recommended for most uses

#### Exaggerate Height
- Make subject appear taller
- Best with: Low angles for heroic effect

#### Diminish Height
- Make subject appear shorter
- Best with: High angles for vulnerability

#### Foreshorten Upper
- Compress upper body visually
- Best with: High angles

#### Foreshorten Lower
- Compress lower body visually
- Best with: Low angles

#### Elongate
- Stretch the figure
- Best with: Fashion photography, artistic shots

---

### Background Adaptation

**Default:** Enabled (Recommended)

When enabled, instructs the model to adjust background elements to complement the new perspective.

**Example:**
- High angle → Background elements appear from above
- Low angle → Background shows more sky/ceiling
- Side angle → Background shows depth

**Disable when:** You want minimal background changes

---

### Lighting Reinforcement

**Default:** Enabled (Recommended)

Ensures lighting adjusts to reinforce the perspective effect naturally.

**Example:**
- High angle → Light may come from above
- Low angle → Dramatic uplighting
- Side angle → Rim lighting from side

**Disable when:** Lighting must stay exactly as-is

---

### Composition Centering

**Default:** Enabled (Recommended)

Keeps the subject centered in frame for consistency.

**Why important:** Centered subjects work best with perspective changes. Off-center subjects can cause identity drift.

**Disable when:** You specifically want off-center composition

---

### Detail Showcase Level

Controls the emphasis on detail in focal areas:

#### High (Default)
- "Prominently showcasing the details of..."
- Maximum detail emphasis
- Best for: Professional work, high-quality output

#### Medium
- "Clearly showcasing the..."
- Balanced detail
- Best for: General use

#### Low
- "Showcasing the..."
- Subtle emphasis
- Best for: Minimalist style

---

### Prompt Style

Controls the verbosity and detail of the generated prompt:

#### Detailed
- Full verbose prompt with all elements
- Maximum control and precision
- Longer prompt
- **Best for:** Critical work, maximum consistency

#### Balanced (Default)
- Medium-length prompt
- Good balance of control and efficiency
- **Best for:** Most use cases

#### Concise
- Shorter, streamlined prompt
- Faster processing
- Less control
- **Best for:** Quick iterations, testing

---

## Workflow Examples

### Example 1: Heroic Low-Angle Portrait

**Goal:** Create powerful, heroic portrait with person looking imposing

**Settings:**
```
Perspective Preset: low_angle_worms_eye
Identity Preservation: strict
Psychological Effect: power
Focal Point: auto (legs and thighs)
Body Proportion: exaggerate_height
Detail Showcase: high
Prompt Style: balanced
```

**System Prompt:** "Portrait Photographer" or "Cinematographer"

**Result:** Subject appears powerful and monumental, shot from ground level looking up

---

### Example 2: Fashion Editorial with Vulnerability

**Goal:** High-fashion shot with emotional vulnerability

**Settings:**
```
Perspective Preset: high_angle_birds_eye
Identity Preservation: strict
Psychological Effect: vulnerability
Focal Point: face_upper
Body Proportion: natural
Background Adaptation: true
Lighting Reinforcement: true
Detail Showcase: high
Prompt Style: detailed
```

**System Prompt:** "Fashion Photographer"

**Scene Context:** "in minimalist white studio with soft natural lighting"

**Result:** Editorial fashion shot from above, emphasizing outfit and emotional tone

---

### Example 3: Character Turnaround (Side View)

**Goal:** Consistent character view for concept art

**Settings:**
```
Perspective Preset: side_profile
Identity Preservation: strict
Psychological Effect: none
Focal Point: full_body
Body Proportion: natural
Composition Centering: true
Detail Showcase: high
Prompt Style: detailed
```

**System Prompt:** "Character Artist"

**Result:** Clean side profile view, perfect for character sheets

---

### Example 4: Dynamic Dramatic Portrait

**Goal:** Artistic, tension-filled portrait

**Settings:**
```
Perspective Preset: dutch_angle
Identity Preservation: moderate
Psychological Effect: mystery
Focal Point: face_upper
Body Proportion: natural
Detail Showcase: medium
Prompt Style: balanced
```

**System Prompt:** "Portrait Photographer"

**Scene Context:** "dramatic lighting with strong shadows"

**Result:** Tilted, dramatic portrait with artistic flair

---

### Example 5: Classic Three-Quarter Portrait

**Goal:** Timeless, professional portrait

**Settings:**
```
Perspective Preset: three_quarter_view
Identity Preservation: strict
Psychological Effect: confidence
Focal Point: auto
Body Proportion: natural
Background Adaptation: true
Lighting Reinforcement: true
Composition Centering: true
Detail Showcase: high
Prompt Style: balanced
```

**System Prompt:** "Portrait Photographer"

**Scene Context:** "professional studio with soft lighting"

**Result:** Classic, professional portrait with depth

---

## Best Practices

### 1. Always Use Strict Identity Preservation

For consistent results with people, **always use "strict" identity preservation** unless you specifically want variation.

### 2. Keep Subject Centered

The model works best when the person is **centered in the frame**. Off-center subjects can cause identity drift or inconsistent results.

### 3. Use Appropriate System Prompts

Choose system prompts that emphasize identity preservation:
- **Portrait Photographer** (best for most cases)
- **Fashion Photographer** (for fashion work)
- **Character Artist** (for concept art)
- **Cinematographer** (for cinematic shots)

### 4. Match Psychological Effect to Angle

Use the preset's intended psychological effect:
- High angles → Vulnerability, intimacy
- Low angles → Power, grandeur
- Eye level → Confidence, balance
- Dutch angle → Tension, drama

### 5. Let Presets Do Their Job

Start with preset defaults (especially focal point and body proportion). These are optimized based on community testing.

### 6. Use Scene Context Effectively

Add scene context to establish:
- Lighting conditions
- Environment type
- Mood and atmosphere
- Background elements

**Example:** "in modern office with large windows and natural light"

### 7. Background Adaptation + Lighting Reinforcement

Keep these **enabled** for natural, cohesive results. Disable only when you need exact preservation of lighting/background.

### 8. Prompt Style Choice

- **Detailed:** For critical professional work
- **Balanced:** For most everyday use (recommended)
- **Concise:** For quick testing and iterations

### 9. Test Multiple Perspectives

Try different presets to find the best angle for your character:
1. Start with eye_level_front (baseline)
2. Try three_quarter_view (most versatile)
3. Experiment with high/low angles for specific effects

### 10. Debug Mode for Learning

Enable debug mode when learning the node. It shows:
- Which preset is active
- Psychological effects
- Full generated prompt
- Preservation hints
- Helpful tips

---

## Troubleshooting

### Problem: Subject's Face Changes

**Cause:** Identity preservation too weak

**Solutions:**
- Set identity_preservation to "strict"
- Use "Portrait Photographer" system prompt
- Ensure subject is centered in frame
- Try "detailed" prompt style for more control

---

### Problem: Clothing/Outfit Changes

**Cause:** Identity preservation not including clothing

**Solutions:**
- Use "strict" identity preservation (includes clothes)
- Add explicit clothing description to scene_context
- Use "Fashion Photographer" system prompt

---

### Problem: Perspective Not Strong Enough

**Cause:** Model not applying dramatic enough angle

**Solutions:**
- Use extreme presets (high_angle_birds_eye or low_angle_worms_eye)
- Add matching psychological effect
- Use "detailed" prompt style
- Increase conditioning strength in encoder (0.9-1.2)

---

### Problem: Background Changes Too Much

**Cause:** Background adaptation too aggressive

**Solutions:**
- Disable background_adaptation
- Add detailed background description to scene_context
- Use camera-specific system prompt ("Cinematographer")
- Add "keep background identical" to custom_additions

---

### Problem: Lighting Changes Dramatically

**Cause:** Lighting reinforcement creating new light

**Solutions:**
- Disable lighting_reinforcement
- Specify exact lighting in scene_context
- Add "maintain original lighting" to custom_additions

---

### Problem: Body Proportions Look Wrong

**Cause:** Body proportion setting not matching angle

**Solutions:**
- Use "natural" body proportion first
- For low angles: Try "exaggerate_height"
- For high angles: Try "diminish_height"
- Check that perspective preset matches your goal

---

### Problem: Results Not Consistent Between Generations

**Cause:** Not enough identity constraints

**Solutions:**
- Use "strict" identity preservation
- Enable composition_centering
- Use "detailed" prompt style
- Add very specific scene_context
- Use same seed in sampler

---

### Problem: Person Rotates Instead of Camera Changing

**Cause:** Model interpreting as object rotation

**Solutions:**
- Make sure you're using Person Perspective node, NOT Object Rotation
- Use person-focused system prompt
- Add "keep pose identical" to custom_additions
- Emphasize identity preservation

---

### Problem: Dutch Angle Too Extreme/Not Enough

**Cause:** Model's interpretation of tilt degree

**Solutions:**
- Add specific tilt degree to custom_additions (e.g., "tilt 30 degrees")
- Try different psychological effects
- Adjust conditioning strength
- Experiment with detail_showcase level

---

## System Prompt Recommendations

### For Different Use Cases

#### Portrait Photography (General)
**System Prompt:** "Portrait Photographer"

Best for: Standard portraits, headshots, character photography

**Pairs well with:**
- eye_level_front
- three_quarter_view
- high_angle_birds_eye

---

#### Fashion Photography
**System Prompt:** "Fashion Photographer"

Best for: Editorial fashion, outfit showcases, dramatic fashion angles

**Pairs well with:**
- high_angle_birds_eye
- low_angle_worms_eye
- dutch_angle
- three_quarter_view

---

#### Character/Concept Art
**System Prompt:** "Character Artist"

Best for: Character turnarounds, concept art, consistent character views

**Pairs well with:**
- side_profile
- three_quarter_view
- eye_level_front

---

#### Cinematic/Dramatic Shots
**System Prompt:** "Cinematographer"

Best for: Cinematic character moments, story-driven shots, dramatic angles

**Pairs well with:**
- low_angle_worms_eye
- high_angle_birds_eye
- dutch_angle

---

### Custom System Prompt Template

If you want to create your own system prompt for person perspective work, use this template:

```
You are a [ROLE]. When given camera angle instructions for photographing people,
change ONLY the camera perspective while preserving the subject's identity completely.
Keep their facial features, clothing, hairstyle, pose, and all other identifying
characteristics identical. Focus on [YOUR SPECIFIC GOAL]. Maintain [YOUR LIGHTING/STYLE
PREFERENCES] appropriate to the perspective.
```

**Example:**
```
You are a celebrity portrait photographer. When given camera angle instructions for
photographing people, change ONLY the camera perspective while preserving the subject's
identity completely. Keep their facial features, clothing, hairstyle, pose, and all
other identifying characteristics identical. Focus on creating flattering angles that
convey confidence and approachability. Maintain soft, professional lighting appropriate
to the perspective.
```

---

## Complete Workflow Setup

### Recommended Node Chain

```
1. Load Image
   ↓
2. ArchAi3D Qwen Image Scale
   ├→ image1_vl (for vision encoder)
   └→ image1_latent (for VAE)
   ↓
3. ArchAi3D Qwen System Prompt
   └→ Choose "Portrait Photographer" or "Fashion Photographer"
   ↓
4. ArchAi3D Qwen Person Perspective
   ├─ perspective_preset: Choose your angle
   ├─ identity_preservation: strict
   ├─ psychological_effect: Choose based on mood
   ├─ scene_context: Add environment description
   └→ perspective_prompt output
   ↓
5. ArchAi3D Qwen Encoder V2 (or Simple)
   ├─ image1_vl: From scale node
   ├─ image1_latent: From scale node
   ├─ system_prompt: From system prompt node
   ├─ prompt: From person perspective node
   ├─ conditioning_strength: 0.9-1.1
   └→ conditioning output
   ↓
6. KSampler
   ├─ Connect conditioning
   ├─ steps: 28-35
   ├─ cfg: 3.5-5.0
   └→ Generate!
   ↓
7. VAE Decode
   └→ Final image
```

### Recommended Settings by Use Case

#### Professional Portrait Photography
```
Encoder V2:
  - conditioning_strength: 1.0
  - context_strength: 0.85
  - user_text_strength: 1.0

KSampler:
  - steps: 30
  - cfg: 4.0
  - sampler: dpmpp_2m
  - scheduler: karras
```

---

#### Fashion Editorial
```
Encoder V2:
  - conditioning_strength: 1.1
  - context_strength: 0.9
  - user_text_strength: 1.05

KSampler:
  - steps: 32
  - cfg: 4.5
  - sampler: dpmpp_2m
  - scheduler: karras
```

---

#### Character Concept Art
```
Encoder V2:
  - conditioning_strength: 0.95
  - context_strength: 0.95
  - user_text_strength: 0.95

KSampler:
  - steps: 35
  - cfg: 3.5
  - sampler: euler_a
  - scheduler: normal
```

---

## Tips from Community Research

Based on the Reddit tutorial by Vortexneonlight:

### 1. Person vs Environment
**Finding:** Perspective changes work BEST with people CENTERED in frame

**Why:** Off-center or environmental shots may cause the model to rotate the person instead of changing camera angle

**Tip:** Frame your subject in the center of the composition before applying perspective changes

---

### 2. Identity is Critical
**Finding:** Without explicit identity preservation, the model will change facial features, clothes, and pose

**Why:** Image editing models want to "improve" or "vary" the subject

**Tip:** Always use "strict" identity preservation with the exact wording: "Keep the subject's id, clothes, facial features, pose, and hairstyle identical"

---

### 3. Psychological Effects Are Real
**Finding:** Camera angles genuinely convey psychological states:
- High angles = vulnerability, small, weak
- Low angles = power, dominance, heroic

**Why:** Deep-rooted visual language from cinema and photography

**Tip:** Use psychological effects intentionally to tell your character's story

---

### 4. System Prompts Help Consistency
**Finding:** Using "Portrait Photographer" or "Cinematographer" system prompts improves identity preservation

**Why:** Sets the AI's role to focus on camera work, not subject redesign

**Tip:** Always pair Person Perspective node with appropriate system prompt

---

### 5. Lighting Should Adapt
**Finding:** Lighting that matches the new perspective creates more realistic results

**Why:** Light naturally comes from different directions at different angles

**Tip:** Keep lighting_reinforcement enabled unless you have specific lighting requirements

---

## Advanced Techniques

### Technique 1: Multi-Angle Character Sheet

Create a complete character turnaround with 4 key angles:

**Setup:**
1. Start with centered character portrait
2. Generate 4 variations using these presets:
   - eye_level_front
   - three_quarter_view
   - side_profile
   - three_quarter_view (opposite direction - add "from the left" to custom_additions)

**Settings for all:**
- identity_preservation: strict
- composition_centering: true
- prompt_style: detailed
- System prompt: "Character Artist"

**Result:** Consistent character views from multiple angles

---

### Technique 2: Narrative Perspective Sequence

Tell a story through perspective changes:

**Scene 1 - Vulnerability:** High angle, subject looks small and isolated
**Scene 2 - Growing Confidence:** Eye level, subject centered and balanced
**Scene 3 - Triumph:** Low angle, subject looks powerful and heroic

**Key:** Use same subject, same clothing, same pose - only perspective changes tell the story

---

### Technique 3: Fashion Lookbook Series

Create professional fashion series:

1. **Front:** Eye level front view
2. **Detail:** Three-quarter view (face + outfit)
3. **Silhouette:** Side profile (body line)
4. **Drama:** High angle or dutch angle (editorial flair)

**Settings:**
- identity_preservation: strict
- System prompt: "Fashion Photographer"
- Scene context: Consistent background/lighting
- detail_showcase: high

---

### Technique 4: Portrait + Psychology Matrix

Create the same portrait with different psychological effects:

**Matrix:**
```
                Low Angle          Eye Level          High Angle
Confidence      Powerful Pro        Professional      Approachable
Vulnerability   Conflicted         Uncertain          Isolated
Mystery         Imposing Shadow     Enigmatic         Watchful
```

**Use:** Explore character personality through angle + psychological effect combinations

---

## Conclusion

The **ArchAi3D Qwen Person Perspective** node gives you professional control over camera perspectives for people photography while maintaining identity consistency.

### Key Takeaways

1. **Identity Preservation is #1** - Always use "strict" for consistency
2. **Center Your Subject** - Works best with centered compositions
3. **Match System Prompts** - Use person-focused system prompts
4. **Understand Psychology** - High = vulnerable, Low = powerful
5. **Let Presets Guide You** - They're optimized from community research
6. **Scene Context Matters** - Add environment details for coherence
7. **Test and Iterate** - Try multiple presets to find the perfect angle

### Resources

- **Reddit Tutorial:** Original research by Vortexneonlight on r/StableDiffusion
- **System Prompts:** Use "Portrait Photographer", "Fashion Photographer", or "Character Artist"
- **Node Pack:** Part of ArchAi3D Qwen toolkit for ComfyUI

### Support

For more workflows, presets, and support:
- **Patreon:** [patreon.com/archai3d](https://patreon.com/archai3d)
- **GitHub:** [github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen](https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen)

---

**Happy photographing! Transform your character portraits with professional perspective control.**

*Based on community research and real-world testing with Qwen Edit 2509*
