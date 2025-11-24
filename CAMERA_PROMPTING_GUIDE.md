# 📸 The 5-Ingredient Camera Prompting Guide

## Master Professional Camera Control for Qwen-VL Image Editing

**Author:** Amir Ferdos (ArchAi3d)
**Based On:** Nanobanan's proven 5-ingredient formula + Research-validated best practices
**For:** ComfyUI Cinematography Prompt Builder & all Qwen camera workflows

---

## Introduction: Why Camera Prompts Matter

Have you ever asked an AI to "change the camera angle" and gotten... something generic and disappointing? Or requested a "close-up" only to receive an image that's somehow both too close and too far at the same time?

**The problem is simple:** Vague camera instructions produce inconsistent results.

When you tell Qwen-VL (or any vision-language model) to "change the view," it has to guess what you mean. Should it move closer or farther? Higher or lower? Should the background be blurred or sharp? What mood are you going for? Without clear guidance, the AI makes random choices—and you get random results.

**The solution is equally simple:** Use the 5-Ingredient Formula.

This guide teaches you a proven framework created by Nanobanan that transforms vague requests into precise, professional camera prompts. Whether you're a complete beginner or a seasoned cinematographer, this formula gives you consistent, predictable control over every aspect of camera positioning and framing.

### What You'll Learn

- **The 5 ingredients** that every great camera prompt needs
- **Professional cinematography techniques** explained in beginner-friendly language
- **15+ real working examples** you can use immediately
- **Quick reference charts** for shot types, angles, depth of field, and styles
- **How to integrate** this knowledge with the Cinematography Prompt Builder node

### Who This Guide Is For

- **Beginners:** Start with the simple explanations and basic examples
- **Professionals:** Dive into the technical details, f-stops, and advanced techniques
- **Everyone:** Use the formula to get consistent, high-quality results every single time

Let's master camera control together!

---

## The 5-Ingredient Formula 🎯🖼️📐🔎🎨

Every professional camera prompt needs exactly **5 ingredients**:

1. **Subject 🎯** - What the camera is focusing on
2. **Shot Type (Framing) 🖼️** - How much of the subject appears in frame
3. **Angle (Vantage Point) 📐** - Where the camera is positioned relative to the subject
4. **Focus (Depth of Field) 🔎** - What's sharp and what's blurred
5. **Style (Mood) 🎨** - The overall aesthetic and lighting

Think of it like cooking: You need all 5 ingredients to create the perfect dish. Leave one out, and the AI has to guess—and guessing leads to inconsistency.

---

## Ingredient 1: Subject 🎯

### What Is It?

The subject is **what your camera is pointing at**. It's the focal point, the star of your shot, the element that matters most.

### Why It Matters

Generic subjects ("a room", "an object") get generic results. Specific subjects ("the green stove", "an elderly woman in a flowing blue dress") get precise, consistent results.

### Beginner Level: Keep It Simple

Just describe what you want to photograph:
- "the chair"
- "the watch"
- "the stove"

**Tip:** Always include "the" or "a/an" before the noun. "The chair" works better than just "chair."

### Professional Level: Add Detail

Go beyond simple naming—include attributes, states, actions, and context:

**Basic:** "a person"
**Professional:** "an elderly woman in a flowing blue dress, standing confidently with arms crossed, backlit by golden sunset"

**Basic:** "the stove"
**Professional:** "the green stove with cast-iron grates and marble backsplash"

**Why this works:** Detailed descriptions help Qwen-VL preserve identity across viewpoint changes. When you specify "the green stove," the AI knows to keep it green even when changing angles.

### Common Mistakes to Avoid

❌ **Too vague:** "room", "object", "thing"
✅ **Just right:** "the modern office", "the vintage watch", "the sculptural vase"

❌ **Changing subject mid-prompt:** "the chair and then the table"
✅ **Pick one:** "the chair" (focus on one subject per prompt)

### Quick Reference: Subject Specificity Levels

| Level | Example | Best For |
|-------|---------|----------|
| **Minimal** | "the chair" | Quick tests, simple workflows |
| **Standard** | "the green leather armchair" | Most use cases |
| **Detailed** | "the emerald green leather armchair with brass studs and curved mahogany legs" | Identity preservation, complex edits |

---

## Ingredient 2: Shot Type (Framing) 🖼️

### What Is It?

Shot type determines **how much of your subject** appears in the frame. It's the difference between seeing someone's entire body (full shot) versus just their eyes (extreme close-up).

### Why It Matters

Shot type controls three critical things:
1. **What information** the viewer sees
2. **Emotional connection** (closer = more intimate)
3. **Camera distance** (automatically determines how far away the camera should be)

### The 8 Professional Shot Types

From closest to farthest:

#### 1. Extreme Close-Up (ECU)
- **Distance:** ~30 centimeters (12 inches)
- **Shows:** Tiny details, textures, micro-elements
- **Emotional Effect:** Intense scrutiny, dramatic detail revelation
- **Example:** "Individual water droplets on a leaf, iris patterns in an eye"

#### 2. Close-Up (CU)
- **Distance:** ~80 centimeters (2.6 feet)
- **Shows:** Head and shoulders, major features
- **Emotional Effect:** Intimate, personal, emotional connection
- **Example:** "A person's face filling the frame, a product's key features"

#### 3. Medium Close-Up (MCU)
- **Distance:** ~1.2 meters (4 feet)
- **Shows:** Chest up, upper body
- **Emotional Effect:** Personal but comfortable, conversational
- **Example:** "News anchor framing, product with immediate context"

#### 4. Medium Shot (MS)
- **Distance:** ~2.5 meters (8 feet)
- **Shows:** Waist up, torso and arms
- **Emotional Effect:** Balanced, natural, social distance
- **Example:** "Two people talking, subject with workspace visible"

#### 5. Medium Long Shot (MLS)
- **Distance:** ~3.5 meters (11.5 feet)
- **Shows:** Knees up, most of body
- **Emotional Effect:** Shows environment context while keeping subject prominent
- **Example:** "Person in their office, chef at their station"

#### 6. Full Shot (FS)
- **Distance:** ~4.5 meters (15 feet)
- **Shows:** Head to toes, complete subject
- **Emotional Effect:** Establishes full subject and immediate surroundings
- **Example:** "Fashion photography, full furniture piece in context"

#### 7. Wide Shot (WS)
- **Distance:** ~6.5 meters (21 feet)
- **Shows:** Subject within environment
- **Emotional Effect:** Establishes setting, shows spatial relationships
- **Example:** "Person in a room, car in a driveway, establishing shots"

#### 8. Extreme Wide Shot (EWS)
- **Distance:** ~10+ meters (33+ feet)
- **Shows:** Vast environment, subject is small element
- **Emotional Effect:** Epic scale, isolation, grandeur
- **Example:** "Landscape with tiny figure, building in cityscape"

### Choosing the Right Shot Type

**For product photography:** ECU (details), CU (features), MS (context)
**For interior design:** FS (furniture), WS (room layout), MS (specific elements)
**For architectural work:** WS (building), EWS (site context), MLS (entrance)

### Common Mistakes to Avoid

❌ **Using abbreviations:** "ecu" instead of "extreme close-up"
✅ **Spell it out:** "extreme close-up" (natural language works best)

❌ **Mixing shot types:** "close-up of the room"
✅ **Match scale:** "wide shot of the room" or "close-up of the door handle"

### Quick Reference: Shot Type Chart

| Shot Type | Distance | What You See | Use When |
|-----------|----------|--------------|----------|
| **ECU** | 30cm | Micro details | Revealing texture, patterns, tiny elements |
| **CU** | 80cm | Main features | Showing emotion, key product details |
| **MCU** | 1.2m | Upper portion | Conversational, personal interaction |
| **MS** | 2.5m | Half of subject | Balanced framing, action with context |
| **MLS** | 3.5m | Most of subject | Subject + near environment |
| **FS** | 4.5m | Complete subject | Full view with immediate surroundings |
| **WS** | 6.5m | Subject in space | Establishing location, spatial relations |
| **EWS** | 10m+ | Vast environment | Epic scale, grand perspective |

---

## Ingredient 3: Angle (Vantage Point) 📐

### What Is It?

Camera angle is **where the camera is positioned** relative to your subject—not just left/right, but critically: **up and down**.

### Why It Matters

Camera angles carry psychological weight. They're not just technical choices—they affect how viewers feel about your subject.

### The 7 Essential Camera Angles

#### 1. Eye Level (Straight On)
- **Position:** Camera at subject's eye height, looking straight ahead
- **Psychological Effect:** Neutral, objective, realistic perspective
- **Use When:** You want natural, unbiased representation
- **Example:** "An eye-level medium shot of the chair"

#### 2. High Angle (Looking Down)
- **Position:** Camera above subject, pointing downward
- **Psychological Effect:** Makes subject appear smaller, vulnerable, diminished
- **Use When:** Showing submission, weakness, or providing overview
- **Example:** "A high angle looking down at the desk from above"

#### 3. Low Angle (Looking Up)
- **Position:** Camera below subject, pointing upward
- **Psychological Effect:** Makes subject appear powerful, imposing, monumental
- **Use When:** Emphasizing strength, dominance, grandeur
- **Example:** "A low angle looking up at the bookshelf from ground level"

#### 4. Bird's Eye View (Directly Overhead)
- **Position:** Camera at 90° directly above, looking straight down
- **Psychological Effect:** God's-eye view, complete spatial overview
- **Use When:** Floor plans, layouts, pattern-focused compositions
- **Example:** "A bird's eye view of the dining table from directly above"

#### 5. Worm's Eye View (Ground Level Up)
- **Position:** Camera at ground level, looking straight up
- **Psychological Effect:** Extreme drama, distortion, monumental scale
- **Use When:** Creating dramatic impact, showing height/scale
- **Example:** "A worm's eye view from floor level looking up at the ceiling"

#### 6. Dutch Angle (Tilted Horizon)
- **Position:** Camera tilted on its roll axis, creating diagonal horizon
- **Psychological Effect:** Unease, disorientation, tension, dynamic energy
- **Use When:** Creating visual interest, suggesting instability
- **Example:** "A Dutch angle tilted 15 degrees showing the hallway"

#### 7. Over-the-Shoulder (OTS)
- **Position:** Camera behind one subject looking toward another
- **Psychological Effect:** Point of view, conversation context, spatial relationship
- **Use When:** Showing interaction, perspective, conversational setups
- **Example:** "An over-the-shoulder view from behind the desk toward the window"

### Angle + Height = Powerful Combination

Don't just think horizontally—combine angles with specific heights:

**"Eye level at 1.5 meters"** → Standard human perspective
**"High angle from 3 meters above"** → Elevated overview
**"Low angle from 0.5 meters off ground"** → Dramatic upward view

### Common Mistakes to Avoid

❌ **Vague directions:** "from the side", "at an angle"
✅ **Specific angles:** "eye level", "high angle looking down", "low angle looking up"

❌ **Conflicting instructions:** "bird's eye low angle"
✅ **Choose one:** "bird's eye view" OR "low angle"

### Quick Reference: Camera Angles Chart

| Angle | Position | Psychological Effect | Best For |
|-------|----------|---------------------|----------|
| **Eye Level** | Straight ahead | Neutral, realistic | Natural representation |
| **High Angle** | Looking down | Vulnerability, submission | Overviews, showing layouts |
| **Low Angle** | Looking up | Power, dominance | Emphasizing strength, height |
| **Bird's Eye** | Overhead 90° | Complete overview | Layouts, patterns, floor plans |
| **Worm's Eye** | Ground looking up | Extreme drama | Architectural drama, scale |
| **Dutch Angle** | Tilted horizon | Unease, dynamism | Visual interest, tension |
| **Over-the-Shoulder** | Behind toward front | POV, relationship | Conversations, interactions |

---

## Ingredient 4: Focus (Depth of Field) 🔎

### What Is It?

Depth of Field (DOF) controls **what's sharp and what's blurred** in your image. It's the difference between everything being in focus (deep DOF) versus only your subject being sharp with a blurred background (shallow DOF).

### Why It Matters

DOF is one of the most powerful storytelling tools in photography:
- **Shallow DOF:** Isolates your subject, creates cinematic bokeh, directs attention
- **Deep DOF:** Shows complete environment, maintains context, keeps everything readable

### The 5 DOF Levels

#### 1. Extreme Shallow (f/1.2 - f/1.8)
- **Effect:** Only a tiny sliver is sharp, everything else is smooth blur
- **Bokeh:** Beautiful circular light orbs, creamy smooth backgrounds
- **Use When:** Macro photography, isolating tiny details, artistic product shots
- **Technical:** Wide apertures (f/1.2, f/1.4, f/1.8), close distances
- **Example:** "with extreme shallow depth of field where only the watch dial is sharp"

#### 2. Very Shallow (f/2.0 - f/2.8)
- **Effect:** Subject isolated, foreground and background blur smoothly
- **Bokeh:** Prominent background blur, subject separation
- **Use When:** Portraits, product photography, cinematic storytelling
- **Technical:** Wide apertures (f/2.0, f/2.8), portrait lenses
- **Example:** "with very shallow depth of field creating blurred background"

#### 3. Shallow (f/4.0 - f/5.6)
- **Effect:** Subject sharp, background noticeably soft but still recognizable
- **Bokeh:** Moderate background softness, balanced separation
- **Use When:** Environmental portraits, products in context, standard photography
- **Technical:** Moderate apertures (f/4, f/5.6)
- **Example:** "with shallow depth of field, subject in focus, background softly blurred"

#### 4. Medium (f/8.0 - f/11)
- **Effect:** Subject and near surroundings sharp, distant areas soft
- **Bokeh:** Minimal, mostly sharp throughout
- **Use When:** Street photography, general use, balanced compositions
- **Technical:** Mid-range apertures (f/8, f/11)
- **Example:** "with medium depth of field balancing subject and environment"

#### 5. Deep (f/16 - f/22)
- **Effect:** Everything from foreground to background is sharp
- **Bokeh:** None, complete sharpness throughout
- **Use When:** Landscapes, architecture, real estate, environmental documentation
- **Technical:** Narrow apertures (f/16, f/22), wider lenses
- **Example:** "with deep depth of field keeping everything in sharp focus"

### Matching DOF to Shot Type

**Automatic combinations that work well:**

- **ECU + Extreme/Very Shallow:** Perfect for macro details
- **CU + Shallow/Very Shallow:** Classic portrait/product isolation
- **MS + Shallow/Medium:** Balanced subject with soft environment
- **FS + Medium/Deep:** Subject clear, environment contextual
- **WS/EWS + Deep:** Everything visible and sharp

### Common Mistakes to Avoid

❌ **Conflicting combinations:** "wide shot with extreme shallow depth of field"
✅ **Match scale:** "wide shot with deep depth of field" (shows environment)

❌ **Technical jargon:** "f/1.4 bokeh aesthetic"
✅ **Natural language:** "very shallow depth of field creating smooth background blur"

### Quick Reference: Depth of Field Chart

| DOF Level | F-Stop | What's Sharp | Bokeh Quality | Best For |
|-----------|--------|--------------|---------------|----------|
| **Extreme Shallow** | f/1.2-1.8 | Tiny sliver | Creamy, circular orbs | Macro, artistic isolation |
| **Very Shallow** | f/2.0-2.8 | Subject only | Prominent blur | Portraits, products |
| **Shallow** | f/4.0-5.6 | Subject + near | Moderate softness | Environmental portraits |
| **Medium** | f/8.0-11 | Subject + mid-range | Minimal blur | General photography |
| **Deep** | f/16-22 | Everything | None, all sharp | Landscapes, architecture |

### Bokeh Quality Descriptors

Use these natural language terms to describe blur characteristics:

- **"smooth creamy bokeh"** → Soft, pleasing blur without harsh edges
- **"circular light orbs"** → Out-of-focus highlights become circles
- **"soft background blur"** → General pleasant defocus
- **"dissolves into bokeh"** → Background melts away smoothly
- **"ring-shaped bokeh"** → Donut-like highlights (specific lens aesthetic)

---

## Ingredient 5: Style (Mood) 🎨

### What Is It?

Style sets the **overall aesthetic, mood, and lighting characteristics** of your shot. It's the difference between a gritty documentary feel and a polished cinematic look.

### Why It Matters

Style guides the AI's choices about:
- **Lighting:** Bright and even vs. dramatic shadows
- **Color grading:** Natural tones vs. stylized palettes
- **Processing:** Clean and sharp vs. film grain and texture
- **Mood:** Energetic vs. contemplative, modern vs. vintage

### 10 Essential Styles

#### 1. Cinematic
- **Characteristics:** Film grain, teal-orange color grading, dramatic contrast
- **Lighting:** Motivated, directional, often low-key
- **Mood:** Dramatic, story-driven, emotional
- **Use When:** Creating film-like aesthetic, dramatic storytelling
- **Example:** "in cinematic style with dramatic lighting and contrast"

#### 2. Clean/Modern
- **Characteristics:** Bright, even lighting, minimal shadows, crisp details
- **Lighting:** Soft, diffused, high-key
- **Mood:** Fresh, professional, contemporary
- **Use When:** Product photography, modern interiors, commercial work
- **Example:** "in clean and modern style with bright even lighting"

#### 3. Architectural
- **Characteristics:** Straight lines, geometric precision, minimal distortion
- **Lighting:** Even, shows materials accurately, no extreme shadows
- **Mood:** Professional, precise, technical
- **Use When:** Real estate, design portfolios, technical documentation
- **Example:** "in architectural style emphasizing clean lines and accurate materials"

#### 4. Natural/Realistic
- **Characteristics:** Mimics human vision, balanced exposure, true colors
- **Lighting:** Appears unmanipulated, natural window light
- **Mood:** Honest, authentic, unposed
- **Use When:** Documentation, lifestyle photography, authentic representation
- **Example:** "in natural realistic style with unprocessed appearance"

#### 5. Documentary/Editorial
- **Characteristics:** Handheld aesthetic, candid moments, journalistic
- **Lighting:** Available light, no obvious manipulation
- **Mood:** Authentic, reportage, storytelling
- **Use When:** Photojournalism, editorial work, authentic moments
- **Example:** "in documentary style with authentic handheld aesthetic"

#### 6. Fine Art
- **Characteristics:** Artistic interpretation, unconventional framing, creative processing
- **Lighting:** Expressive, mood-driven, potentially dramatic
- **Mood:** Contemplative, artistic, interpretive
- **Use When:** Gallery work, artistic expression, creative projects
- **Example:** "in fine art photography style with artistic interpretation"

#### 7. Commercial/Product
- **Characteristics:** Flawless presentation, perfect lighting, high detail
- **Lighting:** Studio-quality, multiple sources, no flaws visible
- **Mood:** Premium, polished, aspirational
- **Use When:** Advertising, product catalogs, marketing materials
- **Example:** "in commercial product style with perfect studio lighting"

#### 8. Vintage/Retro
- **Characteristics:** Faded colors, grain, vignetting, nostalgic processing
- **Lighting:** Softer, lower contrast than modern
- **Mood:** Nostalgic, timeless, sentimental
- **Use When:** Creating period feel, nostalgic mood, artistic projects
- **Example:** "in vintage style with faded colors and film grain"

#### 9. High-Key (Bright & Airy)
- **Characteristics:** Bright exposure, minimal shadows, light backgrounds
- **Lighting:** Soft, diffused, fill light on all sides
- **Mood:** Optimistic, clean, fresh, inviting
- **Use When:** Fashion, beauty, lifestyle, uplifting mood
- **Example:** "in high-key style with bright airy lighting"

#### 10. Low-Key (Dark & Moody)
- **Characteristics:** Dark backgrounds, dramatic shadows, limited highlights
- **Lighting:** Hard, directional, creating strong contrast
- **Mood:** Dramatic, mysterious, intimate, sophisticated
- **Use When:** Luxury products, dramatic portraits, moody atmospheres
- **Example:** "in low-key style with dramatic shadows and dark background"

### Lighting Style Sub-Categories

Add specific lighting descriptions for even more control:

- **"Golden hour sunset"** → Warm, directional, magical quality
- **"Soft window light"** → Natural, gentle, flattering
- **"Hard dramatic"** → Strong shadows, high contrast
- **"Overcast daylight"** → Even, soft, no harsh shadows
- **"Studio lighting"** → Professional, controlled, multi-source

### Common Mistakes to Avoid

❌ **Too many styles:** "cinematic vintage documentary style"
✅ **Pick one primary:** "cinematic style" OR "vintage style"

❌ **Conflicting moods:** "bright dramatic dark lighting"
✅ **Choose coherent:** "bright and even lighting" OR "dramatic with shadows"

### Quick Reference: Style Chart

| Style | Lighting | Mood | Color | Best For |
|-------|----------|------|-------|----------|
| **Cinematic** | Dramatic, directional | Story-driven | Teal-orange grading | Film-like aesthetic |
| **Clean/Modern** | Bright, even | Fresh, contemporary | Neutral, accurate | Products, modern interiors |
| **Architectural** | Even, precise | Professional | True to materials | Real estate, design |
| **Natural** | Unmanipulated | Authentic | True colors | Documentation, lifestyle |
| **Documentary** | Available light | Journalistic | Realistic | Editorial, reportage |
| **Fine Art** | Expressive | Artistic | Interpretive | Gallery, creative |
| **Commercial** | Studio-perfect | Premium | Polished | Advertising, marketing |
| **Vintage** | Soft, low contrast | Nostalgic | Faded, warm | Period feel, artistic |
| **High-Key** | Bright, minimal shadows | Uplifting | Light, airy | Fashion, lifestyle |
| **Low-Key** | Dark, dramatic | Mysterious | Deep, moody | Luxury, dramatic |

---

## Putting It All Together: Working Examples

Now let's see how all 5 ingredients combine to create professional camera prompts. Each example is annotated to show you exactly which ingredient does what.

### Featured Example 1: Full Shot (Environmental Context)

```
An eye-level full shot of the green stove, taken from a vantage point 4 meters away.
The entire stove is centered in the frame, clearly showing it, the marble backsplash,
and the range hood above. The lighting is bright and even, keeping the entire cooking
area in sharp focus.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the green stove" (specific, detailed)
- **🖼️ Shot Type:** "full shot" (4 meters = complete subject with immediate surroundings)
- **📐 Angle:** "eye level" (neutral, realistic perspective)
- **🔎 Focus (DOF):** "entire cooking area in sharp focus" (deep DOF, everything clear)
- **🎨 Style:** "bright and even" lighting (clean/modern aesthetic)

**Why This Works:**
- Clear subject with specific attribute ("green")
- Appropriate shot type for showing furniture in context
- Eye level provides natural, realistic view
- Deep DOF ensures all details (stove, backsplash, hood) are visible
- Bright even lighting suits kitchen/product photography

**Use This Format For:** Interior design, product photography, real estate, furniture showcases

---

### Featured Example 2: Extreme Macro (Micro Detail Focus)

```
An extreme macro photo (1:1 magnification) of the green stove, focusing on the
intricate details of a single burner and the cast-iron grate. The vantage point
is inches away, creating an extremely shallow depth of field where only the front
edge of the burner is in sharp focus, and the rest of the stove and kitchen dissolves
into a soft, blurred bokeh.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the green stove, focusing on... single burner and cast-iron grate" (macro specificity)
- **🖼️ Shot Type:** "extreme macro" (inches away = ECU distance with 1:1 magnification)
- **📐 Angle:** "vantage point is inches away" (extreme close proximity, eye level to subject)
- **🔎 Focus (DOF):** "extremely shallow... only front edge sharp... dissolves into bokeh" (extreme shallow DOF)
- **🎨 Style:** "soft, blurred bokeh" (cinematic/artistic aesthetic)

**Why This Works:**
- Subject narrowed from "stove" to "single burner" for macro scale
- Extreme macro shot type matches the tiny detail focus
- Close vantage point creates intimate perspective
- Extreme shallow DOF perfect for isolating micro details
- Bokeh quality adds artistic, cinematic feel

**Use This Format For:** Product detail photography, texture reveals, artistic close-ups, material showcases

---

### Example 3: Close-Up Portrait Style

```
An eye-level close-up of the vintage watch, taken from a vantage point eighty
centimeters away. The watch dial and hands fill the frame, with very shallow depth
of field creating a smooth creamy bokeh background. The lighting is soft and diffused,
in clean commercial style emphasizing the polished metal surfaces.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the vintage watch" (specific, with attribute)
- **🖼️ Shot Type:** "close-up" (80cm = CU distance, perfect for product features)
- **📐 Angle:** "eye level" (neutral, professional view)
- **🔎 Focus (DOF):** "very shallow... smooth creamy bokeh" (product isolation)
- **🎨 Style:** "soft and diffused... clean commercial style" (professional product aesthetic)

**Why This Works:**
- Vintage watch as subject establishes what to preserve
- Close-up shows key features (dial, hands) while maintaining product context
- Eye level provides authentic, un-dramatized view
- Very shallow DOF isolates product from distractions
- Commercial style with soft lighting suits luxury product photography

**Use This Format For:** Product photography, jewelry, watches, small items, commercial shoots

---

### Example 4: Wide Shot Architectural

```
An eye-level wide shot of the modern living room, taken from a vantage point six and
a half meters away from the far wall. The entire room is visible including the sofa,
coffee table, and floor-to-ceiling windows. Deep depth of field keeps everything from
foreground furniture to background wall art in sharp focus, in clean architectural
style with natural window light.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the modern living room" (room as complete subject)
- **🖼️ Shot Type:** "wide shot" (6.5m = WS distance, shows space and spatial relationships)
- **📐 Angle:** "eye level" (human perspective, realistic)
- **🔎 Focus (DOF):** "deep... everything... in sharp focus" (complete environmental clarity)
- **🎨 Style:** "clean architectural... natural window light" (professional design documentation)

**Why This Works:**
- Room as subject appropriate for wide environmental shot
- Wide shot distance shows complete space layout
- Eye level matches how visitors would see the space
- Deep DOF essential for showing all design elements clearly
- Architectural style with natural light suits interior design photography

**Use This Format For:** Interior design, real estate, architectural photography, room showcases

---

### Example 5: Low Angle Dramatic

```
A low angle looking up at the bookshelf from ground level, taken from a vantage point
thirty centimeters off the floor. The camera tilts upward showing the towering shelves
extending toward the ceiling. Medium depth of field keeps the lower shelves sharp while
upper sections gradually soften, in dramatic cinematic style with side lighting creating
long shadows.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the bookshelf" (clear focal point)
- **🖼️ Shot Type:** "extreme close-up distance" (30cm from floor = dramatic proximity)
- **📐 Angle:** "low angle looking up... from ground level" (power/drama angle)
- **🔎 Focus (DOF):** "medium... lower shelves sharp... upper sections soften" (selective focus)
- **🎨 Style:** "dramatic cinematic... side lighting... shadows" (moody, artistic)

**Why This Works:**
- Bookshelf as subject maintains clear focus
- ECU distance from ground creates extreme perspective
- Low angle makes bookshelf appear monumental, imposing
- Medium DOF creates depth while maintaining readability
- Cinematic style with dramatic lighting enhances the powerful angle

**Use This Format For:** Dramatic product shots, furniture with verticality, artistic interior photography

---

### Example 6: Bird's Eye Layout

```
A bird's eye view of the dining table from directly overhead, taken from a vantage
point three meters above. The camera looks straight down showing the complete table
setting with plates, glasses, and centerpiece arranged in perfect symmetry. Deep
depth of field keeps all tableware in sharp focus, in clean modern style with bright
even lighting eliminating shadows.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the dining table" (clear overhead subject)
- **🖼️ Shot Type:** "medium long shot distance" (3m overhead = MLS equivalent)
- **📐 Angle:** "bird's eye view... directly overhead... straight down" (complete overhead perspective)
- **🔎 Focus (DOF):** "deep... all tableware in sharp focus" (flat field sharpness)
- **🎨 Style:** "clean modern... bright even... eliminating shadows" (catalog-style photography)

**Why This Works:**
- Table as subject perfect for overhead symmetry
- MLS distance provides complete view while maintaining detail
- Bird's eye angle ideal for showing arrangements and layouts
- Deep DOF ensures nothing in the flat plane is blurred
- Clean modern style suits product/design documentation

**Use This Format For:** Food photography, table settings, flat lays, pattern photography, design layouts

---

### Example 7: Medium Shot Conversational

```
An eye-level medium shot of the office chair, taken from a vantage point two and a
half meters away. The chair occupies the center of the frame from seat to headrest,
with the desk and laptop visible in soft focus behind. Shallow depth of field keeps
the chair sharp while the background workspace provides context without distraction,
in natural realistic style with soft window light from the left.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the office chair" (clear furniture subject)
- **🖼️ Shot Type:** "medium shot" (2.5m = MS distance, balanced framing)
- **📐 Angle:** "eye level" (natural seated perspective)
- **🔎 Focus (DOF):** "shallow... chair sharp... background soft" (subject isolation with context)
- **🎨 Style:** "natural realistic... soft window light" (authentic, lifestyle aesthetic)

**Why This Works:**
- Chair as subject with contextual elements mentioned
- Medium shot perfect for furniture with environmental context
- Eye level matches typical viewing height for seated furniture
- Shallow DOF isolates product while maintaining environmental context
- Natural style appropriate for lifestyle/catalog photography

**Use This Format For:** Furniture photography, product-in-context shots, lifestyle imagery, catalog work

---

### Example 8: High Angle Overview

```
A high angle looking down at the kitchen island from above, taken from a vantage
point two meters above the counter. The camera points downward showing the marble
surface, cutting board, and ingredients arranged for cooking. Medium depth of field
keeps the immediate counter sharp while the far edge and floor gradually soften, in
clean editorial style with bright natural daylight.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the kitchen island" (countertop workspace)
- **🖼️ Shot Type:** "medium close-up distance" (2m above = MCU overhead equivalent)
- **📐 Angle:** "high angle looking down... from above... points downward" (elevated overview)
- **🔎 Focus (DOF):** "medium... immediate counter sharp... far edge softens" (dimensional focus)
- **🎨 Style:** "clean editorial... bright natural daylight" (lifestyle/magazine aesthetic)

**Why This Works:**
- Kitchen island as subject with activity context
- MCU distance provides detail while showing workspace
- High angle provides cooking/prep perspective
- Medium DOF creates depth while maintaining usability
- Editorial style suits food/lifestyle photography

**Use This Format For:** Food prep photography, workspace documentation, lifestyle editorial, cooking scenes

---

### Example 9: Medium Close-Up Detail

```
An eye-level medium close-up of the sculptural vase, taken from a vantage point one
point two meters away. The vase fills two-thirds of the frame showing its curved form
and glazed surface, with the shelf and wall art behind visible but secondary. Shallow
depth of field keeps the vase's front surface sharp while the rear curve and background
gently blur, in fine art style with soft directional lighting emphasizing the ceramic
texture.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the sculptural vase" (art object with attributes)
- **🖼️ Shot Type:** "medium close-up" (1.2m = MCU distance, upper portion emphasis)
- **📐 Angle:** "eye level" (art viewing perspective)
- **🔎 Focus (DOF):** "shallow... front sharp... rear and background blur" (dimensional focus)
- **🎨 Style:** "fine art... soft directional... emphasizing texture" (artistic presentation)

**Why This Works:**
- Sculptural vase as art object with surface detail noted
- MCU distance shows form and detail without losing context
- Eye level matches gallery/display viewing height
- Shallow DOF creates dimensional depth on curved object
- Fine art style appropriate for aesthetic object photography

**Use This Format For:** Art object photography, ceramics, sculpture, decorative items, gallery documentation

---

### Example 10: Extreme Wide Establishing

```
An eye-level extreme wide shot of the office building entrance, taken from a vantage
point fifteen meters away across the plaza. The building occupies the upper two-thirds
of the frame while the foreground shows the landscaped walkway and people for scale.
Deep depth of field keeps everything from foreground pavement to background architecture
in sharp focus, in clean architectural style with even overcast daylight eliminating
harsh shadows.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the office building entrance" (architectural subject)
- **🖼️ Shot Type:** "extreme wide shot" (15m = EWS distance, establishing perspective)
- **📐 Angle:** "eye level" (human street-level perspective)
- **🔎 Focus (DOF):** "deep... everything... foreground to background sharp" (environmental clarity)
- **🎨 Style:** "clean architectural... even overcast... eliminating shadows" (documentation aesthetic)

**Why This Works:**
- Building entrance as subject with environmental context
- EWS distance provides site context and scale reference (people)
- Eye level matches pedestrian viewing perspective
- Deep DOF essential for architectural detail and environmental clarity
- Architectural style with even lighting suits professional documentation

**Use This Format For:** Architectural photography, real estate exteriors, site documentation, establishing shots

---

### Example 11: Dutch Angle Dynamic

```
A Dutch angle tilted twenty-five degrees showing the modern hallway, taken from eye
level at a vantage point four and a half meters from the end wall. The camera tilts
creating diagonal lines from the floor, ceiling, and doorframes. Medium depth of field
keeps the immediate hallway sharp while the far end gently softens, in cinematic style
with dramatic side lighting creating angular shadows that emphasize the tilted composition.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the modern hallway" (architectural interior)
- **🖼️ Shot Type:** "full shot distance" (4.5m = FS, complete view)
- **📐 Angle:** "Dutch angle tilted twenty-five degrees" (dynamic diagonal perspective)
- **🔎 Focus (DOF):** "medium... immediate sharp... far end softens" (depth emphasis)
- **🎨 Style:** "cinematic... dramatic side lighting... angular shadows" (artistic mood)

**Why This Works:**
- Hallway as subject benefits from perspective treatment
- FS distance shows complete corridor while maintaining impact
- Dutch angle creates visual interest and dynamic energy
- Medium DOF enhances depth perception in linear space
- Cinematic style with dramatic lighting amplifies the unconventional angle

**Use This Format For:** Artistic interior photography, dynamic compositions, music videos, creative editorial

---

### Example 12: Over-the-Shoulder Context

```
An over-the-shoulder view from behind the desk chair toward the window, taken from
eye level at a vantage point one meter behind the seat. The left edge of the frame
shows the chair back and arm out of focus, while the window and city view beyond are
the sharp focal point. Shallow depth of field keeps the foreground chair blurred and
the background window view sharp, in documentary style with natural window light.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the window (and city view beyond)" (background as primary subject)
- **🖼️ Shot Type:** "medium close-up distance" (1m = MCU, conversational proximity)
- **📐 Angle:** "over-the-shoulder... from behind... toward" (POV perspective)
- **🔎 Focus (DOF):** "shallow... foreground blurred... background sharp" (reverse focus)
- **🎨 Style:** "documentary... natural window light" (authentic, journalistic)

**Why This Works:**
- Window view as subject with chair providing context/POV
- MCU distance creates intimate workspace perspective
- OTS angle provides relational context and viewer immersion
- Shallow DOF with reverse focus (blur foreground, sharp background) is unconventional but effective
- Documentary style suits authentic workspace photography

**Use This Format For:** POV photography, workspace documentation, lifestyle editorial, environmental context

---

### Example 13: Macro Material Detail

```
An extreme close-up of the marble countertop, taken from a vantage point five centimeters
above the surface. The camera captures a hand-sized area showing the intricate veining
patterns, crystal structure, and polished finish in extreme detail. Extremely shallow
depth of field keeps only the top surface in razor-sharp focus while the depth of the
stone quickly falls to bokeh, in commercial product style with even studio lighting
revealing the material's luxurious texture.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the marble countertop (veining, crystal structure, finish)" (material detail)
- **🖼️ Shot Type:** "extreme close-up" (5cm = ECU, macro proximity)
- **📐 Angle:** "vantage point... above the surface" (overhead macro perspective)
- **🔎 Focus (DOF):** "extremely shallow... only top surface sharp" (micro depth isolation)
- **🎨 Style:** "commercial product... even studio lighting... luxurious texture" (high-end presentation)

**Why This Works:**
- Material itself (marble) as subject with specific attributes detailed
- ECU distance perfect for revealing micro textures and patterns
- Overhead angle provides flat, uniform view of surface
- Extremely shallow DOF creates dimensional interest even on flat surface
- Commercial style appropriate for material specification/luxury presentation

**Use This Format For:** Material samples, texture photography, specification documentation, luxury product details

---

### Example 14: Worm's Eye Monumental

```
A worm's eye view from floor level looking straight up at the chandelier, taken from
a vantage point directly beneath the fixture. The camera points upward showing the
crystal structure ascending toward the ceiling, creating dramatic perspective with the
widest elements nearest and the chain receding into the distance. Deep depth of field
keeps all crystal tiers in sharp focus from bottom to top, in dramatic cinematic style
with backlighting from ceiling fixtures creating sparkle through the crystals.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the chandelier (crystal structure, tiers, chain)" (vertical fixture)
- **🖼️ Shot Type:** "extreme close-up distance" (floor level = ECU proximity from below)
- **📐 Angle:** "worm's eye view... floor level... straight up... directly beneath" (extreme upward)
- **🔎 Focus (DOF):** "deep... all tiers sharp from bottom to top" (vertical sharpness)
- **🎨 Style:** "dramatic cinematic... backlighting... sparkle through crystals" (luxury aesthetic)

**Why This Works:**
- Chandelier as vertical subject perfect for extreme upward angle
- ECU floor distance creates monumental scale and drama
- Worm's eye angle maximizes height perception and grandeur
- Deep DOF essential for keeping all tiers sharp across vertical distance
- Cinematic style with backlighting enhances luxury and drama

**Use This Format For:** Chandelier photography, tall architectural elements, monument photography, dramatic fixtures

---

### Example 15: Full Shot Lifestyle

```
An eye-level full shot of the yoga practitioner, taken from a vantage point four and
a half meters away in the home studio. The complete figure is visible from head to
toe in a standing pose, with the hardwood floor, plants, and window light visible
around them. Medium depth of field keeps the person in sharp focus while the room
details remain clear but slightly softened, in natural realistic style with soft golden
hour window light streaming from the left.
```

**Ingredient Breakdown:**

- **🎯 Subject:** "the yoga practitioner (in standing pose)" (person with activity context)
- **🖼️ Shot Type:** "full shot" (4.5m = FS, complete figure with environment)
- **📐 Angle:** "eye level" (natural observational perspective)
- **🔎 Focus (DOF):** "medium... person sharp... room slightly softened" (balanced focus)
- **🎨 Style:** "natural realistic... soft golden hour window light" (lifestyle aesthetic)

**Why This Works:**
- Practitioner as subject with activity and pose specified
- FS distance shows complete figure and practice environment
- Eye level provides natural, respectful perspective
- Medium DOF balances subject emphasis with environmental context
- Natural style with golden hour light perfect for lifestyle/wellness photography

**Use This Format For:** Lifestyle photography, fitness/wellness, environmental portraits, activity documentation

---

## Quick Reference Charts

### Shot Type Distance Chart

| Shot Type | Distance | Natural Language | Best For |
|-----------|----------|------------------|----------|
| **ECU** | 30 cm | "thirty centimeters away" | Micro details, textures, extreme close-ups |
| **CU** | 80 cm | "eighty centimeters away" | Faces, key features, intimate details |
| **MCU** | 1.2 m | "one point two meters away" | Upper body, conversational framing |
| **MS** | 2.5 m | "two and a half meters away" | Waist up, balanced composition |
| **MLS** | 3.5 m | "three and a half meters away" | Knees up, environmental context |
| **FS** | 4.5 m | "four and a half meters away" | Head to toe, complete subject |
| **WS** | 6.5 m | "six and a half meters away" | Subject in environment |
| **EWS** | 10+ m | "ten meters away" (or more) | Epic scale, vast environments |

### Camera Angle Quick Reference

| Angle | Position | Effect | Keywords |
|-------|----------|--------|----------|
| **Eye Level** | Straight ahead | Neutral, realistic | "eye level", "straight on" |
| **High Angle** | Above, looking down | Subject appears smaller | "high angle looking down", "from above" |
| **Low Angle** | Below, looking up | Subject appears powerful | "low angle looking up", "from ground level" |
| **Bird's Eye** | Directly overhead 90° | Complete overview | "bird's eye view", "directly overhead", "from above looking straight down" |
| **Worm's Eye** | Ground level, straight up | Extreme drama | "worm's eye view", "floor level looking up" |
| **Dutch Angle** | Tilted on roll axis | Unease, dynamism | "Dutch angle tilted [X] degrees" |
| **Over-the-Shoulder** | Behind one toward another | POV, relationship | "over-the-shoulder view from behind [A] toward [B]" |

### Depth of Field Quick Reference

| DOF Level | F-Stop | Effect | Natural Language |
|-----------|--------|--------|------------------|
| **Extreme Shallow** | f/1.2-1.8 | Tiny sliver sharp | "extremely shallow depth of field", "only [element] in sharp focus" |
| **Very Shallow** | f/2.0-2.8 | Subject isolated | "very shallow depth of field creating blurred background" |
| **Shallow** | f/4.0-5.6 | Subject sharp, background soft | "shallow depth of field", "subject in focus, background softly blurred" |
| **Medium** | f/8.0-11 | Balanced sharpness | "medium depth of field", "balanced focus throughout" |
| **Deep** | f/16-22 | Everything sharp | "deep depth of field keeping everything in sharp focus" |

### Style Keywords by Category

#### Lighting Styles
- **Bright & Even:** "bright and even lighting", "soft diffused light"
- **Dramatic:** "dramatic lighting with shadows", "hard directional light"
- **Natural:** "natural window light", "soft daylight"
- **Golden Hour:** "warm golden hour sunset", "golden light from the side"
- **Studio:** "professional studio lighting", "even studio illumination"

#### Aesthetic Styles
- **Cinematic:** "cinematic style", "film-like aesthetic"
- **Clean/Modern:** "clean and modern style", "contemporary aesthetic"
- **Architectural:** "architectural style", "precise geometric style"
- **Documentary:** "documentary style", "editorial aesthetic"
- **Fine Art:** "fine art photography style", "artistic interpretation"

#### Mood Descriptors
- **Dramatic:** "dramatic and moody", "high contrast shadows"
- **Bright:** "bright and airy", "high-key lighting"
- **Dark:** "dark and moody", "low-key with shadows"
- **Natural:** "natural realistic", "authentic unprocessed"
- **Polished:** "commercial polished", "premium presentation"

---

## Using with Cinematography Prompt Builder Node

### Parameter Mapping Guide

The Cinematography Prompt Builder node implements the 5-ingredient formula through its parameters:

#### Required Parameters (Layer 1 - Nanobanan's 5 Ingredients)

1. **Subject 🎯** → `target_subject` field
   - Enter: "the green stove", "the vintage watch", "the office chair"

2. **Shot Type 🖼️** → `shot_type` dropdown
   - Select: Extreme Close-Up (ECU) through Extreme Wide Shot (EWS)
   - Distance auto-calculates!

3. **Angle 📐** → `camera_angle` dropdown
   - Select: Eye Level, High Angle, Low Angle, Bird's Eye, etc.

4. **Focus (DOF) 🔎** → `depth_of_field` dropdown
   - Select: Extreme Shallow, Very Shallow, Shallow, Medium, Deep
   - Or choose "Auto (based on shot size)" for smart defaults

5. **Style 🎨** → `style_mood` + `lighting_style` dropdowns
   - Style/Mood: Cinematic, Clean/Modern, Natural/Neutral, etc.
   - Lighting: Auto/Natural, Bright & Even, Soft & Diffused, etc.

#### Optional Parameters (Layers 2-4 - Professional Enhancement)

- **Lens Type Override:** Auto-selected from shot type, or manual override
- **Camera Movement:** Static, Pan, Tilt, Dolly, Truck, Arc, Zoom
- **Material Detail Preset:** 37 material presets (v6 integration)
- **Photography Quality Preset:** 15 quality enhancement presets
- **Custom Details:** FREE TEXT - Add extra specifics here!

### Custom Details: The Secret Sauce

Use the `custom_details` field to add the specific touches from your working examples:

**Example Custom Details:**
- "The entire stove is centered in the frame, clearly showing it, the marble backsplash, and the range hood above"
- "focusing on the intricate details of a single burner and the cast-iron grate"
- "showing dial and hands clearly"
- "with visible veining patterns in the marble"
- "The watch dial fills the frame"
- "dissolves into a soft, blurred bokeh"

**Tip:** Custom details are where you add compositional specifics, what should be visible, and descriptive flourishes beyond the core 5 ingredients.

### Node Outputs Explained

The node generates 4 outputs:

1. **simple_prompt** - Nanobanan-style natural language (English)
   - Uses the format: "An [angle] [shot type] of [subject], taken from [distance]..."
   - Perfect for beginners and general use

2. **professional_prompt** - v7-style with Chinese cinematography terms
   - Starts with "Next Scene:" for dx8152 LoRA compatibility
   - Optimized for professional results with Chinese LoRAs

3. **system_prompt** - Dynamic LLM role instruction
   - Simple/Beginner mode: Focuses on 5-ingredient framework
   - Professional mode: Chinese terms + dx8152 optimization
   - Research mode: M-RoPE, guidance scale 6-8, spatial reasoning

4. **description** - Human-readable summary with emojis
   - Shows all settings, auto-calculated values, warnings
   - Useful for debugging and understanding what the node did

### Workflow Integration

**Basic Workflow:**
```
[Cinematography Prompt Builder]
    ↓ simple_prompt
[Qwen Encoder]
    ↓ conditioning
[KSampler]
    ↓
[VAE Decode]
    ↓
[Save Image]
```

**Advanced Workflow:**
```
[Cinematography Prompt Builder]
    ↓ professional_prompt + system_prompt
[Qwen Encoder V2]
    ↓ conditioning
[GRAG Modifier] (optional)
    ↓
[KSampler]
    ↓
[VAE Decode]
    ↓
[Save Image]
```

---

## Advanced Tips & Techniques

### 1. Combining Ingredients for Specific Effects

**Isolation Effect (Product Photography):**
- Shot Type: Close-Up or Medium Close-Up
- Angle: Eye Level (neutral)
- DOF: Very Shallow or Shallow
- Style: Commercial/Clean
- Result: Product stands out, background disappears

**Environmental Context (Interior Design):**
- Shot Type: Wide Shot or Full Shot
- Angle: Eye Level
- DOF: Deep
- Style: Architectural/Clean
- Result: Complete room view, all details visible

**Dramatic Impact (Artistic):**
- Shot Type: Extreme Close-Up or Medium Shot
- Angle: Low Angle or Worm's Eye
- DOF: Extreme Shallow
- Style: Cinematic/Dramatic
- Result: Powerful, moody, artistic imagery

### 2. When to Break the Rules

**Rules are guidelines, not laws.** Sometimes breaking them creates unique results:

- **Wide Shot + Shallow DOF:** Unusual, but can create dreamy, atmospheric environmental shots
- **Close-Up + Deep DOF:** Shows extreme detail with unusual sharpness across micro distances
- **High Angle + Power Subject:** Subverts expectation, creates irony or vulnerability in powerful subjects

**Key:** If you break a rule, do it intentionally with a specific effect in mind.

### 3. Troubleshooting Common Issues

**Problem:** "The AI keeps making the background too blurry/sharp"
- **Solution:** Explicitly specify DOF: "with deep depth of field keeping everything in sharp focus" OR "with very shallow depth of field creating blurred background"

**Problem:** "The camera angle seems wrong or random"
- **Solution:** Be more specific with angle: Don't say "at an angle"—say "high angle looking down from 2 meters above" or "low angle looking up from ground level"

**Problem:** "The style/mood doesn't match what I imagined"
- **Solution:** Add specific lighting descriptors: "with soft window light" or "with dramatic side lighting creating shadows"

**Problem:** "The shot type doesn't show the right amount of subject"
- **Solution:** Check shot type distance matches your intent. MS (2.5m) is waist-up, FS (4.5m) is head-to-toe. Adjust shot type, not just distance words.

### 4. Research-Validated Best Practices

Based on the vision-language camera control research PDF:

✅ **Use distance-based positioning** ("2.5 meters away") instead of degree-based ("45 degrees to the right")
- Distance = more consistent results
- Degrees = less reliable, more variation

✅ **Specify shot type explicitly** ("full shot") instead of letting it change during angle transitions
- Prevents unintended reframing
- Maintains compositional consistency

✅ **Combine specific language with guidance scale 6-8** (if using manual CFG)
- Lower (6-7) = more creative freedom
- Higher (7-8) = stricter adherence to prompt

✅ **Use natural language, not technical abbreviations**
- "extreme close-up" > "ECU"
- "depth of field" > "DOF"
- Natural language = better AI understanding

✅ **For dx8152 LoRAs, use "Next Scene:" prefix + Chinese terms**
- Triggers LoRA training patterns
- Improves camera control consistency
- Node's professional_prompt output handles this automatically

---

## One-Page Quick Reference Card

### The 5-Ingredient Formula Cheat Sheet

**Every camera prompt needs:**

1. **🎯 Subject** - What you're photographing
   - Specific: "the green stove", "the vintage watch"
   - Detailed: Attributes, states, actions, context

2. **🖼️ Shot Type** - How much fits in frame
   - ECU (30cm), CU (80cm), MCU (1.2m), MS (2.5m)
   - MLS (3.5m), FS (4.5m), WS (6.5m), EWS (10m+)

3. **📐 Angle** - Camera position
   - Eye Level, High (down), Low (up)
   - Bird's Eye (overhead), Worm's Eye (ground up)
   - Dutch (tilted), Over-the-Shoulder

4. **🔎 Focus (DOF)** - Sharp vs. blurred
   - Extreme Shallow, Very Shallow, Shallow
   - Medium, Deep

5. **🎨 Style** - Aesthetic and lighting
   - Cinematic, Clean/Modern, Architectural
   - Natural, Documentary, Fine Art, Commercial

### Formula Template

```
An [ANGLE] [SHOT TYPE] of [SUBJECT], taken from a vantage point [DISTANCE] away.
[Compositional details]. [DOF description]. [STYLE with LIGHTING].
```

### Example Using Template

```
An eye-level full shot of the green stove, taken from a vantage point four and a
half meters away. The entire stove is centered in the frame, clearly showing it,
the marble backsplash, and the range hood above. Deep depth of field keeps everything
in sharp focus. Bright and even lighting in clean modern style.
```

### Most Common Combinations

**Product Detail:** CU + Eye Level + Very Shallow DOF + Commercial Style
**Interior Design:** FS/WS + Eye Level + Deep DOF + Architectural Style
**Artistic/Dramatic:** ECU/MS + Low Angle + Shallow DOF + Cinematic Style
**Documentation:** WS + Eye Level + Deep DOF + Natural Style
**Lifestyle:** MS/FS + Eye Level + Medium DOF + Natural Style

---

## Conclusion: Your Path to Mastery

You now have everything you need to create professional, consistent camera prompts using Nanobanan's proven 5-ingredient formula.

**Your Next Steps:**

1. **Start Simple:** Pick one of the 15 working examples and try it with your own subject
2. **Experiment:** Change one ingredient at a time to see its effect
3. **Build Intuition:** Notice which combinations work best for your projects
4. **Go Professional:** Add technical details, custom specifics, and advanced techniques
5. **Share Your Results:** Help others learn by documenting what works for you

**Remember:**
- All 5 ingredients matter—don't skip any
- Be specific, not vague
- Use natural language, not abbreviations
- Match shot type to your subject scale
- Combine ingredients intentionally for desired effects

**The Cinematography Prompt Builder node** automates much of this formula, but understanding the principles makes you a better prompter regardless of the tools you use.

Happy creating! 📸

---

## Credits & Resources

**5-Ingredient Formula:** Nanobanan (original creator)
**Research Foundation:** "Camera View Control in Vision-Language Image Editing Models" (arXiv)
**Working Examples:** Amir Ferdos (ArchAi3d)
**Implementation:** Cinematography Prompt Builder for ComfyUI

**Cinematography Resources:**
- StudioBinder (shot types, camera angles)
- MasterClass (cinematography techniques)
- B&H Photo (lens characteristics, f-stops)

**For Support:**
- GitHub Issues: https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen/issues
- Patreon: https://patreon.com/archai3d
- Email: Amir84ferdos@gmail.com
- LinkedIn: https://www.linkedin.com/in/archai3d/

---

## License

**Dual License Model:**
- **Personal/Non-Commercial Use:** Free
- **Commercial Use:** License required (contact Amir84ferdos@gmail.com)

For full license details, see license_file.txt

---

**Version:** 1.0
**Last Updated:** 2025-01-06
**Compatible With:** ComfyUI Cinematography Prompt Builder v2.4.0+, all Qwen camera workflows
