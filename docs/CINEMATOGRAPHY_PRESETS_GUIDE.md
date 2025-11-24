# Cinematography Presets Guide

**Complete reference for all 19 cinematography presets in ArchAi3D Qwen Object Rotation V2**

---

## Table of Contents

1. [Overview](#overview)
2. [Product Photography (5 presets)](#product-photography)
3. [Architecture & Real Estate (4 presets)](#architecture--real-estate)
4. [Cinematic & Professional (4 presets)](#cinematic--professional)
5. [Quick & Single Frame (3 presets)](#quick--single-frame)
6. [Special Effects (3 presets)](#special-effects)
7. [Preset Comparison Table](#preset-comparison-table)
8. [Use Case Recommendations](#use-case-recommendations)
9. [Customization Tips](#customization-tips)

---

## Overview

The Object Rotation V2 node includes **19 professional cinematography presets** that instantly configure all rotation parameters for specific use cases. Simply select a preset from the dropdown and all settings (angle, steps, orbit distance, speed, elevation, etc.) are automatically configured.

### How Presets Work

When you select a preset (anything except "custom"), the node automatically sets:
- **Angle** - How many degrees to rotate (45° to 360°)
- **Steps** - Number of frames to generate (1 to 24)
- **Multi-step mode** - Single frame vs video sequence
- **Direction** - Usually "right" (clockwise from top)
- **Orbit distance** - Close/medium/wide
- **Speed hint** - Smooth/slow/fast/cinematic
- **Elevation** - Level/rising/descending
- **Maintain distance** - Keep consistent distance from subject
- **Keep level** - Keep camera level during orbit

**Note:** When a preset is active, manual controls below are overridden. Select "custom" to use manual controls.

---

## Product Photography

Perfect for e-commerce, product showcases, and detail shots.

### 1. Product Turntable
**ID:** `product_turntable`

**Description:** 360° smooth rotation for e-commerce (8 frames)

**Configuration:**
- Angle: 360°
- Steps: 8 frames (45° each)
- Multi-step: Yes
- Orbit distance: Close
- Speed: Smooth
- Elevation: Level

**Best For:**
- E-commerce product listings
- Amazon/eBay product videos
- Furniture showcases
- Consumer electronics
- Fashion accessories

**Example Workflow:**
```
Product in center → Product Turntable preset → 8 frames showing all sides
Perfect for: Shoes, watches, electronics, toys
```

**Pro Tip:** Use with "product" subject type for optimized prompts

---

### 2. Product Four Views
**ID:** `product_four_views`

**Description:** 4 cardinal views - Front/Right/Back/Left (4 frames, 90° each)

**Configuration:**
- Angle: 360°
- Steps: 4 frames (90° each)
- Multi-step: Yes
- Orbit distance: Close
- Speed: Smooth
- Elevation: Level

**Best For:**
- Product comparison sheets
- Technical documentation
- Quick product overview
- Catalog photography
- Product specifications

**Example Use:**
```
Frame 1: Front view (0°)
Frame 2: Right side (90°)
Frame 3: Back view (180°)
Frame 4: Left side (270°)
```

**Pro Tip:** Perfect for creating product comparison grids or technical spec sheets

---

### 3. Inspection View
**ID:** `inspection_view`

**Description:** Multiple angles for detailed examination (12 frames)

**Configuration:**
- Angle: 360°
- Steps: 12 frames (30° each)
- Multi-step: Yes
- Orbit distance: Close
- Speed: Smooth
- Elevation: Level

**Best For:**
- Quality inspection documentation
- Detailed product examination
- High-value items (jewelry, watches)
- Technical parts showcase
- Vintage/collectibles

**Example Use:**
```
12 frames showing every 30° for thorough examination
Perfect for: Antiques, collectibles, jewelry, technical parts
```

**Pro Tip:** Use slower frame rate (1-2 FPS) for inspection video viewers

---

### 4. Detail Close Inspection
**ID:** `detail_close_inspection`

**Description:** Extreme close-up 180° inspection (6 frames)

**Configuration:**
- Angle: 180°
- Steps: 6 frames (30° each)
- Multi-step: Yes
- Orbit distance: Close (extreme)
- Speed: Slow
- Elevation: Level

**Best For:**
- Macro product photography
- Surface detail examination
- Texture showcase
- Material quality demonstration
- Craftsmanship details

**Example Use:**
```
Close-up front → slow orbit to back → showcasing fine details
Perfect for: Watch mechanisms, fabric textures, wood grain, engravings
```

**Pro Tip:** Combine with good lighting to show material quality

---

### 5. Three Angle Showcase
**ID:** `three_angle_showcase`

**Description:** Front/Side/Back views (3 frames, 120° each)

**Configuration:**
- Angle: 360°
- Steps: 3 frames (120° each)
- Multi-step: Yes
- Orbit distance: Medium
- Speed: Smooth
- Elevation: Level

**Best For:**
- Quick product overview
- Social media posts (swipe gallery)
- Product teasers
- Minimalist showcases
- Fast product introductions

**Example Use:**
```
Frame 1: Front view (0°)
Frame 2: Side angle (120°)
Frame 3: Back-ish view (240°)
```

**Pro Tip:** Great for Instagram carousel posts (3 images)

---

## Architecture & Real Estate

Perfect for buildings, interiors, and property showcases.

### 6. Architectural Walkaround
**ID:** `architectural_walkaround`

**Description:** 360° exterior building rotation (8 frames)

**Configuration:**
- Angle: 360°
- Steps: 8 frames (45° each)
- Multi-step: Yes
- Orbit distance: Wide
- Speed: Cinematic
- Elevation: Level

**Best For:**
- Building exteriors
- Architectural visualization
- Real estate marketing
- Property showcases
- Urban planning presentations

**Example Use:**
```
Wide shot circling around building showing all facades
Perfect for: Houses, commercial buildings, monuments
```

**Pro Tip:** Use "building" subject type for best results

---

### 7. Interior Walkthrough
**ID:** `interior_walkthrough`

**Description:** 180° room exploration (4 frames)

**Configuration:**
- Angle: 180°
- Steps: 4 frames (45° each)
- Multi-step: Yes
- Orbit distance: Medium
- Speed: Smooth
- Elevation: Level
- Maintain distance: False (allows natural movement)

**Best For:**
- Interior design showcases
- Real estate room tours
- Virtual staging
- Hotel room showcases
- Airbnb listings

**Example Use:**
```
Sweep through room showing different angles and features
Perfect for: Living rooms, bedrooms, hotel suites
```

**Pro Tip:** Works best with "room" subject type

---

### 8. Room Corner 90
**ID:** `room_corner_90`

**Description:** Interior corner sweep 90° (2 frames)

**Configuration:**
- Angle: 90°
- Steps: 2 frames (45° each)
- Multi-step: Yes
- Orbit distance: Medium
- Speed: Smooth
- Elevation: Level
- Maintain distance: False

**Best For:**
- Corner room views
- Before/after interior shots
- Compact room showcases
- Quick interior previews
- Room layout demonstrations

**Example Use:**
```
Frame 1: View of wall 1
Frame 2: Corner sweep to wall 2
Shows how two walls meet and room layout
```

**Pro Tip:** Great for showing room corners in small spaces

---

### 9. Drone Ascending Orbit
**ID:** `drone_ascending_orbit`

**Description:** 270° orbit while rising (drone-style, 9 frames)

**Configuration:**
- Angle: 270°
- Steps: 9 frames (30° each)
- Multi-step: Yes
- Orbit distance: Wide
- Speed: Cinematic
- Elevation: Rising
- Keep level: False

**Best For:**
- Aerial property shots
- Drone-style reveals
- Dramatic architecture shots
- Luxury property marketing
- Establishing shots

**Example Use:**
```
Start at eye level → orbit while ascending → end at bird's eye view
Perfect for: Luxury homes, estates, commercial properties
```

**Pro Tip:** Creates Hollywood-style property reveals

---

## Cinematic & Professional

Professional film and video production presets.

### 10. Hero Shot
**ID:** `hero_shot`

**Description:** 180° cinematic wide shot (single frame)

**Configuration:**
- Angle: 180°
- Steps: 1 frame
- Multi-step: No
- Orbit distance: Wide
- Speed: Cinematic
- Elevation: Level

**Best For:**
- Dramatic product reveals
- Key showcase images
- Marketing hero images
- Portfolio pieces
- Dramatic before/after

**Example Use:**
```
Single dramatic 180° rotation from front to back
Perfect for: Hero product shots, dramatic reveals, portfolio work
```

**Pro Tip:** Use for the main image in marketing materials

---

### 11. Reveal Shot
**ID:** `reveal_shot`

**Description:** 180° dramatic reveal (4 frames)

**Configuration:**
- Angle: 180°
- Steps: 4 frames (45° each)
- Multi-step: Yes
- Orbit distance: Medium
- Speed: Slow
- Elevation: Level

**Best For:**
- Product launch videos
- Dramatic storytelling
- Teaser videos
- Suspenseful reveals
- Marketing campaigns

**Example Use:**
```
Slow 180° rotation building suspense and revealing product
Perfect for: New product launches, teasers, announcements
```

**Pro Tip:** Add dramatic music for maximum impact

---

### 12. Slow Cinema Orbit
**ID:** `slow_cinema_orbit`

**Description:** Ultra-smooth 360° film-quality rotation (24 frames)

**Configuration:**
- Angle: 360°
- Steps: 24 frames (15° each)
- Multi-step: Yes
- Orbit distance: Medium
- Speed: Cinematic
- Elevation: Level

**Best For:**
- Film production
- High-end commercials
- Luxury brand videos
- Museum showcases
- Art documentation

**Example Use:**
```
24 frames at 24fps = 1 second of ultra-smooth cinema-quality rotation
Perfect for: Luxury watches, cars, fine art, premium products
```

**Pro Tip:** This is the highest quality preset - use for premium content

---

### 13. Dramatic Reveal
**ID:** `dramatic_reveal`

**Description:** 90° slow quarter turn reveal (single cinematic frame)

**Configuration:**
- Angle: 90°
- Steps: 1 frame
- Multi-step: No
- Orbit distance: Medium
- Speed: Cinematic
- Elevation: Level

**Best For:**
- Subtle dramatic moments
- Before/during/after sequences
- Key product features reveal
- Portfolio highlight shots
- Marketing key frames

**Example Use:**
```
Single 90° rotation showing a key feature or dramatic angle change
Perfect for: Revealing hidden features, showing side profiles
```

**Pro Tip:** Great for showing product features that aren't visible from front

---

## Quick & Single Frame

Fast presets for single images and quick angle changes.

### 14. Quick Peek 45
**ID:** `quick_peek_45`

**Description:** Subtle 45° angle change (single frame)

**Configuration:**
- Angle: 45°
- Steps: 1 frame
- Multi-step: No
- Orbit distance: Medium
- Speed: Smooth
- Elevation: Level

**Best For:**
- Subtle angle variations
- A/B testing angles
- Quick perspective changes
- Testing compositions
- Alternative view generation

**Example Use:**
```
Generate a slightly different angle (45°) for comparison or variety
Perfect for: Testing angles, creating variations, subtle changes
```

**Pro Tip:** Use when you want "almost the same" angle but with slight variation

---

### 15. Side by Side Compare
**ID:** `side_by_side_compare`

**Description:** Front vs Back comparison (180° apart, single frame)

**Configuration:**
- Angle: 180°
- Steps: 1 frame
- Multi-step: No
- Orbit distance: Medium
- Speed: None
- Elevation: Level

**Best For:**
- Before/after comparisons
- Front vs back shots
- Design comparisons
- Quality control
- Feature highlights

**Example Use:**
```
Generate opposite view (180°) for side-by-side comparison
Perfect for: Comparing front/back designs, before/after shots
```

**Pro Tip:** Create both views then place side-by-side in design software

---

### 16. Quick Spin
**ID:** `quick_spin`

**Description:** Fast 360° for dynamic effect (16 frames)

**Configuration:**
- Angle: 360°
- Steps: 16 frames (22.5° each)
- Multi-step: Yes
- Orbit distance: Medium
- Speed: Fast
- Elevation: Level

**Best For:**
- Dynamic product videos
- Energetic showcases
- Sports products
- Youth-oriented marketing
- Action product demos

**Example Use:**
```
Fast 360° spin creating energetic, dynamic feel
Perfect for: Sports gear, toys, gadgets, youth products
```

**Pro Tip:** Use faster playback speed (4-6 FPS) for extra energy

---

## Special Effects

Creative and social media optimized presets.

### 17. Spiral Ascent
**ID:** `spiral_ascent`

**Description:** Orbit while rising upward (8 frames)

**Configuration:**
- Angle: 360°
- Steps: 8 frames (45° each)
- Multi-step: Yes
- Orbit distance: Medium
- Speed: Smooth
- Elevation: Rising
- Keep level: False

**Best For:**
- Artistic showcases
- Dramatic reveals
- Luxury product launches
- Creative marketing
- Monument/statue reveals

**Example Use:**
```
Camera orbits while rising from ground level to bird's eye view
Perfect for: Statues, tall products, architectural elements
```

**Pro Tip:** Creates elegant, sophisticated reveal effect

---

### 18. Spiral Descent
**ID:** `spiral_descent`

**Description:** Orbit while descending (8 frames)

**Configuration:**
- Angle: 360°
- Steps: 8 frames (45° each)
- Multi-step: Yes
- Orbit distance: Medium
- Speed: Smooth
- Elevation: Descending
- Keep level: False

**Best For:**
- Dramatic conclusions
- Product "landing" effects
- Creative transitions
- Artistic showcases
- Inverse reveal effects

**Example Use:**
```
Camera orbits while descending from bird's eye to ground level
Perfect for: Ending sequences, dramatic conclusions
```

**Pro Tip:** Reverse of Spiral Ascent - use for outro sequences

---

### 19. Social Media Spin
**ID:** `social_media_spin`

**Description:** 360° optimized for social media (15 frames, ~7.5sec @2fps)

**Configuration:**
- Angle: 360°
- Steps: 15 frames (24° each)
- Multi-step: Yes
- Orbit distance: Medium
- Speed: Fast
- Elevation: Level

**Best For:**
- Instagram posts
- TikTok videos
- Facebook posts
- Social media ads
- Mobile-first content

**Example Use:**
```
15 frames @ 2fps = 7.5 seconds (perfect for Instagram/TikTok)
Perfect for: Social media product posts, mobile viewing
```

**Pro Tip:** Optimized for mobile viewing - not too fast, not too slow

---

## Preset Comparison Table

| Preset Name | Angle | Frames | Distance | Speed | Elevation | Best For |
|-------------|-------|--------|----------|-------|-----------|----------|
| **Product Turntable** | 360° | 8 | Close | Smooth | Level | E-commerce |
| **Product Four Views** | 360° | 4 | Close | Smooth | Level | Tech docs |
| **Inspection View** | 360° | 12 | Close | Smooth | Level | Quality inspection |
| **Detail Close Inspection** | 180° | 6 | Close | Slow | Level | Macro details |
| **Three Angle Showcase** | 360° | 3 | Medium | Smooth | Level | Quick overview |
| **Architectural Walkaround** | 360° | 8 | Wide | Cinematic | Level | Buildings |
| **Interior Walkthrough** | 180° | 4 | Medium | Smooth | Level | Room tours |
| **Room Corner 90** | 90° | 2 | Medium | Smooth | Level | Corner views |
| **Drone Ascending Orbit** | 270° | 9 | Wide | Cinematic | Rising | Aerial shots |
| **Hero Shot** | 180° | 1 | Wide | Cinematic | Level | Marketing hero |
| **Reveal Shot** | 180° | 4 | Medium | Slow | Level | Product launch |
| **Slow Cinema Orbit** | 360° | 24 | Medium | Cinematic | Level | Film quality |
| **Dramatic Reveal** | 90° | 1 | Medium | Cinematic | Level | Feature reveal |
| **Quick Peek 45** | 45° | 1 | Medium | Smooth | Level | Angle variation |
| **Side by Side Compare** | 180° | 1 | Medium | None | Level | Comparison |
| **Quick Spin** | 360° | 16 | Medium | Fast | Level | Dynamic energy |
| **Spiral Ascent** | 360° | 8 | Medium | Smooth | Rising | Elegant reveal |
| **Spiral Descent** | 360° | 8 | Medium | Smooth | Descending | Dramatic end |
| **Social Media Spin** | 360° | 15 | Medium | Fast | Level | Instagram/TikTok |

---

## Use Case Recommendations

### E-commerce Product Listings
**Top 3 Presets:**
1. Product Turntable (8 frames, standard)
2. Product Four Views (4 frames, quick)
3. Three Angle Showcase (3 frames, minimal)

**Why:** Clean, professional, shows all angles efficiently

---

### Real Estate & Architecture
**Top 3 Presets:**
1. Architectural Walkaround (full building)
2. Interior Walkthrough (room tours)
3. Drone Ascending Orbit (dramatic exterior)

**Why:** Wide shots, natural movement, professional presentation

---

### Luxury & High-End Products
**Top 3 Presets:**
1. Slow Cinema Orbit (premium quality)
2. Hero Shot (dramatic single frame)
3. Spiral Ascent (elegant reveal)

**Why:** Cinematic, elegant, premium feel

---

### Social Media Marketing
**Top 3 Presets:**
1. Social Media Spin (optimized duration)
2. Quick Spin (energetic)
3. Three Angle Showcase (swipe gallery)

**Why:** Mobile-optimized, engaging, shareable

---

### Technical Documentation
**Top 3 Presets:**
1. Inspection View (detailed)
2. Product Four Views (cardinal angles)
3. Detail Close Inspection (macro details)

**Why:** Comprehensive, systematic, clear documentation

---

### Creative/Artistic Projects
**Top 3 Presets:**
1. Spiral Ascent (creative reveal)
2. Dramatic Reveal (artistic single frame)
3. Reveal Shot (storytelling)

**Why:** Creative movement, artistic flair, unique perspectives

---

## Customization Tips

### Modifying Presets

While presets auto-configure everything, you can still add:

1. **Scene Context** - Add environment description
   ```
   Example: "in modern studio with dramatic lighting"
   ```

2. **Custom Subject** - Override subject type
   ```
   Example: "the limited edition sneaker"
   ```

3. **Debug Mode** - See exactly what preset configured
   - Enable debug_mode to see all applied settings

### Creating Custom Workflows

**Technique 1: Preset Chaining**
Use multiple presets in sequence:
```
1. Hero Shot (wide establishing)
2. Product Turntable (full rotation)
3. Detail Close Inspection (macro details)
```

**Technique 2: Preset + Manual Tweaks**
- Start with preset
- Switch to "custom"
- Adjust specific parameters
- Save your own workflow

**Technique 3: Subject Type Pairing**
Match presets with subject types:
- Product Turntable + "product"
- Architectural Walkaround + "building"
- Interior Walkthrough + "room"

### Frame Rate Recommendations

| Preset Type | Recommended FPS | Result Duration |
|-------------|----------------|-----------------|
| Cinematic (24 frames) | 24 fps | 1 second smooth |
| Standard (8 frames) | 2-4 fps | 2-4 seconds |
| Quick (16 frames) | 6-8 fps | 2-3 seconds dynamic |
| Social Media (15 frames) | 2 fps | 7.5 seconds |
| Inspection (12 frames) | 1-2 fps | 6-12 seconds slow |

---

## Advanced Techniques

### Multi-Angle Showcase Sequence

Combine multiple single-frame presets:
```
1. Front view (starting point)
2. Quick Peek 45 (slight right)
3. Dramatic Reveal 90 (side view)
4. Side by Side Compare 180 (back view)
```
Result: 4 distinct professional angles

---

### Video Loop Creation

Use full rotation presets for seamless loops:
```
1. Product Turntable (360°/8 frames)
2. Export as video
3. Loop seamlessly (ends where it starts)
```
Perfect for: Website backgrounds, digital signage

---

### Hybrid Manual + Preset Workflow

1. Select preset to auto-configure
2. Note the settings in debug output
3. Switch to "custom"
4. Manually adjust one or two parameters
5. Keep other preset optimizations

Example: Use "Slow Cinema Orbit" but change direction to "left"

---

## Troubleshooting Presets

### Preset Not Applying

**Problem:** Settings don't change when selecting preset

**Solution:**
- Make sure you're NOT on "custom"
- Check debug mode to confirm preset is active
- Restart ComfyUI if node is cached

---

### Results Don't Match Expectations

**Problem:** Output doesn't look like preset description

**Solution:**
- Enable debug_mode to verify settings
- Check subject is centered in frame
- Verify scene context isn't conflicting
- Use appropriate subject type

---

### Multi-Step Not Generating Correctly

**Problem:** Only getting 1 frame instead of multiple

**Solution:**
- Preset automatically enables multi_step_mode
- Use the "multi_step_prompts" output
- Parse and iterate through each frame prompt
- Check frame_count output matches expected

---

## Conclusion

The 19 cinematography presets cover virtually every object rotation use case:
- **5 presets** for product photography
- **4 presets** for architecture & real estate
- **4 presets** for cinematic & professional work
- **3 presets** for quick & single frame shots
- **3 presets** for special effects & social media

Simply select a preset and let it handle the complex configuration automatically!

---

**Need more control?** Select "custom" and adjust all parameters manually.

**Want to learn more?** Enable debug_mode to see how presets configure settings.

**Have a specific use case?** Check the Use Case Recommendations section for guidance.

---

**For more presets, workflows, and support:**
- **Patreon:** [patreon.com/archai3d](https://patreon.com/archai3d)
- **GitHub:** [github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen](https://github.com/amir84ferdos/ComfyUI-ArchAi3d-Qwen)

**Happy rotating! 🎥**
