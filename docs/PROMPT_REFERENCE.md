# Qwen Camera Control - Quick Prompt Reference

## 🎯 Most Reliable Prompts (Based on Community Testing)

### ⭐ ORBIT AROUND (Most Reliable for Rotation)

```
camera orbit left around SUBJECT by 45 degrees
camera orbit left around SUBJECT by 90 degrees
camera orbit right around SUBJECT by 45 degrees
camera orbit right around SUBJECT by 90 degrees
camera orbit up around SUBJECT by 45 degrees
```

### 📍 VANTAGE POINT (Best for Position Changes)

```
change the view to a new vantage point 10m to the left
change the view to a new vantage point 10m to the right
change the view to a new vantage point at the left side of the room
change the view to a new vantage point at the right side of the room
```

### 🎥 TILT (Camera Angle)

```
change the view and tilt the camera up slightly
change the view and tilt the camera down slightly
```

### 🔄 COMBINED MOVEMENT

```
change the view and move the camera up while tilting it down slightly
change the view and move the camera down while tilting it up slightly
change the view and move the camera way left while tilting it right
change the view and move the camera way right while tilting it left
```

### 👁️ SPECIAL VIEWS

```
view from ground level, worm's eye view
change the view to a vantage point at ground level camera tilted way up towards the ceiling
extreme bottom up view
closeup shot from her feet level camera aiming upwards to her face
```

### 🎬 HEIGHT VARIATIONS

```
change the view to a lower vantage point camera is tilted up
change the view to a higher vantage point camera tilted down slightly
change the view to a lower vantage point camera is at her face level
```

### 🔭 FOV (Field of View)

```
change the view to wide 100 degrees FOV
change the view to ultrawide 180 degrees FOV shot on ultrawide lens more of the scene fits the view
change the view to fisheye 180 fov
change the view to ultrawide fisheye lens
```

---

## 📐 Prompt Parameters Quick Guide

### Directions
- **left** = picture left (not subject's left!)
- **right** = picture right (not subject's right!)
- **up** = upward camera movement
- **down** = downward camera movement

### Angles (for Orbits)
- **45 degrees** = Small rotation (reliable)
- **90 degrees** = Quarter turn (orbits more than 45, but consistent)
- **180 degrees** = Opposite view
- **360 degrees** = Full rotation (use multi-step)

### Distances (for Vantage Points)
- **5m** = Close movement
- **10m** = Medium movement (recommended)
- **15-20m** = Large movement

### Heights
- **ground level** = Worm's eye view
- **face level** = Eye level with subject
- **lower** = Below current level
- **higher** = Above current level

### FOV Values
- **100 degrees** = Wide angle
- **180 degrees** = Ultrawide/Fisheye
- **normal** = Standard lens

---

## ✅ Do's and ❌ Don'ts

### ✅ DO:

- Use "orbit around" for rotations
- Add scene descriptions for context
- Use environment-only scenes when possible
- Center subjects before rotating
- Use distance-based positioning (meters)
- Use "dolly" for zoom operations

### ❌ DON'T:

- Don't use vague terms like "rotate" or "turn"
- Don't expect exact angles (direction is consistent, angle may vary)
- Don't rotate around off-center people
- Don't use complex multi-axis movements in one prompt
- Don't forget: left/right is picture left/right, not subject's perspective

---

## 🎨 Scene-Specific Tips

### Interior Scenes
```
[Room description] change the view to a new vantage point 10m to the right
modern living room with sectional sofa change the view to a new vantage point at the left side of the room
```

### Exterior Scenes
```
exterior architectural view camera orbit right around the building by 90 degrees
building facade sunny day change the view to ultrawide 180 degrees FOV
```

### Product Photography
```
product on white background camera orbit right around the product by 45 degrees maintaining distance
studio setup with soft lighting camera orbit left around the product by 90 degrees keeping camera level
```

### Architectural Visualization
```
interior hallway view change the view to ground level camera tilted way up towards the ceiling
lobby space with high ceiling change the view to a new vantage point 10m to the right camera tilted up slightly
```

---

## 🔢 10 Test Prompts for Rotation (Left/Right)

Based on your original request, here are 10 different prompts for testing rotation:

### Horizontal Rotations (Left/Right)

1. **"camera orbit right around SUBJECT by 45 degrees"**
   - Small rotation clockwise

2. **"camera orbit left around SUBJECT by 90 degrees"**
   - Quarter turn counterclockwise

3. **"change the view to a new vantage point 10m to the right"**
   - Position-based right movement

4. **"change the view to a new vantage point at the left side of the room"**
   - Move to left side completely

5. **"change the view and move the camera way right while tilting it left"**
   - Combined right movement with counter-tilt

### Variations with Context

6. **"interior view camera orbit right around the furniture by 90 degrees"**
   - Context-aware right rotation

7. **"change the view dolly in and orbit left around SUBJECT by 45 degrees"**
   - Zoom + left rotation combo

8. **"camera orbit right around SUBJECT by 180 degrees maintaining distance"**
   - Opposite view, right direction

9. **"change the view to a new vantage point 5m to the left camera at same level"**
   - Left movement, level camera

10. **"exterior view camera orbit left around the building by 90 degrees keeping camera level"**
    - Architectural left rotation

---

## 📊 Reliability Rating

| Prompt Type | Reliability | Best Use Case |
|-------------|-------------|---------------|
| orbit around | ⭐⭐⭐⭐⭐ | Object rotation |
| vantage point | ⭐⭐⭐⭐⭐ | Position changes |
| dolly | ⭐⭐⭐⭐⭐ | Zoom in/out |
| tilt | ⭐⭐⭐⭐ | Vertical angle |
| combined movement | ⭐⭐⭐ | Complex motion |
| FOV change | ⭐⭐⭐⭐ | Lens effects |
| rotate/turn | ⭐⭐ | Avoid, use orbit instead |

---

## 🚀 Quick Start Template

```
[SCENE DESCRIPTION] + [CAMERA ACTION] + [PARAMETERS] + [MODIFIERS]

Example:
"modern living room with glass coffee table camera orbit right around SUBJECT by 90 degrees maintaining distance keeping camera level"
```

---

**Generated by ArchAi3D Qwen Camera Control Nodes v2.0.0**
