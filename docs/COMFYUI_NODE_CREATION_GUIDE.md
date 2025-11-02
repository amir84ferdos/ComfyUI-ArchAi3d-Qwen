# ComfyUI Node Creation Guide

**Complete guide for adding custom nodes to ComfyUI-ArchAi3d-Qwen**

**Version:** 1.0.0
**Created:** 2025-10-17
**Based on:** Working patterns from ArchAi3d Qwen nodes

---

## Overview

This guide shows you the **exact pattern** for creating custom nodes that work with ComfyUI. All examples are based on **working nodes** from this project.

---

## The Winning Pattern

After testing multiple approaches, this is the **proven pattern** that works:

```
1. Create node file with comfy_api.latest imports
2. Use io.ComfyNode base class
3. Define schema with define_schema() classmethod
4. Implement execute() classmethod returning io.NodeOutput
5. Add Extension class with comfy_entrypoint()
6. Import and register in main __init__.py
```

This is a **hybrid approach**: node uses modern ComfyAPI (`comfy_entrypoint()`) BUT is also registered in the main `__init__.py` for compatibility.

---

## Step-by-Step: Creating a New Node

### Step 1: Create Node File

**Location:**
```
E:\Comfy\Qwen\ComfyUI-Easy-Install\ComfyUI\custom_nodes\ComfyUI-ArchAi3d-Qwen\nodes\
└── [category]/
    └── your_node_name.py
```

**Categories:**
- `camera/` - Camera control nodes
- `editing/` - Image editing nodes
- `utils/` - Utility nodes
- `core/` - Core functionality

**File structure:**
```python
# -*- coding: utf-8 -*-
"""
Your Node Name
Author: Your Name
Version: 1.0.0
Created: YYYY-MM-DD

Description:
    What your node does...
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# Your code here...
```

---

### Step 2: Define Your Node Class

**Pattern:**

```python
class Your_Node_Name(io.ComfyNode):
    """
    Brief description of your node.

    What it does, when to use it, etc.
    """

    @classmethod
    def define_schema(cls):
        """Define node schema for ComfyUI."""
        return io.Schema(
            node_id="Your_Node_Name",  # Must match class name
            category="ArchAi3d/Category",  # Submenu path
            inputs=[
                # Define inputs here
            ],
            outputs=[
                # Define outputs here
            ],
        )

    @classmethod
    def execute(cls, param1, param2, ...) -> io.NodeOutput:
        """
        Execute the node logic.

        Args:
            param1: Description of param1
            param2: Description of param2

        Returns:
            io.NodeOutput with your results
        """
        # Your logic here

        result = "your result"

        return io.NodeOutput(result)
```

---

### Step 3: Define Inputs

**Available input types:**

```python
inputs=[
    # String (single line)
    io.String.Input(
        "parameter_name",
        default="",
        tooltip="Help text for user"
    ),

    # String (multiline)
    io.String.Input(
        "parameter_name",
        multiline=True,
        default="",
        tooltip="Help text for user"
    ),

    # Combo/Dropdown
    io.Combo.Input(
        "parameter_name",
        options=["option1", "option2", "option3"],
        default="option1",
        tooltip="Help text for user"
    ),

    # Boolean/Checkbox
    io.Boolean.Input(
        "parameter_name",
        default=False,
        tooltip="Help text for user"
    ),

    # Integer
    io.Int.Input(
        "parameter_name",
        default=10,
        min=0,
        max=100,
        tooltip="Help text for user"
    ),

    # Float
    io.Float.Input(
        "parameter_name",
        default=0.5,
        min=0.0,
        max=1.0,
        tooltip="Help text for user"
    ),

    # Image
    io.Image.Input(
        "parameter_name",
        tooltip="Help text for user"
    ),

    # Mask
    io.Mask.Input(
        "parameter_name",
        tooltip="Help text for user"
    ),
]
```

---

### Step 4: Define Outputs

**Available output types:**

```python
outputs=[
    # String output
    io.String.Output(
        "output_name",
        tooltip="Description of this output"
    ),

    # Image output
    io.Image.Output(
        "output_name",
        tooltip="Description of this output"
    ),

    # Mask output
    io.Mask.Output(
        "output_name",
        tooltip="Description of this output"
    ),

    # Integer output
    io.Int.Output(
        "output_name",
        tooltip="Description of this output"
    ),

    # Float output
    io.Float.Output(
        "output_name",
        tooltip="Description of this output"
    ),
]
```

---

### Step 5: Implement Execute Method

**Pattern:**

```python
@classmethod
def execute(cls, param1, param2, param3) -> io.NodeOutput:
    """
    Execute the node.

    Args must match input names from define_schema().
    Return must be io.NodeOutput with values matching output order.
    """

    # Your logic here
    result1 = f"Processing {param1}"
    result2 = param2 * 2

    # Optional: Debug output
    print(f"Debug: {param1}, {param2}")

    # Return outputs in same order as defined in schema
    return io.NodeOutput(result1, result2)
```

**Important rules:**
1. Method MUST be `@classmethod`
2. Args MUST match input names exactly
3. Return MUST be `io.NodeOutput(...)`
4. Output values MUST match order in schema

---

### Step 6: Add Extension Registration

**At the end of your node file:**

```python
# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class YourNodeExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [Your_Node_Name]


async def comfy_entrypoint():
    return YourNodeExtension()
```

**Pattern:**
- Extension class name: `<YourNode>Extension`
- Returns list with your node class
- `comfy_entrypoint()` function returns extension instance

---

### Step 7: Register in Main __init__.py

**File:** `E:\Comfy\Qwen\ComfyUI-Easy-Install\ComfyUI\custom_nodes\ComfyUI-ArchAi3d-Qwen\__init__.py`

**Add import:**

```python
# In the appropriate section (camera/editing/utils/etc)
from .nodes.category.your_node_file import Your_Node_Name
```

**Add to NODE_CLASS_MAPPINGS:**

```python
NODE_CLASS_MAPPINGS = {
    # ... existing nodes ...

    # Your category
    "Your_Node_Name": Your_Node_Name,
}
```

**Add to NODE_DISPLAY_NAME_MAPPINGS:**

```python
NODE_DISPLAY_NAME_MAPPINGS = {
    # ... existing nodes ...

    # Your category
    "Your_Node_Name": "📝 Your Display Name",  # Use emoji for visual clarity
}
```

**Update startup message (optional):**

```python
print(f"  📝 Your Category: X nodes")
```

---

## Complete Working Example

### Example: Simple Text Processor Node

**File:** `nodes/utils/archai3d_text_processor.py`

```python
# -*- coding: utf-8 -*-
"""
ArchAi3D Text Processor Node
Author: ArchAi3d
Version: 1.0.0
Created: 2025-10-17

Description:
    Simple text processing utility.
    Converts text to uppercase and counts characters.
"""

from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class ArchAi3D_Text_Processor(io.ComfyNode):
    """
    Text processing utility node.

    Converts input text to uppercase and returns character count.
    """

    @classmethod
    def define_schema(cls):
        """Define node schema for ComfyUI."""
        return io.Schema(
            node_id="ArchAi3D_Text_Processor",
            category="ArchAi3d/Utils",
            inputs=[
                io.String.Input(
                    "input_text",
                    multiline=True,
                    default="",
                    tooltip="Enter text to process"
                ),
                io.Boolean.Input(
                    "debug_mode",
                    default=False,
                    tooltip="Print debug information"
                ),
            ],
            outputs=[
                io.String.Output(
                    "uppercase_text",
                    tooltip="Text converted to uppercase"
                ),
                io.Int.Output(
                    "character_count",
                    tooltip="Number of characters"
                ),
            ],
        )

    @classmethod
    def execute(cls, input_text, debug_mode) -> io.NodeOutput:
        """
        Process the input text.

        Args:
            input_text: Text to process
            debug_mode: Whether to print debug info

        Returns:
            io.NodeOutput(uppercase_text, character_count)
        """
        # Process
        uppercase_text = input_text.upper()
        character_count = len(input_text)

        # Debug
        if debug_mode:
            print("=" * 70)
            print("ArchAi3D_Text_Processor - Debug")
            print("=" * 70)
            print(f"Input: {input_text}")
            print(f"Output: {uppercase_text}")
            print(f"Count: {character_count}")
            print("=" * 70)

        # Return
        return io.NodeOutput(uppercase_text, character_count)


# ============================================================================
# EXTENSION REGISTRATION
# ============================================================================

class TextProcessorExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [ArchAi3D_Text_Processor]


async def comfy_entrypoint():
    return TextProcessorExtension()
```

**In main __init__.py:**

```python
# Import section
from .nodes.utils.archai3d_text_processor import ArchAi3D_Text_Processor

# NODE_CLASS_MAPPINGS
NODE_CLASS_MAPPINGS = {
    # ... other nodes ...
    "ArchAi3D_Text_Processor": ArchAi3D_Text_Processor,
}

# NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {
    # ... other nodes ...
    "ArchAi3D_Text_Processor": "📝 Text Processor",
}
```

**Result:** Node appears at `ArchAi3d → Utils → 📝 Text Processor`

---

## Common Mistakes & Fixes

### Mistake 1: Wrong Import

❌ **Wrong:**
```python
from comfy_io import io
```

✅ **Correct:**
```python
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
```

---

### Mistake 2: Missing @classmethod

❌ **Wrong:**
```python
def execute(self, param1):
    return io.NodeOutput(result)
```

✅ **Correct:**
```python
@classmethod
def execute(cls, param1):
    return io.NodeOutput(result)
```

---

### Mistake 3: Wrong Return Type

❌ **Wrong:**
```python
return (result,)  # Tuple
return result     # Direct value
```

✅ **Correct:**
```python
return io.NodeOutput(result)           # Single output
return io.NodeOutput(result1, result2)  # Multiple outputs
```

---

### Mistake 4: Mismatched Argument Names

❌ **Wrong:**
```python
# Schema:
inputs=[io.String.Input("input_text", ...)]

# Execute:
def execute(cls, text):  # Different name!
```

✅ **Correct:**
```python
# Schema:
inputs=[io.String.Input("input_text", ...)]

# Execute:
def execute(cls, input_text):  # Same name!
```

---

### Mistake 5: Forgot Extension Registration

❌ **Wrong:**
```python
# Just the node class, no extension
class My_Node(io.ComfyNode):
    ...
# File ends here - MISSING EXTENSION!
```

✅ **Correct:**
```python
class My_Node(io.ComfyNode):
    ...

# Extension registration at end of file
class MyNodeExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [My_Node]

async def comfy_entrypoint():
    return MyNodeExtension()
```

---

### Mistake 6: Forgot Main __init__.py Registration

❌ **Wrong:**
```python
# Only created node file, didn't update __init__.py
```

✅ **Correct:**
```python
# In main __init__.py:
from .nodes.utils.my_node import My_Node

NODE_CLASS_MAPPINGS = {
    "My_Node": My_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "My_Node": "📝 My Node",
}
```

---

## Checklist: Adding a New Node

Use this checklist every time you add a new node:

- [ ] **1. Create node file** in appropriate category folder
- [ ] **2. Add imports:** `from comfy_api.latest import ComfyExtension, io` and `from typing_extensions import override`
- [ ] **3. Create node class** inheriting from `io.ComfyNode`
- [ ] **4. Add `define_schema()` classmethod** with `node_id`, `category`, `inputs`, `outputs`
- [ ] **5. Add `execute()` classmethod** with matching parameter names
- [ ] **6. Return `io.NodeOutput(...)`** with results in correct order
- [ ] **7. Add Extension class** at end of file
- [ ] **8. Add `comfy_entrypoint()` function** returning extension
- [ ] **9. Import node** in main `__init__.py`
- [ ] **10. Add to `NODE_CLASS_MAPPINGS`** dictionary
- [ ] **11. Add to `NODE_DISPLAY_NAME_MAPPINGS`** dictionary with emoji
- [ ] **12. Test:** Restart ComfyUI and verify node appears

---

## Testing Your Node

### Step 1: Restart ComfyUI

**IMPORTANT:** You MUST restart ComfyUI completely after:
- Creating new node files
- Changing imports
- Changing registration

Simply refreshing the browser is NOT enough!

---

### Step 2: Check Console for Errors

Look for:
```
Cannot import ... module for custom nodes
ModuleNotFoundError
ImportError
```

If you see errors, check:
1. Import statements
2. Class name matches in all places
3. Extension registration exists
4. Main __init__.py registration exists

---

### Step 3: Find Your Node

Navigate to:
```
ComfyUI Node Browser → ArchAi3d → [Your Category] → [Your Node]
```

If not visible:
1. Check console for import errors
2. Verify `node_id` in schema
3. Verify `category` path
4. Verify registration in __init__.py
5. Restart ComfyUI again

---

### Step 4: Test Functionality

1. Add node to workflow
2. Connect inputs
3. Run workflow
4. Check outputs
5. Check console for debug messages

---

## Real Examples from This Project

### Example 1: Mask to Position Guide

**File:** [archai3d_mask_to_position_guide.py](E:\Comfy\Qwen\ComfyUI-Easy-Install\ComfyUI\custom_nodes\ComfyUI-ArchAi3d-Qwen\nodes\utils\archai3d_mask_to_position_guide.py)

**Pattern:**
- Uses `io.ComfyNode` base class
- Has `define_schema()` with `node_id="ArchAi3D_Mask_To_Position_Guide"`
- Has `execute()` classmethod returning `io.NodeOutput(...)`
- Has Extension class and `comfy_entrypoint()`
- Registered in main `__init__.py`

**Result:** ✅ **Working**

---

### Example 2: Interior View Control

**File:** [archai3d_qwen_interior_view_control.py](E:\Comfy\Qwen\ComfyUI-Easy-Install\ComfyUI\custom_nodes\ComfyUI-ArchAi3d-Qwen\nodes\camera\archai3d_qwen_interior_view_control.py)

**Pattern:**
- Uses `io.ComfyNode` base class
- Has `define_schema()` with `node_id="ArchAi3D_Qwen_Interior_View_Control"`
- Has `execute()` classmethod returning `io.NodeOutput(...)`
- Has Extension class and `comfy_entrypoint()`
- Registered in main `__init__.py`

**Result:** ✅ **Working**

---

### Example 3: Position Guide Prompt Builder

**File:** [archai3d_position_guide_prompt_builder.py](E:\Comfy\Qwen\ComfyUI-Easy-Install\ComfyUI\custom_nodes\ComfyUI-ArchAi3d-Qwen\nodes\utils\archai3d_position_guide_prompt_builder.py)

**Pattern:**
- Uses `io.ComfyNode` base class
- Has `define_schema()` with `node_id="ArchAi3D_Position_Guide_Prompt_Builder"`
- Has `execute()` classmethod returning `io.NodeOutput(...)`
- Has Extension class and `comfy_entrypoint()`
- Registered in main `__init__.py`

**Result:** ✅ **Working** (after following this guide)

---

## Troubleshooting Guide

### Problem: "ModuleNotFoundError: No module named 'comfy_io'"

**Cause:** Wrong import statement

**Fix:**
```python
# Change this:
from comfy_io import io

# To this:
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override
```

---

### Problem: Node not visible in ComfyUI

**Possible causes:**

1. **Not registered in main __init__.py**
   - Check import statement exists
   - Check NODE_CLASS_MAPPINGS entry exists
   - Check NODE_DISPLAY_NAME_MAPPINGS entry exists

2. **Import error**
   - Check console for errors
   - Verify file path is correct
   - Verify class name matches everywhere

3. **Forgot to restart ComfyUI**
   - Close and reopen ComfyUI completely
   - Browser refresh is NOT enough

4. **Wrong category path**
   - Check `category="ArchAi3d/Category"` in schema
   - Verify category exists in ComfyUI

---

### Problem: "TypeError: execute() missing required argument"

**Cause:** Parameter names don't match between schema and execute method

**Fix:**
```python
# Schema inputs MUST match execute parameters

# Schema:
inputs=[
    io.String.Input("param1", ...),
    io.String.Input("param2", ...),
]

# Execute:
def execute(cls, param1, param2):  # Names must match!
    ...
```

---

### Problem: "TypeError: NodeOutput() argument after * must be an iterable"

**Cause:** Wrong return type

**Fix:**
```python
# Wrong:
return (result,)
return result

# Correct:
return io.NodeOutput(result)
return io.NodeOutput(result1, result2, result3)
```

---

## Advanced Topics

### Multiple Outputs

```python
# Schema:
outputs=[
    io.String.Output("text"),
    io.Int.Output("count"),
    io.Boolean.Output("success"),
]

# Execute:
def execute(cls, input_text):
    text = input_text.upper()
    count = len(input_text)
    success = count > 0

    # Return in SAME ORDER as schema
    return io.NodeOutput(text, count, success)
```

---

### Optional Parameters

```python
# Schema:
inputs=[
    io.String.Input("required_param", default=""),
    io.String.Input("optional_param", default="", tooltip="Optional..."),
]

# Execute:
def execute(cls, required_param, optional_param):
    if optional_param:
        # Use optional param
        result = f"{required_param} + {optional_param}"
    else:
        # Optional not provided
        result = required_param

    return io.NodeOutput(result)
```

---

### Debug Mode Pattern

```python
# Schema:
inputs=[
    # ... your inputs ...
    io.Boolean.Input("debug_mode", default=False, tooltip="Print debug info"),
]

# Execute:
def execute(cls, param1, param2, debug_mode):
    # Your logic
    result = process(param1, param2)

    # Debug output
    if debug_mode:
        print("=" * 70)
        print("Node Name - Debug Output")
        print("=" * 70)
        print(f"Param1: {param1}")
        print(f"Param2: {param2}")
        print(f"Result: {result}")
        print("=" * 70)

    return io.NodeOutput(result)
```

---

## Node Naming Conventions

### Class Names

**Pattern:** `ArchAi3D_<Category>_<Name>`

**Examples:**
- `ArchAi3D_Qwen_Interior_View_Control`
- `ArchAi3D_Mask_To_Position_Guide`
- `ArchAi3D_Position_Guide_Prompt_Builder`

**Rules:**
- Start with `ArchAi3D_`
- Use underscores for spaces
- Use title case for each word
- Be descriptive but concise

---

### File Names

**Pattern:** `archai3d_<category>_<name>.py`

**Examples:**
- `archai3d_qwen_interior_view_control.py`
- `archai3d_mask_to_position_guide.py`
- `archai3d_position_guide_prompt_builder.py`

**Rules:**
- All lowercase
- Use underscores for spaces
- Match class name (but lowercase)

---

### Display Names

**Pattern:** `"🔸 <Human Readable Name>"`

**Examples:**
- `"🏠 Interior View Control"`
- `"🎯 Mask to Position Guide"`
- `"📝 Position Guide Prompt Builder"`

**Rules:**
- Include emoji for visual clarity
- Use title case
- Be user-friendly
- Keep concise

---

## Summary

### The Winning Pattern (TL;DR)

```python
# 1. File structure
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

# 2. Node class
class Your_Node(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Your_Node",
            category="ArchAi3d/Category",
            inputs=[...],
            outputs=[...],
        )

    @classmethod
    def execute(cls, param1, param2) -> io.NodeOutput:
        result = process(param1, param2)
        return io.NodeOutput(result)

# 3. Extension registration
class YourNodeExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [Your_Node]

async def comfy_entrypoint():
    return YourNodeExtension()

# 4. Main __init__.py
from .nodes.category.your_node import Your_Node

NODE_CLASS_MAPPINGS = {"Your_Node": Your_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"Your_Node": "📝 Your Node"}

# 5. Restart ComfyUI!
```

---

## Additional Resources

- **ComfyAPI Documentation:** Check latest ComfyUI API docs
- **Working Examples:** See `nodes/camera/` and `nodes/utils/` folders
- **This Project's Nodes:** 30+ working examples to reference

---

**Author:** Amir Ferdos (ArchAi3d)
**Email:** Amir84ferdos@gmail.com
**LinkedIn:** https://www.linkedin.com/in/archai3d/

*Last Updated: 2025-10-17*
