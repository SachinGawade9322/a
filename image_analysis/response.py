# import json
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "Qwen/Qwen3-4B"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

# def analyze_part(json_path: str) -> str:
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     analysis = data.get("gemini_analysis", {})
#     drawing_analysis = analysis.get("drawing_analysis", {})
#     symmetry_analysis = analysis.get("symmetry_analysis", {})
#     technical_features = analysis.get("technical_features", {})
#     summary = analysis.get("summary", {})
#     detected_objects = analysis.get("detected_objects", [])

#     title_block_object = next(
#         (obj for obj in detected_objects if obj.get("type") == "mechanical_part"),
#         detected_objects[0] if detected_objects else {}
#     )

#     unique_feature_descriptions = sorted(set(
#         obj.get("description", "Unknown Feature") for obj in detected_objects
#     ))

#     part_description = f"""
# 🔍 **PART DRAWING ANALYSIS SUMMARY**

# 📌 **Title Block Info**: {title_block_object.get('title', 'N/A')}
# 🧱 **Material**: {drawing_analysis.get('material_specifications', 'N/A')}
# 🧭 **View Type**: {drawing_analysis.get('view_type', 'N/A')}
# 📏 **Scale**: {drawing_analysis.get('scale', 'N/A')}

# 🔄 **Symmetry**: {symmetry_analysis.get('symmetry_description', 'N/A')}

# 🧩 **Detected Features**: {len(detected_objects)} total
# • {', '.join(unique_feature_descriptions)}

# 📐 **Technical Notes**: {technical_features.get('dimension_lines', 'N/A')}
# 🏭 **Manufacturing Notes**: {summary.get('completeness', 'N/A')}
# """

#     messages = [
#         {"role": "user", "content": part_description}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=True
#     )

#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=1500,
#         do_sample=False,
#         temperature=0.65
#     )

#     output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
#     try:
#         index = len(output_ids) - output_ids[::-1].index(151668)
#     except ValueError:
#         index = 0

#     content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
#     return content

# if __name__ == "__main__":
#     json_path = r"E:\datta_enterprises\detect\image_2_combined_analysis.json"
#     result = analyze_part(json_path)
#     print("\nFinal content:\n", result)
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen3-4B"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    quantization_config=bnb_config,
    device_map="auto"
)

def analyze_part(json_path: str) -> str:
    with open(json_path, "r") as f:
        data = json.load(f)

    analysis = data.get("gemini_analysis", {})
    meta = analysis.get("meta_data", {})
    features = analysis.get("features", [])
    shapes = data.get("shapes_summary", {}).get("shapes", [])

    # 🧾 Build full prompt with all available meta info
    prompt = f"""
📌 **TITLE BLOCK SUMMARY**

• **Title**: {meta.get("title", "N/A")}
• **Drawing Number**: {meta.get("drawing_number", "N/A")}
• **Material**: {meta.get("material", "N/A")}
• **Drawing Type**: {meta.get("drawing_type", "N/A")}
• **Projection Method**: {meta.get("projection_method", "N/A")}
• **Component Weight**: {meta.get("component_weight", "N/A")}
• **Coating**: {meta.get("coating", "N/A")}
• **Finish**: {meta.get("finish", "N/A")}
• **Units**: {meta.get("units", "N/A")}
• **Scale**: {meta.get("scale", "N/A")}
• **Drawing Complexity**: {meta.get("drawing_complexity", "N/A")}

📝 **NOTES**
{meta.get("notes", "None")}

📋 **INSTRUCTIONS**
{meta.get("instructions", "None")}

🔧 **GEOMETRIC FEATURES** ({len(features)} detected)
"""

    # 📐 Geometry Summary
    for i, feat in enumerate(features, 1):
        ftype = feat.get("type", "Unknown").upper()
        desc = feat.get("description", "N/A")
        dims = []

        for key in ['diameter', 'depth', 'width', 'length', 'size', 'radius', 'thickness']:
            val = feat.get(key)
            if val:
                dims.append(f"{key.capitalize()}: {val}")

        count = feat.get("count")
        if count:
            dims.append(f"Count: {count}")

        dim_str = "; ".join(dims) if dims else "No dimensions provided"

        prompt += f"\n  {i}. **{ftype}** — {desc}\n     → {dim_str}"

    if shapes:
        prompt += f"\n\n🔺 **Detected Shapes**: {', '.join(shapes)}"

    messages = [{"role": "user", "content": prompt.strip()}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1200,
        do_sample=False,
        temperature=0.6
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return content

if __name__ == "__main__":
    json_path = r"E:\datta_enterprises\AI_Models\detect\image_8_combined_analysis.json"
    result = analyze_part(json_path)
    print("\nFinal Generative Output:\n", result)
