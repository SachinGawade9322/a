import json
import os
from typing import Dict, List, Any
import google.generativeai as genai
from PIL import Image
from collections import defaultdict
from datetime import datetime
from PIL import ImageDraw, ImageFont

class GeminiCADDetector: 
    def __init__(self, api_key: str):
        """Initialize the detector with Gemini API"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

        self.output_folder = "detect"
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created folder: {self.output_folder}")
        
        self.component_categories = {
            'structural_components': [
                'flat plate',
                'housing',
                'plate',
                'rectangular profile',
                'asymmetric contoured part',
                'enclosure',
                'pad',
                'support frame'
            ],
            'mechanical_features': [
                'groove',
                'chamfer',
                'fillet',
                'hole',
                'tapped hole',
                'precision hole',
                'slot',
                'rib',
                'thread',
                'step',
                'counterbore',
                'mounting flange',
                'gasket sealing groove',
                'deburring chamfer',
                'threaded section',
                'boss',
                'tap'
            ],
            'fasteners': [
                'screw locating feature',
                'bolt',
                'nut',
                'rivet',
                'stud',
                'washer',
                'clip',
                'pin',
                'retaining ring'
            ],
            'mechanical_parts': [
                'shaft',
                'roller',
                'cam',
                'gear',
                'pulley',
                'lever',
                'axle',
                'spindle',
                'rod',
                'tube'
            ],
            'manufacturing_features': [
                'symmetry',
                'section view',
                'detail view',
                'dimension line',
                'center line',
                'leader line',
                'extension line',
                'hatch pattern',
                'projection method'
            ],
            'basic_shapes': [
                'circle',
                'rectangle',
                'square',
                'triangle',
                'polygon',
                'ellipse',
                'arc',
                'line',
                'polyline',
                'spline'
            ],
            'joints_connections': [
                'flange',
                'coupling',
                'union',
                'adapter',
                'connector',
                'elbow',
                'tee',
                'reducer'
            ],
            'others': [
                'cover',
                'cylindrical form',
                'surface finish',
                'curved profile',
                'contoured sheet',
                'flat surface',
                'projection symbol',
                'machined part',
                'burr removal',
                'radius',
                'detail label',
                'm6',
                'm8',
                'fit tolerance'
            ]
        }

    def create_detection_prompt(self) -> str:
        prompt = """
    You are a highly experienced mechanical engineer and computer vision expert.

    Your task is to analyze a 2D mechanical CAD drawing image and return structured JSON output.

    Instructions:
    1. Detect and classify **all mechanical parts** — including:
    - Main components (e.g., flanges, rollers, plates)
    - Subcomponents or detail sections (e.g., Detail A, Section B-B, zoomed views)

    2. Classify each part using the following categories:
    - structural_components
    - mechanical_features
    - fasteners
    - bearings_seals
    - mechanical_parts
    - joints_connections
    - manufacturing_features
    - basic_shapes
    - others

    3. For every detected part:
    - Use a unique `"object_id"` (e.g., `MD_FLANGE_180`)
    - Assign a `"type"` (e.g., `mechanical_part`, `mechanical_feature`)
    - Assign a `"category"` from the categories listed above
    - Include `"description"`, `"material"`, `"dimensions"`, `"position"`, `"confidence"`
    - If the part is from a zoomed or detail section, add a `"parent_object"` field

    4. In `"features"`, list internal features like holes, slots, chamfers, ribs, etc.

    Return your result in the following JSON format:

    {
    "drawing_metadata": {
        "drawing_number": "Extracted from title block",
        "component_weight": "e.g., 247.26G (you have to find out the exact weight from the drawing)",
        "projection_method": "e.g., 3rd Angle",
        "unit_symmetry": "e.g., Bilateral, Radial, None (see the symmetry which you are going to find it should be accurate)"
    },
    "parts_detected": [
        {
        "object_id": "MD_CHUTE_FLANGE_180",
        "type": "mechanical_part",
        "category": "mechanical_parts",
        "description": "Main component flange",
        "material": "Acetal-White (listen this is the main feature okay so you have to find the accurate material from the drawing, without messing and without making any mistakes)",
        "dimensions": {
            "length": "optional",
            "width": "optional",
            "diameter": "optional",
            "radius": "optional",
            "height": "optional"
        (listen this is the main feature so you have to find the accurate dimensions from the drawing, without messing and without making any mistakes, if there is no dimensions then you can leave it as optional, but most of the time there will be dimensions in the drawing, so you have to detect it accurately),
        },
        "features": [
            {
            "type": "slot",
            "length": "23",
            "width": "11",
            "radius": "5.5"
            },
            {
            "type": "hole",
            "diameter": "180"
            }
        (listen this is also the important part okay, so we provide you the above features okay in component_categories okay so ya you have to detect it accurately okay, the shape and all you have to find out okay and you have to detect the all dimensions and all the features accurately, without messing and without making any mistakes),
        ],
        "position": {
            "x": 250,
            "y": 300
        },
        "confidence": 0.95
        },
        {
        "object_id": "DETAIL_A_SLOT",
        "type": "mechanical_feature",
        "category": "mechanical_features",
        "description": "Slot in Detail A",
        "dimensions": {
            "length": "23",
            "width": "11",
            "radius": "5.5"
        },
        "position": {
            "x": 700,
            "y": 400
        },
        "parent_object": "MD_CHUTE_FLANGE_180",
        "confidence": 0.85
        }
    ],
    "shapes_detected": [
    {
      "shape_type": "circle/rectangle/etc",
      "count": 5,
      "characteristics": "diameter=120, evenly spaced"
    }
  ],
  "symmetry_analysis": {
    "has_rotational_symmetry": true,
    "rotational_order": 3,
    "has_bilateral_symmetry": true,
    "symmetry_axes": ["vertical", "horizontal"],
    "has_radial_symmetry": true,
    "has_point_symmetry": false,
    "symmetry_description": "The flange shows 120-degree rotational symmetry with mirrored detail holes."
  },
}
  "other_metadata": {
      "Notes": "Include any additional notes or observations about the drawing",
      "Instructions": "find any other insturctions or notes in the drawing that may be useful, any type of comment or note that may be useful to the user",
      "BOM" : "If there is any BOM (Bill of Materials) in the drawing, extract it and include it here",
      "Drawing Type": "e.g., 2D, 3D, Assembly, Part, etc.",
      "Drawing Complexity": "e.g., Simple, Moderate, Complex (based on the number of parts and features detected)"
  }
    
    }
    
5. Very Important: Split all output into two distinct types for mapping:

🔹 A. "meta_data" (UI TAB: Meta Data)
  - Extract all information from the title block, top notes, bottom notes, etc.
  - Include:
    - "drawing_number"
    - "component_weight"
    - "material"
    - "projection_method"
    - "title"
    - "coating"
    - "drawing_type"
    - "drawing_complexity"
    - "finish"
    - "units"
    - "scale"
    - "notes"
    - "instructions"
    - "bom" (as structured array)
  - These go to the Meta Data tab

🔹 B. "features" (UI TAB: Features – this is GEOMETRIC FEATURES)
  - Extract all physical and measurable features
  - Include:
    - "type" (hole, slot, chamfer, thread, fillet, rib, etc.)
    - "dimensions": with length, width, diameter, radius, depth, tolerance, etc.
    - "location" (x, y reference)
    - "count" if there are multiple
    - "description"
    - "parent_object" if it's inside a part
    - "symmetry" if applicable
  - These go to the Features tab (geometric only)

⚠️ Do not mix the above two. Keep meta_data and features completely separate in JSON.

    Important:
    - Use consistent object naming
    - Follow category mapping strictly
    - Be specific and accurate with shapes, categories, and symmetry
    - Include symmetry and shapes even if approximate
    - Return **only valid JSON** — no Markdown, no text, no explanations
    - Don't hallucinate materials or weights
    """
        return prompt

    def detect_objects(self, image_path: str) -> Dict[str, Any]:
        """Detect objects using Gemini vision model"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Create detection prompt
            prompt = self.create_detection_prompt()
            
            # Analyze with Gemini
            response = self.model.generate_content([prompt, image])
            
            # Parse JSON response
            try:
                # Extract JSON from response text
                response_text = response.text
                
                # Find JSON content (in case there's extra text)
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_text = response_text[start_idx:end_idx]
                    parsed_response = json.loads(json_text)
                else:
                    # If no JSON found, create structured response from text
                    parsed_response = {
                        "raw_response": response_text,
                        "parsing_error": "Could not extract JSON from response"
                    }
                
            except json.JSONDecodeError as e:
                # If JSON parsing fails, return raw response with error info
                parsed_response = {
                    "raw_response": response.text,
                    "json_error": str(e),
                    "parsing_status": "failed"
                }
            
            # Add metadata
            analysis_result = {
                "image_path": image_path,
                "model_used": "gemini-2.0-flash-exp",
                "analysis_timestamp": datetime.now().isoformat(),
                "gemini_analysis": parsed_response
            }
            
            return analysis_result
            
        except Exception as e:
            return {
                "image_path": image_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    # def process_single_image(self, image_path: str) -> Dict[str, Any]:
    #     """Process a single CAD image and save results in unified format"""
    #     print(f"Processing image: {image_path}")
        
    #     image_name = os.path.splitext(os.path.basename(image_path))[0]
        
    #     # Run detection
    #     full_result = self.detect_objects(image_path)
        
    #     # Run shape + symmetry extractor
    #     simple_result = self.extract_simple_results(full_result)
        
    #     # Combine exactly how you want it
    #     combined_output = {
    #         "image_path": full_result.get("image_path"),
    #         "model_used": full_result.get("model_used"),
    #         "analysis_timestamp": full_result.get("analysis_timestamp"),
    #         "gemini_analysis": full_result.get("gemini_analysis"),
    #         "shapes_summary": {
    #             "shapes": simple_result.get("shapes", []),
    #             "symmetry": simple_result.get("symmetry", {}),
    #             "extraction_timestamp": simple_result.get("extraction_timestamp")
    #         }
    #     }
        
    #     # Save to single file
    #     output_path = os.path.join(self.output_folder, f"{image_name}_combined_analysis.json")
    #     with open(output_path, 'w') as f:
    #         json.dump(combined_output, f, indent=2, default=str)

    #     print(f"Combined analysis saved to: {output_path}")
        
    #     return combined_output
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        print(f"Processing image: {image_path}")

        # Run detection
        full_result = self.detect_objects(image_path)
        simple_result = self.extract_simple_results(full_result)

        combined_output = {
            "image_path": full_result.get("image_path"),
            "model_used": full_result.get("model_used"),
            "analysis_timestamp": full_result.get("analysis_timestamp"),
            "gemini_analysis": full_result.get("gemini_analysis"),
            "shapes_summary": {
                "shapes": simple_result.get("shapes", []),
                "symmetry": simple_result.get("symmetry", {}),
                "extraction_timestamp": simple_result.get("extraction_timestamp")
            }
        }

        # 🟡 Save only to: detect/feature_extraction.json
        output_path = os.path.join(self.output_folder, "feature_extraction.json")

        with open(output_path, 'w') as f:
            json.dump(combined_output, f, indent=2, default=str)

        print(f"✅ Combined analysis saved to: {output_path}")
        return combined_output



    def process_batch(self, image_directory: str) -> Dict[str, Any]:
        """Process multiple CAD images in batch and save to detect folder"""
        results = []
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
        
        # Get all image files
        image_files = [f for f in os.listdir(image_directory) 
                      if f.lower().endswith(supported_formats)]
        
        print(f"Processing {len(image_files)} images using Gemini...")
        
        
        for i, filename in enumerate(image_files):
            image_path = os.path.join(image_directory, filename)
            print(f"Processing {i+1}/{len(image_files)}: {filename}")
            
            result = self.process_single_image(image_path)
            result['filename'] = filename
            result['batch_index'] = i + 1
            
            results.append(result)

            if (i + 1) % 5 == 0:
                temp_batch_path = os.path.join(self.output_folder, f"batch_temp_{i+1}.json")
                with open(temp_batch_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
        
        batch_output_path = os.path.join(self.output_folder, "batch_complete_results.json")
        with open(batch_output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        summary = self.generate_batch_summary(results)
        summary_output_path = os.path.join(self.output_folder, "batch_summary.json")
        with open(summary_output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Batch processing completed!")
        print(f"Complete results saved to: {batch_output_path}")
        print(f"Summary saved to: {summary_output_path}")
        
        return {
            'total_processed': len(results),
            'results': results,
            'summary': summary,
            'output_folder': self.output_folder
        }

    def generate_batch_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics from batch processing"""
        summary = {
            'batch_info': {
                'total_images': len(results),
                'successful_analyses': 0,
                'failed_analyses': 0,
                'processing_timestamp': datetime.now().isoformat()
            },
            'detection_stats': {
                'common_objects': defaultdict(int),
                'common_shapes': defaultdict(int),
                'symmetry_stats': defaultdict(int),
                'drawing_types': defaultdict(int)
            }
        }
        
        for result in results:
            if 'error' in result:
                summary['batch_info']['failed_analyses'] += 1
            else:
                summary['batch_info']['successful_analyses'] += 1
                
                # Extract statistics from Gemini analysis
                gemini_analysis = result.get('gemini_analysis', {})
                
                if isinstance(gemini_analysis, dict):
                    detected_objects = gemini_analysis.get('detected_objects', [])
                    for obj in detected_objects:
                        obj_type = obj.get('type', 'unknown')
                        summary['detection_stats']['common_objects'][obj_type] += 1
                    
                    shapes_detected = gemini_analysis.get('shapes_detected', [])
                    for shape in shapes_detected:
                        shape_type = shape.get('shape_type', 'unknown')
                        summary['detection_stats']['common_shapes'][shape_type] += 1
                    
                    symmetry = gemini_analysis.get('symmetry_analysis', {})
                    for sym_type, has_symmetry in symmetry.items():
                        if has_symmetry and sym_type.startswith('has_'):
                            clean_type = sym_type.replace('has_', '')
                            summary['detection_stats']['symmetry_stats'][clean_type] += 1
           
                    drawing_analysis = gemini_analysis.get('drawing_analysis', {})
                    drawing_type = drawing_analysis.get('drawing_type', 'unknown')
                    summary['detection_stats']['drawing_types'][drawing_type] += 1
        
        summary['detection_stats'] = {
            key: dict(value) for key, value in summary['detection_stats'].items()
        }
        
        return summary

    def extract_simple_results(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract simplified results (shapes and symmetry only)"""
        gemini_analysis = analysis_result.get('gemini_analysis', {})
        
        if isinstance(gemini_analysis, dict):
            # Extract shapes
            shapes_detected = gemini_analysis.get('shapes_detected', [])
            shape_types = []
            for shape in shapes_detected:
                shape_type = shape.get('shape_type', 'unknown')
                count = shape.get('count', 1)
                if isinstance(count, (int, str)) and str(count).isdigit():
                    shape_types.extend([shape_type] * int(count))
                else:
                    shape_types.append(shape_type)

            symmetry_analysis = gemini_analysis.get('symmetry_analysis', {})
            
            return {
                'shapes': shape_types,
                'symmetry': symmetry_analysis,
                'extraction_timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'shapes': [],
                'symmetry': {},
                'note': 'Could not parse Gemini response properly',
                'extraction_timestamp': datetime.now().isoformat()
            }
            
    def annotate_detected_parts(self, image_path, analysis_result, save_path=None):
        try:
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            
            gemini_data = analysis_result.get('gemini_analysis', {})
            parts = gemini_data.get('parts_detected', [])

            font = ImageFont.load_default()
            
            for part in parts:
                pos = part.get('position', {})
                x = int(pos.get('x', 0))
                y = int(pos.get('y', 0))
                label = part.get('object_id', 'Unknown')

                # Estimate box size from dimensions (fallback = 50x50)
                dims = part.get('dimensions', {})
                w = int(float(dims.get('width', 50)) if dims.get('width') else 50)
                h = int(float(dims.get('length', 50)) if dims.get('length') else 50)

                # Draw rectangle and label
                draw.rectangle([(x, y), (x + w, y + h)], outline="red", width=2)
                draw.text((x + 5, y - 10), label, fill="red", font=font)

            # Save annotated image
            output_path = save_path or image_path.replace(".jpg", "_annotated.jpg")
            image.save(output_path)
            print(f"✅ Annotated image saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Annotation failed: {str(e)}")
            return None

# Main execution
if __name__ == "__main__":
    IMAGE_PATH = r"E:\datta_enterprises\AI_Models\Processed_data\Image_data\image_3.jpg"  
    BATCH_FOLDER = None  
    API_KEY = "AIzaSyC1P8T888-np_2xy2gx9coOdXWMmGOnV6o"

    print("Initializing Gemini CAD Detector...")
    detector = GeminiCADDetector(API_KEY)
    
    print(f"Output folder created: {detector.output_folder}")
    
    try:
        if BATCH_FOLDER and os.path.exists(BATCH_FOLDER):
            print(f"Starting batch processing from folder: {BATCH_FOLDER}")
            batch_results = detector.process_batch(BATCH_FOLDER)
            print(f"Batch processing completed!")
            print(f"Summary: {batch_results['summary']['batch_info']}")
            
        elif os.path.exists(IMAGE_PATH):
            print(f"Processing single image: {IMAGE_PATH}")
            result = detector.process_single_image(IMAGE_PATH)
            
            detector.annotate_detected_parts(IMAGE_PATH, result)

            gemini_analysis = result.get('gemini_analysis', {})
            if isinstance(gemini_analysis, dict):
                summary = gemini_analysis.get('summary', {})
                if summary:
                    print(f"Analysis Summary:")
                    print(f"   - Total objects detected: {summary.get('total_objects_detected', 'N/A')}")
                    print(f"   - Main components: {summary.get('main_components', 'N/A')}")
                    print(f"   - Complexity level: {summary.get('complexity_level', 'N/A')}")
            
            print("Single image processing completed!")
            
        else:
            print(f"ERROR: Image path '{IMAGE_PATH}' does not exist!")
            print("Please update the IMAGE_PATH variable in the script with a valid image path.")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        print(f"Error type: {type(e).__name__}")
    
    print(f"All results saved in '{detector.output_folder}' folder")
    print("Processing completed using Gemini AI only!")
    
    
"""
Your final JSON must look like this:

{
  "meta_data": {
    "drawing_number": "...",
    "material": "...",
    "component_weight": "...",
    "projection_method": "...",
    "drawing_type": "...",
    "title": "...",
    "notes": "...",
    "instructions": "...",
    "bom": [...]
  },
  "features": [
    {
      "type": "hole",
      "diameter": "85",
      "tolerance": "H7",
      "count": 2,
      "position": {"x": 200, "y": 300}
    },
    {
      "type": "slot",
      "length": "23",
      "width": "12",
      "radius": "6"
    },
    {
      "type": "chamfer",
      "angle": "45°",
      "length": "3"
    }
  ]
}
"""